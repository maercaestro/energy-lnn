"""
PINN v2 — Physics-Informed Neural Network for Single-Feedstock Furnace
========================================================================
Trained on cleaned real plant data (cleaned_furnace_data.csv).

Key improvements over v1:
  1. Per-output loss normalisation (by target variance)
  2. Learnable θ_eff = η·Cp (fold efficiency + heat capacity)
  3. Nonlinear O2 model: small MLP maps (DraftP, OP_Damper, FGFlow) → air_supply
     instead of single k·√(Draft)
  4. Bridgewall as intermediate physics node
  5. Single feedstock → single operating mode → stable learned parameters

Usage:
    cd energy-lnn/dataset
    python train_pinn_v2.py --no_wandb                    # local quick run
    python train_pinn_v2.py --epochs 3000 --lr 1e-3       # full run with WandB
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import argparse
import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import logging

# ──────────────────────────────────────────────────────────────────────────
# 1. Logging
# ──────────────────────────────────────────────────────────────────────────
def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'pinn_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# 2. Physics constants
# ──────────────────────────────────────────────────────────────────────────
CONSTANTS = {
    "LHV": 50_000.0,          # kJ/kg  (clean natural gas / methane, LHV)
    "AFR_STOICH": 17.2,       # kg air / kg fuel (methane-dominant)
    "RHO_FUEL": 0.72,         # kg/Nm³ (natural gas at STP, CH₄ = 0.717)
    "RHO_DIESEL": 850.0,      # kg/m³  (diesel, 0.84-0.86 range)
    "BBL_TO_M3": 0.159,       # m³/bbl (1 US oil barrel = 0.159 m³)
}
# InletFlow is in kbbl/day → kg/s:  ×1000 bbl/kbbl × 0.159 m³/bbl × 850 kg/m³ ÷ 86400 s/day
KBBL_DAY_TO_KG_S = (1000.0 * CONSTANTS["BBL_TO_M3"] * CONSTANTS["RHO_DIESEL"]
                    / 86400.0)   # ≈ 1.564 kg/s per kbbl/day


# ──────────────────────────────────────────────────────────────────────────
# 3. PINN Model
# ──────────────────────────────────────────────────────────────────────────
class FurnacePINN_v2(nn.Module):
    """
    Inputs  (7):  InletT, InletFlow, FGFlow, DraftP, FGPressure, Bridgewall, OP_Damper
    Outputs (2):  OutletT, ExcessO2
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 64, n_layers: int = 4,
                 dropout: float = 0.3):
        super().__init__()

        # ── Main prediction network ──────────────────────────────────
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

        # ── Air-supply sub-network ───────────────────────────────────
        # Maps (DraftP, OP_Damper, FGFlow) → effective air supply (scalar)
        # Replaces the single k·√(DraftP) from v1
        self.air_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Softplus(),          # air supply must be positive
        )

        # ── Learnable physics: θ_eff = η · Cp ───────────────────────
        # Stored as raw (unconstrained) parameter; θ_eff = softplus(self.theta_eff)
        # guarantees strict positivity.
        # With correct kbbl/day units, balance θ ≈ 0.7 (η·Cp lumped).
        # Init raw = 0.0 → softplus(0) = ln(2) ≈ 0.693, near balance.
        self.theta_eff = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Optional: learnable stoichiometric ratio (backup if default is off)
        self.afr_stoich = nn.Parameter(torch.tensor(CONSTANTS["AFR_STOICH"], dtype=torch.float32))

    def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:
        """Returns (B, 2): [pred_OutletT, pred_ExcessO2]"""
        return self.net(x_scaled)

    def air_supply(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        Estimate air supply from raw (un-scaled) DraftP, OP_Damper, FGFlow.

        x_raw columns: [InletT, InletFlow, FGFlow, DraftP, FGPressure, Bridgewall, OP_Damper]
        """
        draft   = x_raw[:, 3:4]     # DraftP
        damper  = x_raw[:, 6:7]     # OP_Damper
        fgflow  = x_raw[:, 2:3]     # FGFlow
        air_in  = torch.cat([draft, damper, fgflow], dim=1)   # (B, 3)
        return self.air_net(air_in).squeeze(-1)               # (B,)


# ──────────────────────────────────────────────────────────────────────────
# 4. Physics Loss (per-output normalised)
# ──────────────────────────────────────────────────────────────────────────
def physics_loss_v2(
    model: FurnacePINN_v2,
    preds: torch.Tensor,      # (B, 2)
    x_raw: torch.Tensor,      # (B, 7) un-scaled
    var_T: float,
    var_O2: float,
    w_energy: float = 1.0,
    w_mass: float = 1.0,
) -> tuple[torch.Tensor, float, float]:
    """
    Energy balance residual + mass balance residual.

    Energy balance:
        m_fuel · LHV · θ_eff ≈ m_proc · (T_out - T_in)
        (θ_eff absorbs η and Cp)

    Mass balance (O2):
        λ = air_supply / (m_fuel · AFR_stoich)
        O2_calc = 21 · (λ - 1) / λ
    """
    pred_T  = preds[:, 0]
    pred_O2 = preds[:, 1]

    # Raw inputs (NOT scaled)
    InletT    = x_raw[:, 0]
    InletFlow = x_raw[:, 1]
    FGFlow    = x_raw[:, 2]

    # Clamp for stability
    FGFlow_c    = torch.clamp(FGFlow, min=50.0)
    InletFlow_c = torch.clamp(InletFlow, min=50.0)

    # Mass flows with correct unit conversions
    # InletFlow: kbbl/day → kg/s
    m_proc = InletFlow_c * KBBL_DAY_TO_KG_S                         # kg/s
    # FGFlow: Nm³/hr → kg/s
    m_fuel = (FGFlow_c * CONSTANTS["RHO_FUEL"]) / 3600.0            # kg/s

    # θ_eff must be strictly positive: apply softplus to raw parameter
    theta_pos = torch.nn.functional.softplus(model.theta_eff)

    # Energy balance: Q_in ≈ Q_out
    Q_in  = m_fuel * CONSTANTS["LHV"]                               # kJ/s
    # θ_eff = η · Cp, so Q_out = m_proc · θ_eff · ΔT
    Q_out = m_proc * theta_pos * (pred_T - InletT)                  # kJ/s

    # Normalise by mean Q_in² → residual dimensionless, ~1 at a good solution
    q_scale = (Q_in.detach() ** 2).mean().clamp(min=1e-6)
    res_energy = torch.mean((Q_in - Q_out) ** 2) / q_scale

    # Mass balance (O2)
    air_supply = model.air_supply(x_raw)                            # (B,)
    afr_pos = torch.nn.functional.softplus(model.afr_stoich)        # keep AFR > 0
    lam_raw = air_supply / (m_fuel * afr_pos + 1e-6)

    # Soft lower bound at 1.01 — gradient ALWAYS flows (hard clamp killed it)
    # softplus(x-1.01)+1.01 ≈ x when x>>1.01, ≈ 1.7 when x=1.01, ≈ 1.35 when x=0
    lam = 1.01 + torch.nn.functional.softplus(lam_raw - 1.01)

    O2_calc = 21.0 * (lam - 1.0) / lam

    # Normalise by mean(O2_calc²) — same self-scaling approach as energy
    o2_scale = (O2_calc.detach() ** 2).mean().clamp(min=1e-6)
    res_mass = torch.mean((pred_O2 - O2_calc) ** 2) / o2_scale

    total = w_energy * res_energy + w_mass * res_mass
    return total, res_energy.item(), res_mass.item()


# ──────────────────────────────────────────────────────────────────────────
# 5. Metrics
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def calculate_metrics(
    model: FurnacePINN_v2,
    X_scaled: torch.Tensor,
    y_true: torch.Tensor,
    device: str = "cpu",
) -> dict:
    model.eval()
    X_s = X_scaled.to(device)
    y_t = y_true.to(device)
    preds = model(X_s)

    mse_T  = nn.MSELoss()(preds[:, 0], y_t[:, 0])
    mse_O2 = nn.MSELoss()(preds[:, 1], y_t[:, 1])
    rmse_T  = torch.sqrt(mse_T)
    rmse_O2 = torch.sqrt(mse_O2)
    mae_T   = (preds[:, 0] - y_t[:, 0]).abs().mean()
    mae_O2  = (preds[:, 1] - y_t[:, 1]).abs().mean()

    ss_tot_T  = ((y_t[:, 0] - y_t[:, 0].mean()) ** 2).sum()
    ss_res_T  = ((y_t[:, 0] - preds[:, 0]) ** 2).sum()
    r2_T = 1 - ss_res_T / (ss_tot_T + 1e-8)

    ss_tot_O2 = ((y_t[:, 1] - y_t[:, 1].mean()) ** 2).sum()
    ss_res_O2 = ((y_t[:, 1] - preds[:, 1]) ** 2).sum()
    r2_O2 = 1 - ss_res_O2 / (ss_tot_O2 + 1e-8)

    return {
        "rmse_T": rmse_T.item(),
        "rmse_O2": rmse_O2.item(),
        "mae_T": mae_T.item(),
        "mae_O2": mae_O2.item(),
        "r2_T": r2_T.item(),
        "r2_O2": r2_O2.item(),
        "theta_eff": torch.nn.functional.softplus(model.theta_eff).item(),
        "afr_stoich": torch.nn.functional.softplus(model.afr_stoich).item(),
    }


# ──────────────────────────────────────────────────────────────────────────
# 6. Training
# ──────────────────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = setup_logging(args.log_dir)
    logger.info("=" * 70)
    logger.info("FURNACE PINN v2 TRAINING")
    logger.info("=" * 70)

    # ── WandB ─────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"pinn_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        config = wandb.config
    else:
        config = args
        logger.info("WandB disabled — running locally.")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    # ── Data loading ──────────────────────────────────────────────────
    data_path = config.data_path if hasattr(config, "data_path") else args.data_path
    logger.info(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Total rows: {len(df):,}")

    # Chronological split: 70/15/15
    n = len(df)
    i_train = int(n * 0.70)
    i_val   = int(n * 0.85)

    train_df = df.iloc[:i_train].copy()
    val_df   = df.iloc[i_train:i_val].copy()
    test_df  = df.iloc[i_val:].copy()
    logger.info(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    feature_cols = ["InletT", "InletFlow", "FGFlow", "DraftP", "FGPressure", "Bridgewall", "OP_Damper"]
    target_cols  = ["OutletT", "ExcessO2"]

    # Verify all columns exist
    missing = set(feature_cols + target_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Scaler (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_df[feature_cols])
    X_val_s   = scaler.transform(val_df[feature_cols])
    X_test_s  = scaler.transform(test_df[feature_cols])

    # Raw (un-scaled) feature arrays for physics loss
    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    X_val_raw   = val_df[feature_cols].values.astype(np.float32)
    X_test_raw  = test_df[feature_cols].values.astype(np.float32)

    y_train = train_df[target_cols].values.astype(np.float32)
    y_val   = val_df[target_cols].values.astype(np.float32)
    y_test  = test_df[target_cols].values.astype(np.float32)

    # Target variance (for per-output normalisation)
    var_T  = float(np.var(y_train[:, 0]))
    var_O2 = float(np.var(y_train[:, 1]))
    logger.info(f"Target variance — OutletT: {var_T:.4f}  ExcessO2: {var_O2:.4f}")

    # Save scaler
    scaler_path = os.path.join(args.checkpoint_dir, "scaler_v2.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved: {scaler_path}")

    # Tensors
    def to_tensor(*arrays):
        return [torch.tensor(a, dtype=torch.float32).to(device) for a in arrays]

    X_tr_s, X_tr_r, y_tr = to_tensor(X_train_s, X_train_raw, y_train)
    X_va_s, X_va_r, y_va = to_tensor(X_val_s, X_val_raw, y_val)
    X_te_s, X_te_r, y_te = to_tensor(X_test_s, X_test_raw, y_test)

    # DataLoader
    bs = getattr(config, "batch_size", args.batch_size)
    train_ds = TensorDataset(X_tr_s, X_tr_r, y_tr)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)

    # ── Model ─────────────────────────────────────────────────────────
    hidden = getattr(config, "hidden_dim", args.hidden_dim)
    n_layers = getattr(config, "n_layers", args.n_layers)
    dropout = getattr(config, "dropout", args.dropout)
    model = FurnacePINN_v2(
        input_dim=len(feature_cols),
        hidden_dim=hidden,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {total_params:,}")

    lr = getattr(config, "lr", args.lr)

    # Separate param groups: MLP gets weight-decay, physics params do NOT
    # (weight-decay drives unused params to 0 during warmup)
    physics_param_names = {"theta_eff", "afr_stoich"}
    mlp_params = []
    phys_params = []   # theta_eff, afr_stoich, and air_net
    for name, param in model.named_parameters():
        if any(pn in name for pn in physics_param_names) or "air_net" in name:
            phys_params.append(param)
        else:
            mlp_params.append(param)
    logger.info(f"Param groups — MLP: {sum(p.numel() for p in mlp_params):,}  "
                f"Physics: {sum(p.numel() for p in phys_params):,}")

    optimizer = optim.Adam([
        {"params": mlp_params,  "weight_decay": 1e-3},
        {"params": phys_params, "weight_decay": 0.0},
    ], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100
    )

    epochs = getattr(config, "epochs", args.epochs)
    w_data    = getattr(config, "w_data",    args.w_data)
    w_physics = getattr(config, "w_physics", args.w_physics)
    w_energy  = getattr(config, "w_energy",  args.w_energy)
    w_mass    = getattr(config, "w_mass",    args.w_mass)
    warmup_epochs = getattr(config, "warmup_epochs", args.warmup_epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    patience_limit = getattr(config, "patience", args.patience)

    # ── Training loop ─────────────────────────────────────────────────
    logger.info(f"\nStarting training — {epochs} epochs")
    logger.info(f"  w_data={w_data}  w_physics={w_physics}  w_energy={w_energy}  w_mass={w_mass}  warmup_epochs={warmup_epochs}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0
        epoch_res_e = 0.0
        epoch_res_m = 0.0
        n_batches = 0

        for x_s, x_r, y_b in train_loader:
            optimizer.zero_grad()

            preds = model(x_s)

            # Data loss: per-output normalised MSE
            loss_T  = nn.MSELoss()(preds[:, 0], y_b[:, 0]) / (var_T + 1e-8)
            loss_O2 = nn.MSELoss()(preds[:, 1], y_b[:, 1]) / (var_O2 + 1e-8)
            data_loss = loss_T + loss_O2

            # Physics loss — gated by warmup
            phys_loss, res_e, res_m = physics_loss_v2(
                model, preds, x_r, var_T, var_O2,
                w_energy=w_energy, w_mass=w_mass,
            )

            # Linear ramp: 0 during warmup, then ramp 0→w_physics over ramp_len epochs
            ramp_len = 100
            if epoch <= warmup_epochs:
                w_phys_eff = 0.0
            elif epoch <= warmup_epochs + ramp_len:
                w_phys_eff = w_physics * (epoch - warmup_epochs) / ramp_len
            else:
                w_phys_eff = w_physics
            total_loss = w_data * data_loss + w_phys_eff * phys_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_data_loss += data_loss.item()
            epoch_phys_loss += phys_loss.item()
            epoch_res_e += res_e
            epoch_res_m += res_m
            n_batches += 1

        avg_data = epoch_data_loss / n_batches
        avg_phys = epoch_phys_loss / n_batches
        avg_res_e = epoch_res_e / n_batches
        avg_res_m = epoch_res_m / n_batches

        # ── Validation (data + physics so physics improvements count) ──
        model.eval()
        with torch.no_grad():
            v_preds = model(X_va_s)
            v_loss_T  = nn.MSELoss()(v_preds[:, 0], y_va[:, 0]) / (var_T + 1e-8)
            v_loss_O2 = nn.MSELoss()(v_preds[:, 1], y_va[:, 1]) / (var_O2 + 1e-8)
            val_data = (v_loss_T + v_loss_O2).item()

            v_phys, _, _ = physics_loss_v2(
                model, v_preds, X_va_r, var_T, var_O2,
                w_energy=w_energy, w_mass=w_mass,
            )
            val_loss = val_data + w_phys_eff * v_phys.item()

        scheduler.step(val_loss)

        # Logging
        if epoch % 50 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:>5}/{epochs}  "
                f"data={avg_data:.4f}  phys={avg_phys:.4f}  (E={avg_res_e:.4f} M={avg_res_m:.4f})  "
                f"val={val_loss:.4f}  "
                f"θ_eff={torch.nn.functional.softplus(model.theta_eff).item():.4f}  "
                f"AFR_s={torch.nn.functional.softplus(model.afr_stoich).item():.2f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/data_loss": avg_data,
                "train/phys_loss": avg_phys,
                "train/total_loss": avg_data + avg_phys,
                "val/loss": val_loss,
                "params/theta_eff": torch.nn.functional.softplus(model.theta_eff).item(),
                "params/afr_stoich": torch.nn.functional.softplus(model.afr_stoich).item(),
                "lr": optimizer.param_groups[0]["lr"],
            })

        # Reset patience when warmup ends — give physics a fair shot
        if epoch == warmup_epochs + 1:
            best_val_loss = float("inf")
            patience_counter = 0
            logger.info("  [warmup ended → patience & best_val reset]")

        # Early stopping + checkpoint
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(args.checkpoint_dir, "best_pinn_v2.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "theta_eff": model.theta_eff.item(),
                "afr_stoich": model.afr_stoich.item(),
                "var_T": var_T,
                "var_O2": var_O2,
                "feature_cols": feature_cols,
                "target_cols": target_cols,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # ── Test evaluation ───────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("TEST EVALUATION")
    logger.info("=" * 70)

    # Load best model
    ckpt_path = os.path.join(args.checkpoint_dir, "best_pinn_v2.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best model from epoch {ckpt['epoch']}")

    metrics = calculate_metrics(model, X_te_s, y_te, device)
    logger.info(f"  RMSE  OutletT : {metrics['rmse_T']:.4f} °C")
    logger.info(f"  RMSE  ExcessO2: {metrics['rmse_O2']:.4f} %")
    logger.info(f"  MAE   OutletT : {metrics['mae_T']:.4f} °C")
    logger.info(f"  MAE   ExcessO2: {metrics['mae_O2']:.4f} %")
    logger.info(f"  R²    OutletT : {metrics['r2_T']:.4f}")
    logger.info(f"  R²    ExcessO2: {metrics['r2_O2']:.4f}")
    logger.info(f"  θ_eff (η·Cp)  : {metrics['theta_eff']:.4f}")
    logger.info(f"  AFR_stoich    : {metrics['afr_stoich']:.4f}")

    if use_wandb:
        import wandb
        wandb.log({f"test/{k}": v for k, v in metrics.items()})
        wandb.finish()

    # Save metrics
    metrics_path = os.path.join(args.checkpoint_dir, "test_metrics_v2.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nMetrics saved: {metrics_path}")
    logger.info("Done.")

    return metrics


# ──────────────────────────────────────────────────────────────────────────
# 7. CLI
# ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="PINN v2 — Single-Feedstock Furnace")
    # Data
    p.add_argument("--data_path", default="cleaned_furnace_data.csv")
    # Model
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--n_layers",   type=int, default=3)
    p.add_argument("--dropout",    type=float, default=0.3)
    # Training
    p.add_argument("--epochs",     type=int,   default=3000)
    p.add_argument("--batch_size", type=int,   default=512)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int,   default=300)
    # Loss weights
    p.add_argument("--w_data",     type=float, default=1.0)
    p.add_argument("--w_physics",  type=float, default=0.1,
                   help="Physics loss weight (active after warmup)")
    p.add_argument("--w_energy",   type=float, default=1.0)
    p.add_argument("--w_mass",     type=float, default=1.0)
    p.add_argument("--warmup_epochs", type=int, default=20,
                   help="Short warmup: data-only so MLP is non-random before physics")
    # Infra
    p.add_argument("--checkpoint_dir", default="checkpoints_v2")
    p.add_argument("--log_dir",        default="logs_v2")
    p.add_argument("--no_wandb",       action="store_true")
    p.add_argument("--wandb_project",  default="furnace-pinn-v2")
    p.add_argument("--wandb_entity",   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
