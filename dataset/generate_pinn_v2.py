"""
PINN v2 — Synthetic Data Generator
====================================
Loads the trained PINN v2 checkpoint + scaler and generates synthetic
furnace data by:
  1. Sampling inputs from the real data distribution (with optional perturbation)
  2. Running PINN inference → OutletT, ExcessO2
  3. Adding small Gaussian noise (sensor-level)
  4. Saving to CSV

Usage:
    cd energy-lnn/dataset
    python generate_pinn_v2.py                              # default 500k rows
    python generate_pinn_v2.py --n_samples 1000000 --noise_T 0.3
"""

import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
import argparse
import os
import sys
import json
from datetime import datetime, timedelta

# ── Import model class from train script ─────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from train_pinn_v2 import FurnacePINN_v2, CONSTANTS


# ──────────────────────────────────────────────────────────────────────────
# 1. Load checkpoint + scaler
# ──────────────────────────────────────────────────────────────────────────
def load_pinn(checkpoint_dir: str = "checkpoints_v2", device: str = "cpu"):
    ckpt_path = os.path.join(checkpoint_dir, "best_pinn_v2.pth")
    scaler_path = os.path.join(checkpoint_dir, "scaler_v2.pkl")

    # Check existence
    for p in [ckpt_path, scaler_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    feature_cols = ckpt["feature_cols"]
    target_cols  = ckpt["target_cols"]
    var_T  = ckpt["var_T"]
    var_O2 = ckpt["var_O2"]

    # Target normalization stats (model predicts in z-scored space)
    y_mean = np.array(ckpt["y_mean"], dtype=np.float32)  # (2,)
    y_std  = np.array(ckpt["y_std"],  dtype=np.float32)  # (2,)

    # Reconstruct model
    model = FurnacePINN_v2(input_dim=len(feature_cols))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Effective physics params
    theta_eff = torch.nn.functional.softplus(model.theta_eff).item()
    afr_stoich = torch.nn.functional.softplus(model.afr_stoich).item()

    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print(f"✅ Loaded PINN v2 from {ckpt_path}")
    print(f"   Best epoch : {ckpt['epoch']}")
    print(f"   Val loss   : {ckpt['val_loss']:.6f}")
    print(f"   θ_eff (η·Cp): {theta_eff:.4f}")
    print(f"   AFR_stoich  : {afr_stoich:.4f}")
    print(f"   y_mean      : {y_mean}")
    print(f"   y_std       : {y_std}")
    print(f"   Features    : {feature_cols}")

    return model, scaler, feature_cols, target_cols, var_T, var_O2, y_mean, y_std


# ──────────────────────────────────────────────────────────────────────────
# 2. Sample input conditions
# ──────────────────────────────────────────────────────────────────────────
def sample_inputs(
    real_data_path: str,
    feature_cols: list,
    n_samples: int = 500_000,
    perturb_sigma: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample synthetic inputs by:
      - Drawing rows uniformly from real data (with replacement)
      - Adding Gaussian noise ∼ N(0, perturb_sigma × column_std)
      - Clipping to the observed min/max range (no impossible inputs)
    """
    rng = np.random.default_rng(seed)
    df_real = pd.read_csv(real_data_path)

    X_real = df_real[feature_cols].values  # (N, 7)
    col_std = X_real.std(axis=0)
    col_min = X_real.min(axis=0)
    col_max = X_real.max(axis=0)

    # Random draw with replacement
    idx = rng.integers(0, len(X_real), size=n_samples)
    X_sampled = X_real[idx].copy()

    # Perturb each column
    noise = rng.normal(0.0, 1.0, size=X_sampled.shape) * (col_std * perturb_sigma)
    X_perturbed = X_sampled + noise

    # Clip to observed range (no negative flows, no impossible temps)
    X_clipped = np.clip(X_perturbed, col_min, col_max)

    return pd.DataFrame(X_clipped, columns=feature_cols)


# ──────────────────────────────────────────────────────────────────────────
# 3. Run PINN inference
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def pinn_inference(
    model: FurnacePINN_v2,
    scaler,
    X_raw: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    batch_size: int = 4096,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run the trained PINN on raw inputs → (OutletT, ExcessO2).
    Model outputs normalised predictions; we inverse-transform here.
    """
    # Scale inputs
    X_scaled = scaler.transform(X_raw)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    preds_list = []
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i : i + batch_size]
        preds_n = model(batch)  # normalised space
        preds_list.append(preds_n.cpu().numpy())

    preds_n = np.concatenate(preds_list, axis=0)  # (N, 2)
    # Inverse-transform to raw scale
    return preds_n * y_std + y_mean


# ──────────────────────────────────────────────────────────────────────────
# 4. Generate and save
# ──────────────────────────────────────────────────────────────────────────
def generate(args):
    print("=" * 70)
    print("PINN v2 — SYNTHETIC DATA GENERATION")
    print("=" * 70)

    # Load model
    model, scaler, feature_cols, target_cols, var_T, var_O2, y_mean, y_std = load_pinn(
        checkpoint_dir=args.checkpoint_dir,
    )

    # Sample inputs
    print(f"\n📊 Sampling {args.n_samples:,} input conditions...")
    print(f"   Perturbation σ = {args.perturb_sigma} × column_std")
    X_synth = sample_inputs(
        real_data_path=args.real_data_path,
        feature_cols=feature_cols,
        n_samples=args.n_samples,
        perturb_sigma=args.perturb_sigma,
        seed=args.seed,
    )
    print(f"   ✅ Input samples: {X_synth.shape}")

    # PINN inference
    print(f"\n🔮 Running PINN inference...")
    preds = pinn_inference(
        model, scaler, X_synth.values, y_mean, y_std,
        batch_size=args.batch_size,
    )
    print(f"   ✅ Predictions: {preds.shape}")

    # Add sensor-level noise
    rng = np.random.default_rng(args.seed + 1)
    noise_T  = rng.normal(0.0, args.noise_T, size=preds.shape[0])
    noise_O2 = rng.normal(0.0, args.noise_O2, size=preds.shape[0])
    preds[:, 0] += noise_T
    preds[:, 1] += noise_O2

    # Assemble full DataFrame
    df_out = X_synth.copy()
    df_out[target_cols[0]] = preds[:, 0]  # OutletT
    df_out[target_cols[1]] = preds[:, 1]  # ExcessO2

    # Add synthetic timestamps (1-minute interval starting from 2025-01-01)
    start_time = datetime(2025, 1, 1)
    timestamps = [start_time + timedelta(minutes=i) for i in range(len(df_out))]
    df_out.insert(0, "Date", range(1, len(df_out) + 1))  # simple index like original

    # Reorder columns to match cleaned_furnace_data.csv format
    # Original: Date, InletT, OutletT, DraftP, OP_Damper, InletFlow, ExcessO2, FGFlow, FGPressure, Bridgewall
    col_order = ["Date", "InletT", "OutletT", "DraftP", "OP_Damper",
                 "InletFlow", "ExcessO2", "FGFlow", "FGPressure", "Bridgewall"]
    df_out = df_out[col_order]

    # Summary statistics
    print(f"\n📈 Generated data summary:")
    print(f"   Rows           : {len(df_out):,}")
    print(f"   OutletT  range : [{df_out['OutletT'].min():.1f}, {df_out['OutletT'].max():.1f}] °C")
    print(f"   OutletT  mean  : {df_out['OutletT'].mean():.1f} °C  (real: ~323.8)")
    print(f"   OutletT  std   : {df_out['OutletT'].std():.2f} °C  (real: ~5.4)")
    print(f"   ExcessO2 range : [{df_out['ExcessO2'].min():.2f}, {df_out['ExcessO2'].max():.2f}] %")
    print(f"   ExcessO2 mean  : {df_out['ExcessO2'].mean():.2f} %  (real: ~3.75)")
    print(f"   ExcessO2 std   : {df_out['ExcessO2'].std():.3f} %  (real: ~0.78)")

    # Compare real vs synthetic distributions
    df_real = pd.read_csv(args.real_data_path)
    print(f"\n📊 Distribution comparison (real → synthetic):")
    for col in ["OutletT", "ExcessO2"] + feature_cols:
        r_mean, r_std = df_real[col].mean(), df_real[col].std()
        s_mean, s_std = df_out[col].mean(), df_out[col].std()
        print(f"   {col:12s}  real: {r_mean:9.2f} ± {r_std:7.2f}  "
              f"synth: {s_mean:9.2f} ± {s_std:7.2f}")

    # Save
    output_path = args.output_path
    df_out.to_csv(output_path, index=False)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n💾 Saved: {output_path} ({size_mb:.1f} MB)")

    # Also save generation metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "n_samples": args.n_samples,
        "perturb_sigma": args.perturb_sigma,
        "noise_T": args.noise_T,
        "noise_O2": args.noise_O2,
        "seed": args.seed,
        "checkpoint": os.path.join(args.checkpoint_dir, "best_pinn_v2.pth"),
        "real_data": args.real_data_path,
        "theta_eff": torch.nn.functional.softplus(model.theta_eff).item(),
        "afr_stoich": torch.nn.functional.softplus(model.afr_stoich).item(),
        "stats": {
            "outlet_T_mean": float(df_out["OutletT"].mean()),
            "outlet_T_std": float(df_out["OutletT"].std()),
            "excess_O2_mean": float(df_out["ExcessO2"].mean()),
            "excess_O2_std": float(df_out["ExcessO2"].std()),
        },
    }
    meta_path = output_path.replace(".csv", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"📝 Metadata: {meta_path}")
    print("\n✅ Done.")

    return df_out


# ──────────────────────────────────────────────────────────────────────────
# 5. CLI
# ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="PINN v2 — Synthetic Data Generator")
    p.add_argument("--real_data_path", default="cleaned_furnace_data.csv",
                   help="Path to the cleaned real data")
    p.add_argument("--checkpoint_dir", default="checkpoints_v2",
                   help="Directory containing best_pinn_v2.pth and scaler_v2.pkl")
    p.add_argument("--output_path", default="synthetic_furnace_v2.csv",
                   help="Output path for generated data")
    p.add_argument("--n_samples", type=int, default=500_000,
                   help="Number of synthetic rows to generate")
    p.add_argument("--perturb_sigma", type=float, default=0.5,
                   help="Input perturbation: σ × column_std")
    p.add_argument("--noise_T", type=float, default=0.5,
                   help="Sensor noise on OutletT (°C)")
    p.add_argument("--noise_O2", type=float, default=0.05,
                   help="Sensor noise on ExcessO2 (%)")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args)
