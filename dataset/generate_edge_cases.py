"""
PINN v2 — Edge-Case Augmentation Generator
=============================================
Generates physics-guided edge-case scenarios that are absent from
the real furnace data, using the trained PINN v2 as a surrogate.

Scenarios:
  S1  Flame-Out / Rich Combustion    (low O2)
  S2  Air Leak / Excessive Draft     (high O2)
  S3  Tube Rupture / Overheating     (high ΔT)
  S4  Positive Furnace Pressure      (blowback)
  S5  Fuel Trip / Cold Furnace       (loss of firing)
  S6  Fuel Gas Contamination         (lower effective LHV)

Features:
  - AR(1) coloured noise (autocorrelated, not i.i.d. white noise)
  - Mahalanobis distance confidence flag
  - Latin Hypercube Sampling within each scenario envelope
  - 5-column EBLNN derivation (fuel_flow, air_fuel_ratio,
    current_temp, inflow_temp, inflow_rate)

Usage:
    python generate_edge_cases.py                    # default 25k per scenario
    python generate_edge_cases.py --n_per_scenario 50000
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import os
import sys
import json
from datetime import datetime
from scipy.stats import qmc            # Latin Hypercube Sampling

sys.path.insert(0, os.path.dirname(__file__))
from train_pinn_v2 import FurnacePINN_v2, CONSTANTS, KBBL_DAY_TO_KG_S


# ──────────────────────────────────────────────────────────────────────────
# 0. Column order (must match train_pinn_v2.py)
# ──────────────────────────────────────────────────────────────────────────
FEATURE_COLS = ["InletT", "InletFlow", "FGFlow", "DraftP",
                "FGPressure", "Bridgewall", "OP_Damper"]
TARGET_COLS  = ["OutletT", "ExcessO2"]

# Index mapping for readability
_I = {c: i for i, c in enumerate(FEATURE_COLS)}


# ──────────────────────────────────────────────────────────────────────────
# 1. Load trained PINN (learned θ_eff, AFR_stoich, air_net)
# ──────────────────────────────────────────────────────────────────────────
def load_pinn(checkpoint_dir="checkpoints_v2", device="cpu"):
    ckpt_path   = os.path.join(checkpoint_dir, "best_pinn_v2.pth")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FurnacePINN_v2(input_dim=len(ckpt["feature_cols"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    print(f"✅ PINN v2 loaded (epoch {ckpt['epoch']}, "
          f"θ_eff={torch.nn.functional.softplus(model.theta_eff).item():.4f}, "
          f"AFR_s={torch.nn.functional.softplus(model.afr_stoich).item():.2f})")
    return model


# ──────────────────────────────────────────────────────────────────────────
# 2. Estimate AR(1) noise parameters from real data
# ──────────────────────────────────────────────────────────────────────────
def estimate_ar1_params(real_data_path: str):
    """
    Fit AR(1) per column on the residuals (real − rolling mean).
    Returns phi (autocorrelation coeff) and sigma (innovation std) per column.
    """
    df = pd.read_csv(real_data_path)
    all_cols = FEATURE_COLS + TARGET_COLS
    phi_dict, sigma_dict = {}, {}

    for col in all_cols:
        x = df[col].values.astype(np.float64)
        # Residuals: subtract 60-minute rolling mean to isolate noise
        trend = pd.Series(x).rolling(60, min_periods=1, center=True).mean().values
        resid = x - trend
        # AR(1) parameter: phi = corr(resid[t], resid[t-1])
        r = resid[~np.isnan(resid)]
        if len(r) > 2:
            phi = np.corrcoef(r[:-1], r[1:])[0, 1]
            phi = np.clip(phi, 0.0, 0.999)  # ensure stable
            innovation_std = np.std(r) * np.sqrt(1 - phi ** 2)
        else:
            phi, innovation_std = 0.0, 0.01
        phi_dict[col] = phi
        sigma_dict[col] = innovation_std

    print("\n📊 AR(1) noise parameters (from real data):")
    for col in all_cols:
        print(f"   {col:12s}  φ={phi_dict[col]:.4f}  σ_innov={sigma_dict[col]:.4f}")
    return phi_dict, sigma_dict


def generate_ar1_noise(n: int, phi: float, sigma: float, rng: np.random.Generator):
    """Generate n samples of AR(1) coloured noise."""
    noise = np.zeros(n)
    noise[0] = rng.normal(0, sigma / np.sqrt(1 - phi ** 2 + 1e-12))
    for t in range(1, n):
        noise[t] = phi * noise[t - 1] + sigma * rng.normal()
    return noise


# ──────────────────────────────────────────────────────────────────────────
# 3. Mahalanobis distance for confidence flagging
# ──────────────────────────────────────────────────────────────────────────
def fit_mahalanobis(real_data_path: str):
    """Compute mean and inverse covariance of real feature distribution."""
    df = pd.read_csv(real_data_path)
    X = df[FEATURE_COLS].values
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    # Regularise covariance for numerical stability
    cov += np.eye(len(FEATURE_COLS)) * 1e-6
    cov_inv = np.linalg.inv(cov)
    return mu, cov_inv


def compute_mahal_distances(X: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray):
    """Vectorised Mahalanobis distance for each row."""
    diff = X - mu
    # d² = diff @ cov_inv @ diff.T  (diagonal)
    left = diff @ cov_inv
    d_sq = np.sum(left * diff, axis=1)
    return np.sqrt(np.clip(d_sq, 0, None))


# ──────────────────────────────────────────────────────────────────────────
# 4. Latin Hypercube Sampling within bounds
# ──────────────────────────────────────────────────────────────────────────
def lhs_sample(bounds: dict, n: int, seed: int) -> np.ndarray:
    """
    bounds: {col_name: (low, high)} for each of the 7 feature columns.
    Returns (n, 7) array with LHS samples.
    """
    d = len(FEATURE_COLS)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit_cube = sampler.random(n)  # (n, d) in [0, 1]

    X = np.zeros((n, d))
    for j, col in enumerate(FEATURE_COLS):
        lo, hi = bounds[col]
        X[:, j] = lo + (hi - lo) * unit_cube[:, j]
    return X


# ──────────────────────────────────────────────────────────────────────────
# 5. Scenario definitions
# ──────────────────────────────────────────────────────────────────────────
def get_real_stats(real_data_path: str):
    """Get basic stats from real data for bounding."""
    df = pd.read_csv(real_data_path)
    stats = {}
    for col in FEATURE_COLS:
        stats[col] = {
            "min": df[col].min(), "max": df[col].max(),
            "mean": df[col].mean(), "std": df[col].std(),
            "p01": df[col].quantile(0.01), "p99": df[col].quantile(0.99),
        }
    return stats


def define_scenarios(stats: dict) -> dict:
    """
    Each scenario returns a dict of LHS bounds per feature column.
    Columns not specified default to the real p01–p99 range.
    """
    def default_bounds():
        return {col: (stats[col]["p01"], stats[col]["p99"]) for col in FEATURE_COLS}

    scenarios = {}

    # S1 — Flame-Out / Rich Combustion (low O2)
    # High fuel, restricted air
    s1 = default_bounds()
    s1["FGFlow"]    = (1800.0, 2500.0)     # well above normal max ~1800
    s1["OP_Damper"] = (36.0, 42.0)         # restricted damper
    s1["DraftP"]    = (-2.0, 0.5)          # weak draft
    scenarios["S1_flame_out"] = s1

    # S2 — Air Leak / Excessive Draft (high O2)
    # Damper wide open, strong draft, normal/low fuel
    s2 = default_bounds()
    s2["OP_Damper"] = (63.0, 68.0)         # wide open (real max ~67.5)
    s2["DraftP"]    = (-12.0, -9.0)        # very strong draft
    s2["FGFlow"]    = (800.0, 1200.0)      # low-to-normal firing
    scenarios["S2_air_leak"] = s2

    # S3 — Tube Rupture / Overheating (high ΔT, high Bridgewall)
    # Low process flow, high firing
    s3 = default_bounds()
    s3["InletFlow"]  = (245.0, 310.0)      # reduced process flow
    s3["FGFlow"]     = (1600.0, 2000.0)    # high firing
    s3["Bridgewall"] = (700.0, 800.0)      # extending beyond real max ~723
    scenarios["S3_tube_rupture"] = s3

    # S4 — Positive Furnace Pressure (blowback)
    # DraftP near zero or positive, restricted damper
    s4 = default_bounds()
    s4["DraftP"]    = (-0.5, 3.0)          # positive pressure zone
    s4["OP_Damper"] = (36.0, 45.0)         # restricted
    scenarios["S4_positive_pressure"] = s4

    # S5 — Fuel Trip / Cold Furnace (loss of firing)
    # Very low or zero fuel flow
    s5 = default_bounds()
    s5["FGFlow"]    = (0.0, 500.0)         # near-zero firing
    s5["FGPressure"] = (0.2, 0.8)          # low fuel pressure (supply issue)
    scenarios["S5_fuel_trip"] = s5

    # S6 — Fuel Gas Contamination (lower effective LHV)
    # Normal FGFlow by volume, but energy content is 60–85% of clean gas.
    # We simulate by scaling FGFlow down (the PINN learned FGFlow → energy).
    # A real 1400 Nm³/hr of contaminated gas ≈ 840–1190 Nm³/hr of clean gas.
    s6 = default_bounds()
    s6["FGFlow"]     = (800.0, 1200.0)     # effective energy = contaminated
    s6["FGPressure"] = (0.8, 1.3)          # slightly low supply pressure
    # Everything else stays normal — that's the danger: operators see "normal" but
    # the furnace isn't getting enough energy.
    scenarios["S6_fuel_contamination"] = s6

    return scenarios


# ──────────────────────────────────────────────────────────────────────────
# 6. Physics-only predictions (analytical energy + mass balance)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def physics_predict(model, X_raw: np.ndarray, s_name: str,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Compute OutletT and ExcessO2 from analytical physics equations,
    using the PINN's learned θ_eff and a first-principles air-supply model.

    Energy balance:  T_out = T_in + (m_fuel · LHV · θ_eff) / m_proc
    Mass balance:    O2 = 21 · (λ − 1) / λ   where λ = m_air / (m_fuel · AFR_s)

    Air supply is estimated analytically (not from the learned air_net,
    which is unreliable outside its training distribution):
        m_air = k_base · (OP_Damper / 55) · √|DraftP|   [kg/s]
    where k_base is calibrated so that at normal conditions (DraftP=-4.4,
    OP_Damper=55, FGFlow=1398) the excess-air ratio λ ≈ 1.25 (≈ 5% O2,
    matching the real-data mean of 3.75% excess O2).
    """
    # --- Learned physics parameters ---
    theta_eff  = torch.nn.functional.softplus(model.theta_eff).item()
    afr_stoich = torch.nn.functional.softplus(model.afr_stoich).item()

    # --- Mass flows ---
    InletT    = X_raw[:, _I["InletT"]]
    InletFlow = np.clip(X_raw[:, _I["InletFlow"]], 50.0, None)    # kbbl/day
    FGFlow    = np.clip(X_raw[:, _I["FGFlow"]], 0.1, None)        # Nm³/hr
    DraftP    = X_raw[:, _I["DraftP"]]
    OP_Damper = X_raw[:, _I["OP_Damper"]]

    m_proc = InletFlow * KBBL_DAY_TO_KG_S                          # kg/s
    m_fuel = FGFlow * CONSTANTS["RHO_FUEL"] / 3600.0               # kg/s
    lhv    = CONSTANTS["LHV"]

    # --- Combustion efficiency factor for contamination ---
    # Contaminated gas: random LHV multiplier 0.60–0.85
    if "contamination" in s_name:
        lhv = lhv * rng.uniform(0.60, 0.85, size=len(X_raw))

    # --- Energy balance → OutletT ---
    Q_in    = m_fuel * lhv                                          # kW
    delta_T = (Q_in * theta_eff) / (m_proc + 1e-8)
    T_out   = InletT + delta_T

    # --- Analytical air-supply model ---
    # Calibrate k_base from normal operating point:
    #   Normal: DraftP=-4.4, OP_Damper=55, FGFlow=1398 Nm³/hr
    #   m_fuel_norm = 1398 × 0.72 / 3600 = 0.2796 kg/s
    #   At λ=1.22 (real O2 ≈ 3.75%):  m_air = 1.22 × 0.2796 × 17.47 = 5.96 kg/s
    #   k_base × (55/55) × √4.4 = 5.96  →  k_base = 5.96 / 2.098 = 2.841
    K_AIR_BASE = 2.841

    # Air supply: proportional to damper opening, proportional to √|draft|
    # If draft is positive (blowback), air supply drops sharply
    draft_abs = np.abs(DraftP)
    draft_sign = np.where(DraftP < 0, 1.0, -0.3)   # positive draft → reduced/reversed air
    m_air = K_AIR_BASE * (OP_Damper / 55.0) * np.sqrt(draft_abs + 0.01) * draft_sign
    m_air = np.clip(m_air, 0.0, None)               # can't be negative mass

    # --- Excess-air ratio λ ---
    lam = m_air / (m_fuel * afr_stoich + 1e-8)
    # Numerically stable soft lower bound: λ >= 1.0 (can't burn more than stoich)
    # Use stable softplus: softplus(x) = x if x > 20, else log1p(exp(x))
    shift = lam - 1.0
    lam_soft = np.where(shift > 20.0, shift, np.log1p(np.exp(np.clip(shift, -50, 20))))
    lam = 1.0 + lam_soft

    # ExcessO2 = 21 × (λ − 1) / λ
    O2 = 21.0 * (lam - 1.0) / lam

    # --- Clamp to physically plausible ranges ---
    T_out = np.clip(T_out, InletT, 500.0)     # can't be below inlet
    O2    = np.clip(O2, 0.0, 21.0)            # 0–21% range

    return np.column_stack([T_out, O2])


# ──────────────────────────────────────────────────────────────────────────
# 7. Derive 5-column EBLNN features
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def derive_eblnn_cols(model, X_raw: np.ndarray, preds: np.ndarray):
    """
    Derive 5 EBLNN columns from 7 PINN features + 2 PINN outputs.

    EBLNN cols:
      fuel_flow       = FGFlow [Nm³/hr] × ρ_fuel / 3600  → kg/s
      air_fuel_ratio  = air_supply / fuel_mass_flow
      current_temp    = OutletT  (PINN prediction)
      inflow_temp     = InletT
      inflow_rate     = InletFlow [kbbl/day] × KBBL_DAY_TO_KG_S → kg/s
    """
    rho_fuel = CONSTANTS["RHO_FUEL"]

    # Fuel mass flow (kg/s)
    fg_flow_nm3h = X_raw[:, _I["FGFlow"]]
    fuel_flow_kgs = fg_flow_nm3h * rho_fuel / 3600.0

    # Air supply from PINN's air_net (DraftP, OP_Damper, FGFlow)
    x_raw_t = torch.tensor(X_raw, dtype=torch.float32)
    air_supply = model.air_supply(x_raw_t).numpy()  # (N,)

    # Air-fuel ratio
    afr = air_supply / (fuel_flow_kgs + 1e-8)

    # Temperatures
    current_temp = preds[:, 0]   # OutletT
    inflow_temp  = X_raw[:, _I["InletT"]]

    # Process inflow rate (kg/s)
    inflow_rate = X_raw[:, _I["InletFlow"]] * KBBL_DAY_TO_KG_S

    return np.column_stack([fuel_flow_kgs, afr, current_temp,
                            inflow_temp, inflow_rate])

EBLNN_COLS = ["fuel_flow", "air_fuel_ratio", "current_temp",
              "inflow_temp", "inflow_rate"]


# ──────────────────────────────────────────────────────────────────────────
# 8. Main generation pipeline
# ──────────────────────────────────────────────────────────────────────────
def generate(args):
    print("=" * 70)
    print("PINN v2 — EDGE-CASE AUGMENTATION GENERATOR")
    print("=" * 70)

    rng = np.random.default_rng(args.seed)

    # Load model (only need learned params + air_net, not MLP)
    model = load_pinn(args.checkpoint_dir)

    # AR(1) noise parameters
    phi_dict, sigma_dict = estimate_ar1_params(args.real_data_path)

    # Mahalanobis baseline
    mu_mah, cov_inv = fit_mahalanobis(args.real_data_path)

    # Real data stats for scenario bounds
    stats = get_real_stats(args.real_data_path)
    scenarios = define_scenarios(stats)

    all_frames = []
    n = args.n_per_scenario

    for s_name, bounds in scenarios.items():
        print(f"\n🔧 Scenario: {s_name}  ({n:,} samples)")

        # 1. LHS sampling of inputs
        X_raw = lhs_sample(bounds, n, seed=rng.integers(0, 2**31))

        # 2. Physics-only predictions (analytical, not MLP)
        preds = physics_predict(model, X_raw, s_name, rng)

        # 3. Add AR(1) coloured noise to all columns
        for j, col in enumerate(FEATURE_COLS):
            noise = generate_ar1_noise(n, phi_dict[col], sigma_dict[col], rng)
            X_raw[:, j] += noise

        for j, col in enumerate(TARGET_COLS):
            noise = generate_ar1_noise(n, phi_dict[col], sigma_dict[col], rng)
            preds[:, j] += noise

        # 4. Mahalanobis distance (confidence flag)
        mahal = compute_mahal_distances(X_raw, mu_mah, cov_inv)

        # 5. Derive 5 EBLNN columns
        eblnn = derive_eblnn_cols(model, X_raw, preds)

        # 6. Build DataFrame
        df = pd.DataFrame(X_raw, columns=FEATURE_COLS)
        df["OutletT"]   = preds[:, 0]
        df["ExcessO2"]  = preds[:, 1]
        df["scenario"]  = s_name
        df["mahal_dist"] = mahal
        df["confidence"] = np.where(mahal <= 3.0, "high",
                           np.where(mahal <= 5.0, "medium", "low"))

        # EBLNN derived columns
        for k, col_name in enumerate(EBLNN_COLS):
            df[col_name] = eblnn[:, k]

        all_frames.append(df)

        # Quick summary
        print(f"   OutletT : {preds[:,0].mean():.1f} ± {preds[:,0].std():.2f} °C")
        print(f"   ExcessO2: {preds[:,1].mean():.2f} ± {preds[:,1].std():.3f} %")
        print(f"   Mahal   : mean={mahal.mean():.1f}  "
              f"high={np.sum(mahal<=3)}/{n}  "
              f"med={np.sum((mahal>3)&(mahal<=5))}/{n}  "
              f"low={np.sum(mahal>5)}/{n}")

    # ── Combine all scenarios ─────────────────────────────────────────
    df_all = pd.concat(all_frames, ignore_index=True)
    total = len(df_all)

    print(f"\n{'='*70}")
    print(f"📊 COMBINED EDGE-CASE DATASET: {total:,} rows")
    print(f"{'='*70}")

    # Per-scenario summary
    for s_name in scenarios:
        sub = df_all[df_all.scenario == s_name]
        print(f"\n  {s_name}:")
        print(f"    OutletT  : {sub.OutletT.mean():7.1f} ± {sub.OutletT.std():5.2f}")
        print(f"    ExcessO2 : {sub.ExcessO2.mean():7.2f} ± {sub.ExcessO2.std():5.3f}")
        conf = sub.confidence.value_counts()
        print(f"    Confidence: {conf.to_dict()}")

    # ── Save full 7+2+metadata CSV ───────────────────────────────────
    out_path = args.output_path
    df_all.to_csv(out_path, index=False)
    sz = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n💾 Full edge-case CSV: {out_path} ({sz:.1f} MB)")

    # ── Save 5-col EBLNN-ready CSV ───────────────────────────────────
    # Columns: fuel_flow, air_fuel_ratio, current_temp, inflow_temp, inflow_rate,
    #          next_temp (= OutletT), next_excess_o2 (= ExcessO2),
    #          scenario, confidence
    eblnn_df = df_all[EBLNN_COLS + ["OutletT", "ExcessO2", "scenario", "confidence"]].copy()
    eblnn_df.rename(columns={"OutletT": "next_temp", "ExcessO2": "next_excess_o2"},
                    inplace=True)
    eblnn_path = out_path.replace(".csv", "_eblnn.csv")
    eblnn_df.to_csv(eblnn_path, index=False)
    sz2 = os.path.getsize(eblnn_path) / 1024 / 1024
    print(f"💾 EBLNN-ready CSV:   {eblnn_path} ({sz2:.1f} MB)")

    # ── Also convert real data to 5-col EBLNN format ─────────────────
    print(f"\n🔄 Converting real data to EBLNN format...")
    df_real = pd.read_csv(args.real_data_path)
    X_real_raw = df_real[FEATURE_COLS].values.astype(np.float32)
    preds_real = df_real[TARGET_COLS].values.astype(np.float32)
    eblnn_real = derive_eblnn_cols(model, X_real_raw, preds_real)

    real_eblnn_df = pd.DataFrame(eblnn_real, columns=EBLNN_COLS)
    real_eblnn_df["next_temp"]      = df_real["OutletT"].values
    real_eblnn_df["next_excess_o2"] = df_real["ExcessO2"].values
    real_eblnn_df["scenario"]       = "real"
    real_eblnn_df["confidence"]     = "high"

    real_eblnn_path = "real_furnace_eblnn.csv"
    real_eblnn_df.to_csv(real_eblnn_path, index=False)
    sz3 = os.path.getsize(real_eblnn_path) / 1024 / 1024
    print(f"💾 Real EBLNN CSV:    {real_eblnn_path} ({sz3:.1f} MB)")

    # ── Merged dataset (real + edge-cases) ────────────────────────────
    merged = pd.concat([real_eblnn_df, eblnn_df], ignore_index=True)
    merged_path = "merged_eblnn_dataset.csv"
    merged.to_csv(merged_path, index=False)
    sz4 = os.path.getsize(merged_path) / 1024 / 1024
    print(f"💾 Merged EBLNN CSV:  {merged_path} ({sz4:.1f} MB)")
    print(f"   Real: {len(real_eblnn_df):,}  Edge: {len(eblnn_df):,}  "
          f"Total: {len(merged):,}")

    # ── Metadata ─────────────────────────────────────────────────────
    meta = {
        "generated_at": datetime.now().isoformat(),
        "n_per_scenario": args.n_per_scenario,
        "total_edge_case_rows": total,
        "total_real_rows": len(df_real),
        "total_merged": len(merged),
        "scenarios": list(scenarios.keys()),
        "ar1_phi": {k: round(v, 4) for k, v in phi_dict.items()},
        "ar1_sigma": {k: round(v, 4) for k, v in sigma_dict.items()},
        "noise_type": "AR(1) coloured noise",
        "confidence_thresholds": {"high": "mahal <= 3", "medium": "3 < mahal <= 5", "low": "mahal > 5"},
        "files": {
            "full_edge_cases": out_path,
            "eblnn_edge_cases": eblnn_path,
            "eblnn_real": real_eblnn_path,
            "eblnn_merged": merged_path,
        },
    }
    meta_path = out_path.replace(".csv", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"📝 Metadata: {meta_path}")
    print("\n✅ Done.")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="PINN v2 — Edge-Case Augmentation")
    p.add_argument("--real_data_path",  default="cleaned_furnace_data.csv")
    p.add_argument("--checkpoint_dir",  default="checkpoints_v2")
    p.add_argument("--output_path",     default="edge_cases_v2.csv")
    p.add_argument("--n_per_scenario",  type=int, default=25_000)
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args)
