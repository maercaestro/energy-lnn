"""
PINN v2 — Edge-Case Augmentation Generator
=============================================
Legacy snapshot exporter, NOT a future-state forecast dataset.
Its current_temp and next_temp both refer to the same OutletT.
Use eblnn-benchmark/config/forecast_config.yaml for real-data forecasting.

Generates physics-guided edge-case scenarios that are absent from
the real furnace data, using physics parameters learned by PINN v2.

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
from collections import defaultdict
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

    # S1 — Flame-Out / Rich Combustion
    # Fuel remains available, combustion air is lost, and the cold-feed
    # envelope allows the outlet to fall below the engineering threshold.
    s1 = default_bounds()
    s1["InletT"]    = (180.0, 240.0)
    s1["FGFlow"]    = (1800.0, 2500.0)     # well above normal max ~1800
    s1["OP_Damper"] = (20.0, 38.0)         # restricted damper
    s1["DraftP"]    = (-1.0, 0.5)          # weak/reversed draft
    scenarios["S1_flame_out"] = s1

    # S2 — Air Leak / Excessive Draft (high O2)
    # Damper wide open, strong draft, normal/low fuel
    s2 = default_bounds()
    s2["OP_Damper"] = (63.0, 68.0)         # wide open (real max ~67.5)
    s2["DraftP"]    = (-12.0, -9.0)        # very strong draft
    s2["FGFlow"]    = (800.0, 1200.0)      # low-to-normal firing
    scenarios["S2_air_leak"] = s2

    # S3 — Tube Rupture / Overheating
    # Severe loss of process flow with sustained high firing.
    s3 = default_bounds()
    s3["InletFlow"]  = (30.0, 100.0)
    s3["FGFlow"]     = (3000.0, 5000.0)
    s3["Bridgewall"] = (900.0, 1100.0)
    scenarios["S3_tube_rupture"] = s3

    # S4 — Positive Furnace Pressure (blowback)
    # DraftP near zero or positive, restricted damper
    s4 = default_bounds()
    s4["DraftP"]    = (10.0, 15.0)         # engineering critical zone
    s4["OP_Damper"] = (36.0, 45.0)         # restricted
    scenarios["S4_positive_pressure"] = s4

    # S5 — Fuel Trip / Cold Furnace (loss of firing)
    # Very low or zero fuel flow
    s5 = default_bounds()
    s5["FGFlow"]    = (0.0, 50.0)
    s5["FGPressure"] = (0.0, 0.3)
    scenarios["S5_fuel_trip"] = s5

    # S6 — Fuel Gas Contamination (lower effective LHV)
    # Normal indicated fuel flow, but effective LHV is 60–85% of clean gas.
    s6 = default_bounds()
    s6["FGFlow"]     = (1200.0, 1600.0)
    s6["FGPressure"] = (0.8, 1.3)
    scenarios["S6_fuel_contamination"] = s6

    return scenarios


# ──────────────────────────────────────────────────────────────────────────
# 6. Physics-only predictions (analytical energy + mass balance)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def physics_predict(model, X_raw: np.ndarray, s_name: str,
                    rng: np.random.Generator) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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
    lhv_factor = np.ones(len(X_raw), dtype=np.float64)

    # Flame-out retains indicated fuel flow but releases little useful heat.
    if "flame_out" in s_name:
        lhv_factor = rng.uniform(0.0, 0.05, size=len(X_raw))
    # Contaminated gas retains indicated flow with 60–85% effective LHV.
    if "contamination" in s_name:
        lhv_factor = rng.uniform(0.60, 0.85, size=len(X_raw))
    lhv = CONSTANTS["LHV"] * lhv_factor

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
    lam_raw = m_air / (m_fuel * afr_stoich + 1e-8)
    lam = np.maximum(lam_raw, 1.0)

    # ExcessO2 = 21 × (λ − 1) / λ
    O2 = 21.0 * (lam - 1.0) / lam

    # --- Clamp to physically plausible ranges ---
    T_out = np.clip(T_out, InletT, 550.0)
    O2    = np.clip(O2, 0.0, 21.0)            # 0–21% range

    diagnostics = {
        "lambda": lam_raw,
        "lhv_factor": lhv_factor,
        "thermal_gain_ratio": lhv_factor.copy(),
        "analytical_air_fuel_ratio": m_air / (m_fuel + 1e-8),
    }
    return np.column_stack([T_out, O2]), diagnostics


def scenario_acceptance_mask(
    s_name: str,
    X_raw: np.ndarray,
    preds: np.ndarray,
    diagnostics: dict[str, np.ndarray],
) -> np.ndarray:
    """Engineering acceptance rules agreed for the final scenario dataset."""
    outlet = preds[:, 0]
    oxygen = preds[:, 1]
    delta_t = outlet - X_raw[:, _I["InletT"]]
    if s_name == "S1_flame_out":
        return (outlet < 250.0) & (oxygen < 1.0) & (X_raw[:, _I["FGFlow"]] > 1800.0)
    if s_name == "S2_air_leak":
        return (X_raw[:, _I["DraftP"]] < -9.0) & (oxygen > 6.0)
    if s_name == "S3_tube_rupture":
        return outlet > 460.0
    if s_name == "S4_positive_pressure":
        return X_raw[:, _I["DraftP"]] > 10.0
    if s_name == "S5_fuel_trip":
        return (
            (X_raw[:, _I["FGFlow"]] < 50.0)
            & (X_raw[:, _I["FGPressure"]] < 0.8)
            & (delta_t < 10.0)
        )
    if s_name == "S6_fuel_contamination":
        ratio = diagnostics["thermal_gain_ratio"]
        return (ratio >= 0.60) & (ratio <= 0.85)
    raise ValueError(f"Unknown scenario {s_name!r}.")


# ──────────────────────────────────────────────────────────────────────────
# 7. Derive 5-column EBLNN features
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def derive_eblnn_cols(
    model,
    X_raw: np.ndarray,
    preds: np.ndarray,
    analytical_air_fuel_ratio: np.ndarray | None = None,
):
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
    if analytical_air_fuel_ratio is None:
        x_raw_t = torch.tensor(X_raw, dtype=torch.float32)
        air_supply = model.air_supply(x_raw_t).numpy()
        afr = air_supply / (fuel_flow_kgs + 1e-8)
    else:
        afr = analytical_air_fuel_ratio

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
    print("WARNING: exporting legacy same-row snapshots, not future-state forecasting targets.")
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
        print(f"\n🔧 Scenario: {s_name}  ({n:,} accepted samples)")
        accepted_x = []
        accepted_preds = []
        accepted_diagnostics = defaultdict(list)
        attempted = 0
        for _ in range(50):
            remaining = n - sum(len(values) for values in accepted_x)
            if remaining <= 0:
                break
            batch_n = max(10_000, remaining * 2)
            X_batch = lhs_sample(
                bounds, batch_n, seed=int(rng.integers(0, 2**31)),
            )
            for j, col in enumerate(FEATURE_COLS):
                X_batch[:, j] += generate_ar1_noise(
                    batch_n, phi_dict[col], sigma_dict[col], rng,
                )
            preds_batch, diagnostics_batch = physics_predict(
                model, X_batch, s_name, rng,
            )
            for j, col in enumerate(TARGET_COLS):
                preds_batch[:, j] += generate_ar1_noise(
                    batch_n, phi_dict[col], sigma_dict[col], rng,
                )
            preds_batch[:, 0] = np.clip(preds_batch[:, 0], 0.0, 550.0)
            preds_batch[:, 1] = np.clip(preds_batch[:, 1], 0.0, 21.0)
            mask = scenario_acceptance_mask(
                s_name, X_batch, preds_batch, diagnostics_batch,
            )
            attempted += batch_n
            if np.any(mask):
                accepted_x.append(X_batch[mask])
                accepted_preds.append(preds_batch[mask])
                for key, values in diagnostics_batch.items():
                    accepted_diagnostics[key].append(values[mask])

        accepted_count = sum(len(values) for values in accepted_x)
        if accepted_count < n:
            raise RuntimeError(
                f"Scenario {s_name} produced only {accepted_count:,} accepted "
                f"rows from {attempted:,} candidates."
            )
        X_raw = np.concatenate(accepted_x)[:n]
        preds = np.concatenate(accepted_preds)[:n]
        diagnostics = {
            key: np.concatenate(parts)[:n]
            for key, parts in accepted_diagnostics.items()
        }
        print(
            f"   Accepted {n:,}/{attempted:,} candidates "
            f"({100 * n / attempted:.2f}%)"
        )

        # 4. Mahalanobis distance (confidence flag)
        mahal = compute_mahal_distances(X_raw, mu_mah, cov_inv)

        # 5. Derive 5 EBLNN columns
        eblnn = derive_eblnn_cols(
            model,
            X_raw,
            preds,
            diagnostics["analytical_air_fuel_ratio"],
        )

        # 6. Build DataFrame
        df = pd.DataFrame(X_raw, columns=FEATURE_COLS)
        df["OutletT"]   = preds[:, 0]
        df["ExcessO2"]  = preds[:, 1]
        df["scenario"]  = s_name
        df["lambda"] = diagnostics["lambda"]
        df["lhv_factor"] = diagnostics["lhv_factor"]
        df["thermal_gain_ratio"] = diagnostics["thermal_gain_ratio"]
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
        "temperature_source": (
            "Analytical energy balance using theta_eff learned by PINN v2; "
            "not the PINN MLP prediction head."
        ),
        "scenario_acceptance_rules": {
            "S1_flame_out": "OutletT < 250 C, ExcessO2 < 1%, FGFlow > 1800 Nm3/h",
            "S2_air_leak": "DraftP < -9 mmH2O, ExcessO2 > 6%",
            "S3_tube_rupture": "OutletT > 460 C",
            "S4_positive_pressure": "DraftP > 10 mmH2O",
            "S5_fuel_trip": (
                "FGFlow < 50 Nm3/h, FGPressure < 0.8 kg/cm2g, deltaT < 10 C"
            ),
            "S6_fuel_contamination": (
                "Effective LHV ratio 0.60-0.85 at normal indicated fuel flow"
            ),
        },
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
