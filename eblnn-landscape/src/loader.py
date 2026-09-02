"""
loader.py
=========
Model and data loading for the energy landscape analysis.

Handles:
  - Loading EBLNNWrapper from a benchmark checkpoint
  - Building per-scenario normalised sequence arrays
  - Fitting the StandardScaler on the real-data training split
    (identical split to the benchmark run so normalisation is consistent)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── project root on sys.path ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eblnn.src.model import EBLNN_Generative  # noqa: E402

# ── column names (must match benchmark data.py) ───────────────────────────
INPUT_COLS = [
    "fuel_flow",
    "air_fuel_ratio",
    "current_temp",
    "inflow_temp",
    "inflow_rate",
]
TARGET_COLS = ["next_temp", "next_excess_o2"]

# ── scenario ordering for consistent plots ────────────────────────────────
SCENARIO_ORDER = [
    "real",
    "S1_flame_out",
    "S2_air_leak",
    "S3_tube_rupture",
    "S4_positive_pressure",
    "S5_fuel_trip",
    "S6_fuel_contamination",
]

SCENARIO_LABELS = {
    "real":                  "Normal (Real)",
    "S1_flame_out":          "S1 Flame-Out",
    "S2_air_leak":           "S2 Air Leak",
    "S3_tube_rupture":       "S3 Tube Rupture",
    "S4_positive_pressure":  "S4 Positive Pressure",
    "S5_fuel_trip":          "S5 Fuel Trip",
    "S6_fuel_contamination": "S6 Fuel Contamination",
}

# Safety thresholds (physical units, same as benchmark evaluate.py)
O2_LOW_CRIT   = 1.5    # %   — ExcessO2 below this = critical
TEMP_HIGH_CRIT = 500.0  # °C — OutletT above this = critical


# ── helpers ───────────────────────────────────────────────────────────────

def _sliding_windows(
    x: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    starts = range(0, len(x) - seq_len + 1, stride)
    x_seq = np.array([x[i : i + seq_len] for i in starts], dtype=np.float32)
    y_seq = np.array([y[i : i + seq_len] for i in starts], dtype=np.float32)
    return x_seq, y_seq


# ── model loading ─────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str | Path,
    hidden_size: int = 128,
    ebm_hidden_dims: List[int] | None = None,
    input_size: int = 5,
    phys_output_size: int = 2,
    mixed_memory: bool = True,
    device: str = "cpu",
) -> EBLNN_Generative:
    """
    Instantiate EBLNN_Generative and load benchmark checkpoint weights.

    The benchmark checkpoint stores weights under the 'core.*' namespace
    because EBLNNWrapper wraps EBLNN_Generative as self.core.
    We strip the 'core.' prefix to load directly into EBLNN_Generative.
    """
    if ebm_hidden_dims is None:
        ebm_hidden_dims = [128, 64]

    model = EBLNN_Generative(
        input_size=input_size,
        hidden_size=hidden_size,
        phys_output_size=phys_output_size,
        ebm_hidden_dims=ebm_hidden_dims,
        mixed_memory=mixed_memory,
    )

    raw_state = torch.load(checkpoint_path, map_location=device)
    # Strip 'core.' prefix added by EBLNNWrapper
    state_dict = {
        k.replace("core.", "", 1): v
        for k, v in raw_state.items()
    }
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[loader] Loaded checkpoint: {checkpoint_path}")
    print(f"[loader] Model parameters : {n_params:,}")
    return model


# ── scaler ────────────────────────────────────────────────────────────────

def fit_scaler(
    real_csv: str | Path,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> StandardScaler:
    """
    Fit a StandardScaler on the *training split* of the real furnace data.

    Uses the same fraction and seed as the benchmark training run so that
    normalised values the model sees here are on the same scale as during
    training.
    """
    df = pd.read_csv(real_csv)
    x = df[INPUT_COLS].values.astype(np.float32)

    # Replicate benchmark train/test split (benchmark uses val_size+test_size
    # but scaler is fit only on train features, so a single split is enough).
    x_train, _ = train_test_split(x, test_size=1.0 - train_fraction, random_state=seed)

    scaler = StandardScaler()
    scaler.fit(x_train)
    print(f"[loader] Scaler fitted on {len(x_train):,} real-data training rows")
    return scaler


# ── sequence building ─────────────────────────────────────────────────────

def build_scenario_sequences(
    real_csv: str | Path,
    edge_csv: str | Path,
    scaler: StandardScaler,
    seq_len: int = 30,
    stride: int = 30,
    max_per_group: Optional[int] = 2000,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Build normalised input sequences for each scenario group.

    Returns
    -------
    Dict mapping scenario name → float32 array of shape (N, seq_len, 5)
    """
    rng = np.random.default_rng(seed)
    sequences: Dict[str, np.ndarray] = {}

    # ── real data ─────────────────────────────────────────────────────────
    df_real = pd.read_csv(real_csv)
    x_real = scaler.transform(df_real[INPUT_COLS].values.astype(np.float32))
    y_real = df_real[TARGET_COLS].values.astype(np.float32)
    x_seq_r, _ = _sliding_windows(x_real, y_real, seq_len, stride)

    if max_per_group is not None and len(x_seq_r) > max_per_group:
        idx = rng.choice(len(x_seq_r), max_per_group, replace=False)
        x_seq_r = x_seq_r[idx]
    sequences["real"] = x_seq_r
    print(f"[loader] real            : {len(x_seq_r):5d} sequences")

    # ── edge cases — per scenario ──────────────────────────────────────────
    df_edge = pd.read_csv(edge_csv)
    for scenario in SCENARIO_ORDER[1:]:          # skip 'real'
        df_s = df_edge[df_edge["scenario"] == scenario].reset_index(drop=True)
        if len(df_s) == 0:
            print(f"[loader] {scenario:<24s}: no rows found, skipping")
            continue

        x_s = scaler.transform(df_s[INPUT_COLS].values.astype(np.float32))
        y_s = df_s[TARGET_COLS].values.astype(np.float32)
        x_seq_s, _ = _sliding_windows(x_s, y_s, seq_len, stride)

        if max_per_group is not None and len(x_seq_s) > max_per_group:
            idx = rng.choice(len(x_seq_s), max_per_group, replace=False)
            x_seq_s = x_seq_s[idx]

        sequences[scenario] = x_seq_s
        label = SCENARIO_LABELS.get(scenario, scenario)
        print(f"[loader] {label:<24s}: {len(x_seq_s):5d} sequences")

    return sequences


# ── grid probe builder (for 2-D energy landscape) ─────────────────────────

def build_grid_probe(
    x_mean: np.ndarray,
    x_std:  np.ndarray,
    dim_i:  int,
    dim_j:  int,
    grid_res: int,
    seq_len:  int,
    n_sigma:  float = 2.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a flat batch of (grid_res², seq_len, D) probes for a 2-D sweep.

    Returns
    -------
    xi_vals : (grid_res,)  — normalised values along dim_i
    xj_vals : (grid_res,)  — normalised values along dim_j
    probes  : (grid_res², seq_len, D)  float32
    """
    D = len(x_mean)
    xi_vals = np.linspace(
        x_mean[dim_i] - n_sigma * x_std[dim_i],
        x_mean[dim_i] + n_sigma * x_std[dim_i],
        grid_res,
    )
    xj_vals = np.linspace(
        x_mean[dim_j] - n_sigma * x_std[dim_j],
        x_mean[dim_j] + n_sigma * x_std[dim_j],
        grid_res,
    )
    XI, XJ = np.meshgrid(xi_vals, xj_vals)   # (G, G)
    G = grid_res * grid_res

    probes = np.tile(x_mean, (G, seq_len, 1)).astype(np.float32)
    probes[:, :, dim_i] = np.repeat(XI.ravel(), seq_len).reshape(G, seq_len)
    probes[:, :, dim_j] = np.repeat(XJ.ravel(), seq_len).reshape(G, seq_len)

    return xi_vals, xj_vals, probes
