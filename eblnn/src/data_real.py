"""
Real Data Pipeline for Ablation Studies
========================================
Loads real furnace data and/or physics-generated edge-case data from CSV,
creates temporal sequences, applies StandardScaler, and produces DataLoaders.

Key differences from ``data.py`` (which generates synthetic data):
  - Reads pre-existing CSV files (real + edge cases)
  - Supports filtering by scenario, confidence, and volume fraction
  - Creates sequences via sliding windows (real) or non-overlapping
    blocks within each scenario (edge cases, which have AR(1) noise
    correlation between consecutive rows)

Input features  (5): fuel_flow, air_fuel_ratio, current_temp,
                     inflow_temp, inflow_rate
Physics targets (2): next_temp, next_excess_o2
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Column definitions (must match EBLNN-format CSVs)
# ---------------------------------------------------------------------------

INPUT_COLS = [
    "fuel_flow",
    "air_fuel_ratio",
    "current_temp",
    "inflow_temp",
    "inflow_rate",
]

TARGET_COLS = [
    "next_temp",
    "next_excess_o2",
]

INPUT_SIZE = len(INPUT_COLS)   # 5
TARGET_SIZE = len(TARGET_COLS)  # 2


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RealDataPipeline:
    """
    End-to-end data preparation for ablation studies on real furnace data.

    Parameters
    ----------
    real_csv : str or None
        Path to the real furnace data in EBLNN format.  Set *None* to
        train on edge cases only (experiment A3).
    edge_csv : str or None
        Path to the edge-case data in EBLNN format.  Set *None* to
        train on real data only (experiment A1).
    scenarios : list[str] or None
        If set, only keep edge cases whose ``scenario`` column is in
        this list (e.g. ``["S1_flame_out", "S3_tube_rupture"]``).
    confidence_filter : str or None
        ``"high"``  — keep only high-confidence edge cases.
        ``"high+medium"`` — keep high + medium.
        ``None``   — keep all (default).
    edge_fraction : float
        Fraction of edge cases to use (0.1 = 10%).  Values > 1.0
        oversample by repeating (2.0 = 2× edge cases).
    seq_len : int
        Sequence length for the CfC model (default 30).
    stride : int or None
        Sliding-window stride for real data.  Defaults to ``seq_len``
        (non-overlapping).  Use smaller values for more sequences.
    batch_size : int
    test_size : float
    val_size : float
    seed : int

    Attributes (available after ``build()``)
    -----------------------------------------
    input_scaler   : StandardScaler
    target_scaler  : StandardScaler
    train_loader   : DataLoader
    val_loader     : DataLoader
    test_loader    : DataLoader
    data_summary   : dict with sequence counts per source
    """

    def __init__(
        self,
        real_csv: Optional[str] = None,
        edge_csv: Optional[str] = None,
        scenarios: Optional[List[str]] = None,
        confidence_filter: Optional[str] = None,
        edge_fraction: float = 1.0,
        seq_len: int = 30,
        stride: Optional[int] = None,
        batch_size: int = 64,
        test_size: float = 0.2,
        val_size: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.real_csv = real_csv
        self.edge_csv = edge_csv
        self.scenarios = scenarios
        self.confidence_filter = confidence_filter
        self.edge_fraction = edge_fraction
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed

        # Populated by build()
        self.input_scaler: Optional[StandardScaler] = None
        self.target_scaler: Optional[StandardScaler] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.data_summary: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sliding_windows(
        x: np.ndarray, y: np.ndarray, seq_len: int, stride: int
    ):
        """Create sequences via sliding window."""
        n = len(x)
        starts = range(0, n - seq_len + 1, stride)
        x_seq = np.array([x[i : i + seq_len] for i in starts])
        y_seq = np.array([y[i : i + seq_len] for i in starts])
        return x_seq, y_seq

    @staticmethod
    def _block_windows(x: np.ndarray, y: np.ndarray, seq_len: int):
        """Create sequences via non-overlapping blocks."""
        n_blocks = len(x) // seq_len
        if n_blocks == 0:
            return np.empty((0, seq_len, x.shape[1])), np.empty((0, seq_len, y.shape[1]))
        x_trim = x[: n_blocks * seq_len]
        y_trim = y[: n_blocks * seq_len]
        return (
            x_trim.reshape(n_blocks, seq_len, x.shape[1]),
            y_trim.reshape(n_blocks, seq_len, y.shape[1]),
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> "RealDataPipeline":
        """Load data, create sequences, scale, split, build DataLoaders."""

        rng = np.random.default_rng(self.seed)
        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []

        # ── 1. Real data ─────────────────────────────────────────────
        if self.real_csv is not None:
            df_real = pd.read_csv(self.real_csv)
            x_r = df_real[INPUT_COLS].values
            y_r = df_real[TARGET_COLS].values

            x_seq, y_seq = self._sliding_windows(
                x_r, y_r, self.seq_len, self.stride
            )
            all_x.append(x_seq)
            all_y.append(y_seq)
            self.data_summary["real_rows"] = len(df_real)
            self.data_summary["real_sequences"] = len(x_seq)
            print(
                f"Real data : {len(df_real):,} rows  →  "
                f"{len(x_seq):,} sequences  (stride={self.stride})"
            )

        # ── 2. Edge-case data ────────────────────────────────────────
        if self.edge_csv is not None:
            df_edge = pd.read_csv(self.edge_csv)
            n_raw = len(df_edge)

            # Filter by scenario
            if self.scenarios is not None:
                df_edge = df_edge[df_edge["scenario"].isin(self.scenarios)]
                print(
                    f"Scenario filter {self.scenarios}: "
                    f"{n_raw:,} → {len(df_edge):,} rows"
                )

            # Filter by confidence
            if self.confidence_filter == "high":
                df_edge = df_edge[df_edge["confidence"] == "high"]
            elif self.confidence_filter == "high+medium":
                df_edge = df_edge[
                    df_edge["confidence"].isin(["high", "medium"])
                ]
            if self.confidence_filter:
                print(
                    f"Confidence filter '{self.confidence_filter}': "
                    f"{len(df_edge):,} rows kept"
                )

            # Volume fraction (sub/over-sample)
            if self.edge_fraction < 1.0:
                n_keep = max(self.seq_len, int(len(df_edge) * self.edge_fraction))
                df_edge = df_edge.sample(n=n_keep, random_state=self.seed)
            elif self.edge_fraction > 1.0:
                n_full = len(df_edge)
                n_repeats = int(self.edge_fraction)
                remainder = self.edge_fraction - n_repeats
                parts = [df_edge] * n_repeats
                if remainder > 0:
                    n_extra = int(n_full * remainder)
                    parts.append(
                        df_edge.sample(n=n_extra, random_state=self.seed)
                    )
                df_edge = pd.concat(parts, ignore_index=True)

            # Create sequences per scenario (non-overlapping blocks)
            edge_x_list, edge_y_list = [], []
            for scenario in sorted(df_edge["scenario"].unique()):
                df_s = df_edge[df_edge["scenario"] == scenario].reset_index(
                    drop=True
                )
                xs = df_s[INPUT_COLS].values
                ys = df_s[TARGET_COLS].values
                x_blk, y_blk = self._block_windows(xs, ys, self.seq_len)
                if len(x_blk) > 0:
                    edge_x_list.append(x_blk)
                    edge_y_list.append(y_blk)

            if edge_x_list:
                x_e = np.concatenate(edge_x_list)
                y_e = np.concatenate(edge_y_list)
                all_x.append(x_e)
                all_y.append(y_e)
                self.data_summary["edge_rows"] = len(df_edge)
                self.data_summary["edge_sequences"] = len(x_e)
                print(
                    f"Edge cases: {len(df_edge):,} rows  →  "
                    f"{len(x_e):,} sequences  (block={self.seq_len})"
                )
            else:
                print("⚠  No edge-case sequences produced (too few rows?)")

        # ── 3. Combine ───────────────────────────────────────────────
        if not all_x:
            raise ValueError(
                "No data loaded — set at least one of real_csv / edge_csv"
            )

        x_all = np.concatenate(all_x)
        y_all = np.concatenate(all_y)
        self.data_summary["total_sequences"] = len(x_all)
        print(
            f"Total     : {len(x_all):,} sequences  "
            f"(seq_len={self.seq_len})"
        )

        # ── 4. Train / Val / Test split ──────────────────────────────
        x_tr, x_te, y_tr, y_te = train_test_split(
            x_all, y_all, test_size=self.test_size, random_state=self.seed
        )
        x_val, x_te, y_val, y_te = train_test_split(
            x_te, y_te, test_size=0.5, random_state=self.seed
        )
        print(
            f"Split     : {len(x_tr):,} train / "
            f"{len(x_val):,} val / {len(x_te):,} test"
        )

        # ── 5. StandardScaler (fit on train only) ────────────────────
        self.input_scaler = StandardScaler().fit(
            x_tr.reshape(-1, INPUT_SIZE)
        )
        self.target_scaler = StandardScaler().fit(
            y_tr.reshape(-1, TARGET_SIZE)
        )

        def _sx(a: np.ndarray) -> np.ndarray:
            return self.input_scaler.transform(
                a.reshape(-1, INPUT_SIZE)
            ).reshape(a.shape)

        def _sy(a: np.ndarray) -> np.ndarray:
            return self.target_scaler.transform(
                a.reshape(-1, TARGET_SIZE)
            ).reshape(a.shape)

        x_tr, x_val, x_te = _sx(x_tr), _sx(x_val), _sx(x_te)
        y_tr, y_val, y_te = _sy(y_tr), _sy(y_val), _sy(y_te)
        print("Scaling complete")

        # ── 6. DataLoaders ───────────────────────────────────────────
        def _loader(xa, ya, shuffle: bool) -> DataLoader:
            ds = TensorDataset(torch.FloatTensor(xa), torch.FloatTensor(ya))
            return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

        self.train_loader = _loader(x_tr, y_tr, shuffle=True)
        self.val_loader = _loader(x_val, y_val, shuffle=False)
        self.test_loader = _loader(x_te, y_te, shuffle=False)

        print("DataLoaders ready\n")
        return self

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def denorm_target(self, arr: np.ndarray) -> np.ndarray:
        """Inverse-transform (N, TARGET_SIZE) array to physical units."""
        return self.target_scaler.inverse_transform(arr)

    def denorm_input(self, arr: np.ndarray) -> np.ndarray:
        """Inverse-transform (N, INPUT_SIZE) array to physical units."""
        return self.input_scaler.inverse_transform(arr)

    @property
    def input_size(self) -> int:
        return INPUT_SIZE

    @property
    def target_size(self) -> int:
        return TARGET_SIZE
