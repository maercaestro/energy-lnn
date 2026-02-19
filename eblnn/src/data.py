"""
Data Pipeline for Generative EB-LNN
=====================================
Reuses the FurnaceDataGenerator from the pilot study (physics-based
synthetic data).  Key difference vs pilot: we no longer compute or store
a `calculate_true_energy` target column — the EBM learns its own landscape
via Contrastive Divergence, so we only need physics targets (next_temp,
next_excess_o2).

Input features  (5): fuel_flow, air_fuel_ratio, current_temp,
                     inflow_temp, inflow_rate
Physics targets (2): next_temp, next_excess_o2
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# --- Load pilot-study's data_generation by absolute file path ---
# Avoids sys.path pollution and is reliable regardless of cwd or how
# the package is imported.
_DATA_GEN_PATH = (
    Path(__file__).resolve().parents[2]
    / "pilot-study" / "src" / "data_generation.py"
)
_spec = importlib.util.spec_from_file_location("_pilot_data_generation", _DATA_GEN_PATH)
_data_gen_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data_gen_mod)

FurnaceDataGenerator = _data_gen_mod.FurnaceDataGenerator
load_or_generate_data = _data_gen_mod.load_or_generate_data


# ---------------------------------------------------------------------------
# Feature / target columns
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
TARGET_SIZE = len(TARGET_COLS) # 2


# ---------------------------------------------------------------------------
# DataPipeline
# ---------------------------------------------------------------------------

class DataPipeline:
    """
    End-to-end data preparation for generative EB-LNN training.

    Steps
    -----
    1. Load or generate synthetic furnace data.
    2. Reshape flat rows into (n_sequences, seq_len, features) arrays.
    3. Train / val / test split.
    4. StandardScaler fit on training data only.
    5. Create PyTorch DataLoaders.

    Attributes exposed after ``build()``
    -------------------------------------
    input_scaler    : StandardScaler for input features
    target_scaler   : StandardScaler for physics targets
    train_loader    : DataLoader
    val_loader      : DataLoader
    test_loader     : DataLoader
    seq_len         : int
    input_size      : int
    target_size     : int
    """

    def __init__(
        self,
        data_path: str = "data/synthetic_temperature_data.csv",
        num_sequences: int = 10_000,
        seq_len: int = 30,
        batch_size: int = 64,
        test_size: float = 0.2,
        val_size: float = 0.1,
        seed: int = 42,
        force_regenerate: bool = False,
    ) -> None:
        self.data_path = data_path
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.force_regenerate = force_regenerate

        # Set after build()
        self.input_scaler: StandardScaler | None = None
        self.target_scaler: StandardScaler | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

    # ------------------------------------------------------------------

    def build(self) -> "DataPipeline":
        """Load data, scale, split, and create DataLoaders.  Returns self."""

        # 1. Load / generate
        df = load_or_generate_data(
            data_path=self.data_path,
            num_sequences=self.num_sequences,
            sequence_length=self.seq_len,
            seed=self.seed,
            force_regenerate=self.force_regenerate,
        )

        # 2. Reshape to sequences
        n_seq = len(df) // self.seq_len
        x = df[INPUT_COLS].values[:n_seq * self.seq_len]
        y = df[TARGET_COLS].values[:n_seq * self.seq_len]

        x = x.reshape(n_seq, self.seq_len, INPUT_SIZE)   # (N, T, 5)
        y = y.reshape(n_seq, self.seq_len, TARGET_SIZE)  # (N, T, 2)

        # 3. Split
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=self.test_size, random_state=self.seed
        )
        x_val, x_te, y_val, y_te = train_test_split(
            x_te, y_te, test_size=0.5, random_state=self.seed
        )

        print(
            f"Data split: {len(x_tr)} train / {len(x_val)} val / {len(x_te)} test sequences"
        )

        # 4. Scale (fit on train only)
        self.input_scaler = StandardScaler().fit(x_tr.reshape(-1, INPUT_SIZE))
        self.target_scaler = StandardScaler().fit(y_tr.reshape(-1, TARGET_SIZE))

        def _scale_x(arr: np.ndarray) -> np.ndarray:
            return self.input_scaler.transform(
                arr.reshape(-1, INPUT_SIZE)
            ).reshape(arr.shape)

        def _scale_y(arr: np.ndarray) -> np.ndarray:
            return self.target_scaler.transform(
                arr.reshape(-1, TARGET_SIZE)
            ).reshape(arr.shape)

        x_tr, x_val, x_te = _scale_x(x_tr), _scale_x(x_val), _scale_x(x_te)
        y_tr, y_val, y_te = _scale_y(y_tr), _scale_y(y_val), _scale_y(y_te)

        print("Scaling complete")

        # 5. DataLoaders
        def _loader(xa, ya, shuffle: bool) -> DataLoader:
            ds = TensorDataset(
                torch.FloatTensor(xa), torch.FloatTensor(ya)
            )
            return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

        self.train_loader = _loader(x_tr, y_tr, shuffle=True)
        self.val_loader = _loader(x_val, y_val, shuffle=False)
        self.test_loader = _loader(x_te, y_te, shuffle=False)

        print("DataLoaders ready")
        return self

    # ------------------------------------------------------------------

    def denorm_target(self, arr: np.ndarray) -> np.ndarray:
        """Inverse-transform a (N, TARGET_SIZE) array back to physical units."""
        return self.target_scaler.inverse_transform(arr)

    def denorm_input(self, arr: np.ndarray) -> np.ndarray:
        """Inverse-transform a (N, INPUT_SIZE) array back to physical units."""
        return self.input_scaler.inverse_transform(arr)

    # ------------------------------------------------------------------

    @property
    def input_size(self) -> int:
        return INPUT_SIZE

    @property
    def target_size(self) -> int:
        return TARGET_SIZE
