from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch

from .data import RealDataPipeline


SENSOR_COLUMNS = {
    "fuel_flow": "FGFlow",
    "draft_pressure": "DraftP",
    "current_temp": "OutletT",
    "inflow_temp": "InletT",
    "inflow_rate": "InletFlow",
    "current_o2": "ExcessO2",
    "damper_position": "OP_Damper",
    "fuel_pressure": "FGPressure",
    "bridgewall_temp": "Bridgewall",
}
FORECAST_FEATURES = list(SENSOR_COLUMNS)


class ForecastDataPipeline(RealDataPipeline):
    """Future targets from contiguous real observations, split before windowing."""

    def __init__(self, *, horizon: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("Forecast horizon must be a positive integer number of recorded steps.")
        if self.seq_len < 1 or self.stride < 1 or self.batch_size < 1:
            raise ValueError("Sequence length, stride, and batch size must be positive.")
        if not (0 < self.val_size < 1 and 0 < self.test_size < 1
                and self.val_size + self.test_size < 1):
            raise ValueError("Validation/test fractions must be positive and sum to less than one.")
        if (self.edge_csv is not None or self.scenarios is not None
                or self.confidence_filter is not None or self.edge_fraction != 1.0):
            raise ValueError("Forecast v1 supports unfiltered real data only; no synthetic snapshots.")
        self.horizon = horizon
        self.feature_names = FORECAST_FEATURES.copy()
        self.window_dates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.protocol_metadata: dict = {}

    @property
    def input_size(self) -> int:
        return len(self.feature_names)

    def build(self) -> "ForecastDataPipeline":
        if self.real_csv is None:
            raise ValueError("Forecasting requires cleaned_furnace_data.csv with Date and raw sensors.")
        path = Path(self.real_csv)
        df = pd.read_csv(path)
        required = ["Date", *SENSOR_COLUMNS.values()]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing forecast source columns: {sorted(missing)}")
        if not np.isfinite(df[required].to_numpy(dtype=np.float64)).all():
            raise ValueError("Forecast source contains nonfinite dates or sensor values.")
        dates = df["Date"].to_numpy()
        if not np.equal(dates, np.floor(dates)).all() or np.any(np.diff(dates) <= 0):
            raise ValueError("Date must contain strictly increasing, unique integer sample IDs.")

        n = len(df)
        train_end = int(n * (1 - self.val_size - self.test_size))
        val_end = int(n * (1 - self.test_size))
        boundaries = {
            "train": (0, train_end), "val": (train_end, val_end), "test": (val_end, n),
        }
        arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        ranges = {}
        for split, (lo, hi) in boundaries.items():
            frame = df.iloc[lo:hi]
            split_dates = frame["Date"].to_numpy()
            x = frame[list(SENSOR_COLUMNS.values())].to_numpy(dtype=np.float64)
            y = frame[["OutletT", "ExcessO2"]].to_numpy(dtype=np.float64)
            breaks = np.flatnonzero(np.diff(split_dates) != 1) + 1
            edges = np.r_[0, breaks, len(frame)]
            starts = [
                start
                for left, right in zip(edges[:-1], edges[1:])
                for start in range(left, right - self.seq_len - self.horizon + 1, self.stride)
            ]
            if not starts:
                raise ValueError(f"No contiguous forecast windows in {split}; check split sizes/gaps.")
            input_rows = np.asarray(starts)[:, None] + np.arange(self.seq_len)
            target_rows = input_rows + self.horizon
            arrays[split] = x[input_rows], y[target_rows]
            self.window_dates[split] = split_dates[input_rows], split_dates[target_rows]
            ranges[split] = {
                "first_date": int(split_dates[0]), "last_date": int(split_dates[-1]),
                "source_rows": len(frame), "sequences": len(starts),
            }

        self.input_scaler = StandardScaler().fit(arrays["train"][0].reshape(-1, self.input_size))
        self.target_scaler = StandardScaler().fit(arrays["train"][1].reshape(-1, self.target_size))
        for split, (x, y) in arrays.items():
            x_scaled = self.input_scaler.transform(x.reshape(-1, self.input_size)).reshape(x.shape)
            y_scaled = self.target_scaler.transform(y.reshape(-1, self.target_size)).reshape(y.shape)
            loader = DataLoader(
                TensorDataset(torch.tensor(x_scaled, dtype=torch.float32),
                              torch.tensor(y_scaled, dtype=torch.float32)),
                batch_size=self.batch_size, shuffle=(split == "train"),
            )
            setattr(self, f"{split}_loader", loader)
        self.data_summary = {
            "real_rows": n,
            **{f"{split}_sequences": len(x) for split, (x, _) in arrays.items()},
            "total_sequences": sum(len(x) for x, _ in arrays.values()),
        }
        self.protocol_metadata = {
            "protocol": "forecast-real-v1",
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "horizon_recorded_steps": self.horizon,
            "date_semantics": "Integer sample ID; consecutive IDs required. Not a wall-clock timestamp.",
            "sensor_columns": SENSOR_COLUMNS,
            "feature_names": self.feature_names,
            "target_columns": ["OutletT", "ExcessO2"],
            "split_strategy": "chronological-before-windowing",
            "split_ranges": ranges,
        }
        return self
