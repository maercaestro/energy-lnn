from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


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

INPUT_SIZE = len(INPUT_COLS)
TARGET_SIZE = len(TARGET_COLS)


class RealDataPipeline:
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

        self.input_scaler: Optional[StandardScaler] = None
        self.target_scaler: Optional[StandardScaler] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.data_summary: Dict[str, int] = {}

    @staticmethod
    def _sliding_windows(x: np.ndarray, y: np.ndarray, seq_len: int, stride: int):
        starts = range(0, len(x) - seq_len + 1, stride)
        x_seq = np.array([x[i : i + seq_len] for i in starts])
        y_seq = np.array([y[i : i + seq_len] for i in starts])
        return x_seq, y_seq

    @staticmethod
    def _block_windows(x: np.ndarray, y: np.ndarray, seq_len: int):
        n_blocks = len(x) // seq_len
        if n_blocks == 0:
            return np.empty((0, seq_len, x.shape[1])), np.empty((0, seq_len, y.shape[1]))

        x_trim = x[: n_blocks * seq_len]
        y_trim = y[: n_blocks * seq_len]
        return (
            x_trim.reshape(n_blocks, seq_len, x.shape[1]),
            y_trim.reshape(n_blocks, seq_len, y.shape[1]),
        )

    def build(self) -> "RealDataPipeline":
        all_x: List[np.ndarray] = []
        all_y: List[np.ndarray] = []

        if self.real_csv is not None:
            df_real = pd.read_csv(self.real_csv)
            x_real = df_real[INPUT_COLS].values
            y_real = df_real[TARGET_COLS].values
            x_seq, y_seq = self._sliding_windows(x_real, y_real, self.seq_len, self.stride)
            all_x.append(x_seq)
            all_y.append(y_seq)
            self.data_summary["real_rows"] = len(df_real)
            self.data_summary["real_sequences"] = len(x_seq)

        if self.edge_csv is not None:
            df_edge = pd.read_csv(self.edge_csv)

            if self.scenarios is not None:
                df_edge = df_edge[df_edge["scenario"].isin(self.scenarios)]

            if self.confidence_filter == "high":
                df_edge = df_edge[df_edge["confidence"] == "high"]
            elif self.confidence_filter == "high+medium":
                df_edge = df_edge[df_edge["confidence"].isin(["high", "medium"])]

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
                    parts.append(df_edge.sample(n=n_extra, random_state=self.seed))
                df_edge = pd.concat(parts, ignore_index=True)

            edge_x_list: List[np.ndarray] = []
            edge_y_list: List[np.ndarray] = []
            for scenario in sorted(df_edge["scenario"].unique()):
                df_scenario = df_edge[df_edge["scenario"] == scenario].reset_index(drop=True)
                x_edge = df_scenario[INPUT_COLS].values
                y_edge = df_scenario[TARGET_COLS].values
                x_blk, y_blk = self._block_windows(x_edge, y_edge, self.seq_len)
                if len(x_blk) > 0:
                    edge_x_list.append(x_blk)
                    edge_y_list.append(y_blk)

            if edge_x_list:
                x_edge_all = np.concatenate(edge_x_list)
                y_edge_all = np.concatenate(edge_y_list)
                all_x.append(x_edge_all)
                all_y.append(y_edge_all)
                self.data_summary["edge_rows"] = len(df_edge)
                self.data_summary["edge_sequences"] = len(x_edge_all)

        if not all_x:
            raise ValueError("No data loaded. Provide real_csv and/or edge_csv.")

        x_all = np.concatenate(all_x)
        y_all = np.concatenate(all_y)
        self.data_summary["total_sequences"] = len(x_all)

        x_train, x_test, y_train, y_test = train_test_split(
            x_all,
            y_all,
            test_size=self.test_size,
            random_state=self.seed,
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_test,
            y_test,
            test_size=0.5,
            random_state=self.seed,
        )

        self.input_scaler = StandardScaler().fit(x_train.reshape(-1, INPUT_SIZE))
        self.target_scaler = StandardScaler().fit(y_train.reshape(-1, TARGET_SIZE))

        def scale_inputs(array: np.ndarray) -> np.ndarray:
            return self.input_scaler.transform(array.reshape(-1, INPUT_SIZE)).reshape(array.shape)

        def scale_targets(array: np.ndarray) -> np.ndarray:
            return self.target_scaler.transform(array.reshape(-1, TARGET_SIZE)).reshape(array.shape)

        x_train = scale_inputs(x_train)
        x_val = scale_inputs(x_val)
        x_test = scale_inputs(x_test)
        y_train = scale_targets(y_train)
        y_val = scale_targets(y_val)
        y_test = scale_targets(y_test)

        def make_loader(x_data: np.ndarray, y_data: np.ndarray, shuffle: bool) -> DataLoader:
            dataset = TensorDataset(torch.FloatTensor(x_data), torch.FloatTensor(y_data))
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

        self.train_loader = make_loader(x_train, y_train, shuffle=True)
        self.val_loader = make_loader(x_val, y_val, shuffle=False)
        self.test_loader = make_loader(x_test, y_test, shuffle=False)

        self.data_summary["train_sequences"] = len(x_train)
        self.data_summary["val_sequences"] = len(x_val)
        self.data_summary["test_sequences"] = len(x_test)

        return self

    def denorm_target(self, array: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(array)

    @property
    def input_size(self) -> int:
        return INPUT_SIZE

    @property
    def target_size(self) -> int:
        return TARGET_SIZE
