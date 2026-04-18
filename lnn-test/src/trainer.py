from __future__ import annotations

import copy
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class BenchmarkTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.config = config
        self.device = device

        self.epochs = config.get("epochs", 200)
        self.lr = config.get("learning_rate", 1e-3)
        self.patience = config.get("patience", 20)
        self.min_delta = config.get("min_delta", 1e-4)
        self.early_stopping = config.get("early_stopping", True)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.best_state_dict: Optional[dict] = None
        self.best_model_path: Optional[str] = None

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            pred, _ = self.model(x_batch)
            loss = self.criterion(pred, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred, _ = self.model(x_batch)
                total_loss += self.criterion(pred, y_batch).item()

        return total_loss / len(loader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, save_path: str) -> None:
        os.makedirs(save_path, exist_ok=True)
        self.best_model_path = os.path.join(save_path, "best_model.pth")

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.epochs_no_improve = 0
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_state_dict, self.best_model_path)
            else:
                self.epochs_no_improve += 1

            if self.early_stopping and self.epochs_no_improve >= self.patience:
                break

        torch.save(self.model.state_dict(), os.path.join(save_path, "last_model.pth"))

    def load_best_model(self) -> None:
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            return

        if self.best_model_path is None or not os.path.exists(self.best_model_path):
            raise FileNotFoundError("Best model checkpoint is not available.")

        state_dict = torch.load(self.best_model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_true = []
        all_pred = []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                pred, _ = self.model(x_batch)
                all_true.append(y_batch.cpu().numpy())
                all_pred.append(pred.cpu().numpy())

        y_true = np.concatenate(all_true).reshape(-1, 2)
        y_pred = np.concatenate(all_pred).reshape(-1, 2)
        return y_true, y_pred
