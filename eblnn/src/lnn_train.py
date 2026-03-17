"""
Supervised Trainer for Standalone LNN
======================================
Pure MSE-based training loop — no contrastive divergence, no Langevin
sampling, no replay buffer.  Much simpler than GenerativeTrainer.

Training Loop
─────────────
    For each batch (x, y):
      1.  phys_pred, _ = model(x)
      2.  loss = MSE(phys_pred, y)
      3.  backward + Adam step

Early stopping on validation MSE (patience=20).
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .lnn_model import LNN


class LNNTrainer:
    """
    Supervised training, validation, and evaluation for standalone LNN.

    Parameters
    ----------
    model     : LNN instance
    config    : dict — hyperparameters
    device    : 'cpu' | 'cuda'
    use_wandb : bool
    """

    def __init__(
        self,
        model: LNN,
        config: Dict,
        device: str = "cpu",
        use_wandb: bool = True,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        self.epochs = config.get("epochs", 200)
        self.lr = config.get("learning_rate", 1e-3)
        self.patience = config.get("patience", 20)
        self.min_delta = config.get("min_delta", 1e-4)
        self.early_stopping = config.get("early_stopping", True)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0

    # ------------------------------------------------------------------
    # Per-epoch
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            phys_pred, _ = self.model(x_batch)
            loss = self.criterion(phys_pred, y_batch)

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

                phys_pred, _ = self.model(x_batch)
                loss = self.criterion(phys_pred, y_batch)
                total_loss += loss.item()

        return total_loss / len(loader)

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: str = "results/models",
    ) -> None:
        os.makedirs(save_path, exist_ok=True)

        print(f"\nStarting LNN training  ({self.epochs} max epochs)")
        print(f"  lr={self.lr}")
        if self.early_stopping:
            print(f"  Early stopping: patience={self.patience}")

        for epoch in range(self.epochs):
            tr_loss = self.train_epoch(train_loader)
            va_loss = self.validate(val_loader)

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)

            if self.use_wandb:
                wandb.log({
                    "epoch":      epoch + 1,
                    "train/loss": tr_loss,
                    "val/loss":   va_loss,
                })

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1:4d}/{self.epochs} | "
                    f"Train {tr_loss:.6f} | Val {va_loss:.6f}"
                )

            # --- Best model checkpoint ---
            if va_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = va_loss
                self.best_epoch = epoch + 1
                self.epochs_no_improve = 0
                path = os.path.join(save_path, "best_model.pth")
                torch.save(self.model.state_dict(), path)
                print(f"  -> Best model saved (val_loss={self.best_val_loss:.6f})")
                if self.use_wandb:
                    wandb.run.summary["best_val_loss"] = self.best_val_loss
                    wandb.run.summary["best_epoch"] = self.best_epoch
            else:
                self.epochs_no_improve += 1

            if self.early_stopping and self.epochs_no_improve >= self.patience:
                print(f"\nEarly stopping at epoch {epoch + 1} "
                      f"(no improvement for {self.patience} epochs)")
                if self.use_wandb:
                    wandb.run.summary["early_stopped"] = True
                    wandb.run.summary["stopped_epoch"] = epoch + 1
                break

        last_path = os.path.join(save_path, "last_model.pth")
        torch.save(self.model.state_dict(), last_path)
        print(f"\nLast model saved: {last_path}")
        print(f"Best val loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_loader: DataLoader,
        target_scaler=None,
    ) -> Dict[str, float]:
        """
        Evaluate on test set, returning RMSE for temp and O2.
        """
        self.model.eval()

        all_y_true, all_y_pred = [], []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                phys_pred, _ = self.model(x_batch)

                all_y_true.append(y_batch.cpu().numpy())
                all_y_pred.append(phys_pred.cpu().numpy())

        y_true = np.concatenate(all_y_true).reshape(-1, 2)
        y_pred = np.concatenate(all_y_pred).reshape(-1, 2)

        if target_scaler is not None:
            y_true = target_scaler.inverse_transform(y_true)
            y_pred = target_scaler.inverse_transform(y_pred)

        metrics = {}
        col_names = ["temp", "o2"]
        for i, col in enumerate(col_names):
            err = y_pred[:, i] - y_true[:, i]
            abs_err = np.abs(err)
            # Avoid division by zero for MAPE
            nonzero = np.abs(y_true[:, i]) > 1e-8
            mape = float(np.mean(abs_err[nonzero] / np.abs(y_true[:, i][nonzero])) * 100) if nonzero.any() else float("nan")
            ss_res = np.sum(err ** 2)
            ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

            metrics[f"test_rmse_{col}"] = float(np.sqrt(np.mean(err ** 2)))
            metrics[f"test_mae_{col}"]  = float(np.mean(abs_err))
            metrics[f"test_mape_{col}"] = mape
            metrics[f"test_r2_{col}"]   = r2
            metrics[f"test_max_ae_{col}"] = float(np.max(abs_err))

        print("\n--- Test Set Evaluation ---")
        for col in col_names:
            label = "Temperature" if col == "temp" else "Excess O2  "
            unit  = "°C" if col == "temp" else "%"
            print(f"  {label} RMSE   : {metrics[f'test_rmse_{col}']:.4f} {unit}")
            print(f"  {label} MAE    : {metrics[f'test_mae_{col}']:.4f} {unit}")
            print(f"  {label} MAPE   : {metrics[f'test_mape_{col}']:.2f} %")
            print(f"  {label} R²     : {metrics[f'test_r2_{col}']:.4f}")
            print(f"  {label} MaxAE  : {metrics[f'test_max_ae_{col}']:.4f} {unit}")

        if self.use_wandb:
            wandb.log(metrics)
            wandb.run.summary.update(metrics)

        self.test_predictions = {
            "true_temp":  y_true[:, 0],
            "pred_temp":  y_pred[:, 0],
            "true_o2":    y_true[:, 1],
            "pred_o2":    y_pred[:, 1],
        }

        return metrics
