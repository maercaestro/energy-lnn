"""
Generative Trainer for EB-LNN
===============================
Orchestrates the joint training loop:

  ┌─────────────────────────────────────────────────────────────┐
  │  For each batch x_pos:                                       │
  │                                                              │
  │  1. Physics forward:  phys_pred, E_pos, _ = model(x_pos)    │
  │                                                              │
  │  2. Langevin sample:  x_neg = sampler.sample(x_init)        │
  │     • x_init drawn from replay buffer (95%) or noise (5%)   │
  │     • run n_steps updates guided by ∇_x E_θ(x)             │
  │     • write x_neg back to buffer                            │
  │                                                              │
  │  3. Negative energy:  _, E_neg, _ = model(x_neg)            │
  │                                                              │
  │  4. Joint loss:       L = L_phys + α · L_CD                 │
  │                                                              │
  │  5. Backward + step optimiser                                │
  └─────────────────────────────────────────────────────────────┘

Key differences vs pilot study
-------------------------------
- No `calculate_true_energy` supervision → EBM is self-supervised.
- Replay buffer avoids running long MCMC chains from scratch each step.
- W&B logs include CD-specific metrics: e_pos, e_neg, cd_gap.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .model import EBLNN_Generative
from .losses import JointLoss
from .sampler import LangevinSampler, ReplayBuffer


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GenerativeTrainer:
    """
    Full training, validation, and evaluation loop for EBLNN_Generative.

    Parameters
    ----------
    model     : EBLNN_Generative instance
    sampler   : LangevinSampler for fantasy-state generation
    config    : dict — hyperparameters (see base_config.yaml for keys)
    device    : 'cpu' | 'cuda'
    use_wandb : bool
    """

    def __init__(
        self,
        model: EBLNN_Generative,
        sampler: LangevinSampler,
        config: Dict,
        device: str = "cpu",
        use_wandb: bool = True,
    ) -> None:
        self.model = model
        self.sampler = sampler
        self.config = config
        self.device = device
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # ---- Hyper-parameters ----
        self.epochs = config.get("epochs", 200)
        self.lr = config.get("learning_rate", 1e-3)
        self.alpha = config.get("alpha", 1.0)
        self.l2_reg = config.get("l2_reg", 0.01)
        self.margin = config.get("margin", 0.0)
        self.patience = config.get("patience", 20)
        self.min_delta = config.get("min_delta", 1e-4)
        self.early_stopping = config.get("early_stopping", True)

        # ---- Buffer config ----
        buf_capacity = config.get("buffer_capacity", 10_000)
        buf_prob = config.get("buffer_prob", 0.95)
        self.buffer_prob = buf_prob

        seq_len = config.get("seq_len", 30)
        input_size = config.get("input_size", 5)

        self.replay_buffer = ReplayBuffer(
            capacity=buf_capacity,
            input_shape=(seq_len, input_size),
            device=device,
        )

        # ---- Loss + optimiser ----
        self.criterion = JointLoss(
            alpha=self.alpha,
            l2_reg=self.l2_reg,
            margin=self.margin,
            energy_clamp=config.get("energy_clamp", 20.0),
        )
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # ---- Training state ----
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_phys": [], "val_phys": [],
            "train_cd": [],   "val_cd": [],
            "e_pos": [],      "e_neg": [],    "cd_gap": [],
            "val_phys_loss": [],   # used for early stopping
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0

    # ------------------------------------------------------------------
    # Core per-epoch methods
    # ------------------------------------------------------------------

    def _get_fantasy(self, batch_size: int, seq_shape: tuple) -> torch.Tensor:
        """
        Draw initial states from replay buffer, run Langevin, write back.

        Parameters
        ----------
        batch_size : int
        seq_shape  : (seq_len, input_size)

        Returns
        -------
        x_neg : (B, T, input_size) on self.device
        """
        x_init, buf_idxs = self.replay_buffer.sample(
            batch_size,
            buffer_prob=self.buffer_prob,
        )
        x_init = x_init.to(self.device)

        # Langevin MCMC — model must be in eval mode during sampling
        # so batch-norm / dropout behave correctly
        self.model.eval()
        x_neg = self.sampler.sample(x_init)
        self.model.train()

        # Write back to buffer
        self.replay_buffer.update(buf_idxs, x_neg.cpu())

        return x_neg

    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()

        totals: Dict[str, float] = {
            "loss": 0.0, "phys": 0.0, "cd": 0.0,
            "e_pos": 0.0, "e_neg": 0.0, "cd_gap": 0.0,
        }

        for x_pos, y_pos in loader:
            x_pos = x_pos.to(self.device)   # (B, T, input_size)
            y_pos = y_pos.to(self.device)   # (B, T, target_size)

            B = x_pos.size(0)

            # --- 1. Physics + positive energy forward pass ---
            phys_pred, e_pos_seq, _ = self.model(x_pos)
            e_pos = e_pos_seq.squeeze(-1).mean(dim=1)    # (B,)

            # --- 2. Generate fantasy (negative) state ---
            seq_shape = (x_pos.size(1), x_pos.size(2))
            x_neg = self._get_fantasy(B, seq_shape)      # (B, T, input_size)

            # --- 3. Negative energy ---
            _, e_neg_seq, _ = self.model(x_neg)
            e_neg = e_neg_seq.squeeze(-1).mean(dim=1)    # (B,)

            # --- 4. Joint loss ---
            loss, metrics, phys_loss_val = self.criterion(
                phys_pred, y_pos, e_pos, e_neg
            )

            # --- 5. Back-prop ---
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clip for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            totals["loss"]   += metrics["loss_total"]
            totals["phys"]   += metrics["loss_physics"]
            totals["cd"]     += metrics["loss_cd"]
            totals["e_pos"]  += metrics["e_pos"]
            totals["e_neg"]  += metrics["e_neg"]
            totals["cd_gap"] += metrics["cd_gap"]

        n = len(loader)
        return {k: v / n for k, v in totals.items()}

    # ------------------------------------------------------------------

    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()

        totals: Dict[str, float] = {
            "loss": 0.0, "phys": 0.0, "cd": 0.0,
        }

        with torch.no_grad():
            for x_pos, y_pos in loader:
                x_pos = x_pos.to(self.device)
                y_pos = y_pos.to(self.device)

                B = x_pos.size(0)

                phys_pred, e_pos_seq, _ = self.model(x_pos)
                e_pos = e_pos_seq.squeeze(-1).mean(dim=1)

                # Validation negatives: fresh noise (no Langevin — too slow)
                x_neg = torch.randn_like(x_pos)
                _, e_neg_seq, _ = self.model(x_neg)
                e_neg = e_neg_seq.squeeze(-1).mean(dim=1)

                loss, metrics = self.criterion(phys_pred, y_pos, e_pos, e_neg)[:2]

                totals["loss"] += metrics["loss_total"]
                totals["phys"] += metrics["loss_physics"]
                totals["cd"]   += metrics["loss_cd"]

        n = len(loader)
        return {k: v / n for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: str = "results/models",
    ) -> None:
        """
        Run the full training loop with early stopping and W&B logging.

        Parameters
        ----------
        train_loader : DataLoader for training sequences
        val_loader   : DataLoader for validation sequences
        save_path    : directory for model checkpoints
        """
        os.makedirs(save_path, exist_ok=True)

        print(f"\nStarting Generative EB-LNN training  ({self.epochs} max epochs)")
        print(f"  alpha={self.alpha}, lr={self.lr}, l2_reg={self.l2_reg}")
        print(f"  Langevin: {self.sampler.n_steps} steps, "
              f"step_size={self.sampler.step_size}, "
              f"noise_scale={self.sampler.noise_scale}")
        print(f"  Replay buffer capacity={self.replay_buffer.capacity}, "
              f"prob={self.buffer_prob}")
        if self.early_stopping:
            print(f"  Early stopping: patience={self.patience}")

        for epoch in range(self.epochs):

            # --- Train ---
            tr = self.train_epoch(train_loader)
            # --- Validate ---
            va = self.validate(val_loader)

            # --- Store history ---
            self.history["train_loss"].append(tr["loss"])
            self.history["val_loss"].append(va["loss"])
            self.history["train_phys"].append(tr["phys"])
            self.history["val_phys"].append(va["phys"])
            self.history["val_phys_loss"].append(va["phys"])
            self.history["train_cd"].append(tr["cd"])
            self.history["val_cd"].append(va["cd"])
            self.history["e_pos"].append(tr["e_pos"])
            self.history["e_neg"].append(tr["e_neg"])
            self.history["cd_gap"].append(tr["cd_gap"])

            # --- W&B ---
            if self.use_wandb:
                wandb.log({
                    "epoch":         epoch + 1,
                    "train/loss":    tr["loss"],
                    "train/physics": tr["phys"],
                    "train/cd":      tr["cd"],
                    "train/e_pos":   tr["e_pos"],
                    "train/e_neg":   tr["e_neg"],
                    "train/cd_gap":  tr["cd_gap"],
                    "val/loss":      va["loss"],
                    "val/physics":   va["phys"],
                    "val/cd":        va["cd"],
                })

            # --- Console log ---
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1:4d}/{self.epochs} | "
                    f"Train {tr['loss']:.4f} (phys {tr['phys']:.4f} cd {tr['cd']:.4f}) | "
                    f"Val {va['loss']:.4f} | "
                    f"E+={tr['e_pos']:.3f} E-={tr['e_neg']:.3f} gap={tr['cd_gap']:.3f}"
                )

            # --- Best model: use physics loss as checkpoint criterion ---
            # (total loss can legitimately go negative due to CD; physics
            #  loss is a stable, unsigned measure of prediction quality)
            checkpoint_metric = va["phys"]
            if checkpoint_metric < self.best_val_loss - self.min_delta:
                self.best_val_loss = checkpoint_metric
                self.best_epoch = epoch + 1
                self.epochs_no_improve = 0
                path = os.path.join(save_path, "best_model.pth")
                torch.save(self.model.state_dict(), path)
                print(f"  -> Best model saved (val_phys={self.best_val_loss:.4f})")
                if self.use_wandb:
                    wandb.run.summary["best_val_phys_loss"] = self.best_val_loss
                    wandb.run.summary["best_epoch"] = self.best_epoch
            else:
                self.epochs_no_improve += 1

            # --- Early stop ---
            if self.early_stopping and self.epochs_no_improve >= self.patience:
                print(f"\nEarly stopping at epoch {epoch + 1} "
                      f"(no improvement for {self.patience} epochs)")
                if self.use_wandb:
                    wandb.run.summary["early_stopped"] = True
                    wandb.run.summary["stopped_epoch"] = epoch + 1
                break

        # Save last checkpoint
        last_path = os.path.join(save_path, "last_model.pth")
        torch.save(self.model.state_dict(), last_path)
        print(f"\nLast model saved: {last_path}")
        print(f"Best val loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_loader: DataLoader,
        target_scaler=None,
    ) -> Dict[str, float]:
        """
        Evaluate physics prediction quality on the test set.

        Parameters
        ----------
        test_loader    : DataLoader for test sequences
        target_scaler  : optional sklearn scaler for denormalising outputs

        Returns
        -------
        metrics dict with RMSE for temp and O2
        """
        self.model.eval()

        all_y_true, all_y_pred = [], []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                phys_pred, _, _ = self.model(x_batch)

                all_y_true.append(y_batch.cpu().numpy())
                all_y_pred.append(phys_pred.cpu().numpy())

        y_true = np.concatenate(all_y_true).reshape(-1, 2)
        y_pred = np.concatenate(all_y_pred).reshape(-1, 2)

        if target_scaler is not None:
            y_true = target_scaler.inverse_transform(y_true)
            y_pred = target_scaler.inverse_transform(y_pred)

        rmse_temp = float(np.sqrt(np.mean((y_pred[:, 0] - y_true[:, 0]) ** 2)))
        rmse_o2   = float(np.sqrt(np.mean((y_pred[:, 1] - y_true[:, 1]) ** 2)))

        metrics = {
            "test_rmse_temp": rmse_temp,
            "test_rmse_o2":   rmse_o2,
        }

        print("\n--- Test Set Evaluation ---")
        print(f"  Temperature RMSE : {rmse_temp:.4f} °C")
        print(f"  Excess O2   RMSE : {rmse_o2:.4f} %")

        if self.use_wandb:
            wandb.log(metrics)
            wandb.run.summary.update(metrics)

        # Store predictions for downstream visualisation
        self.test_predictions = {
            "true_temp":  y_true[:, 0],
            "pred_temp":  y_pred[:, 0],
            "true_o2":    y_true[:, 1],
            "pred_o2":    y_pred[:, 1],
        }

        return metrics
