"""
Trainers for the EB-LNN vs LNN vs LSTM vs PID benchmark.

Three strategies, one shared interface:

    .train(train_loader, val_loader, save_path)
    .load_best_model()
    .predict(loader) -> (y_true, y_pred)
    .history (dict with at least 'train_loss', 'val_loss')
    .best_val_loss, .best_epoch, .best_model_path

  - SupervisedTrainer : MSE training for LNN and LSTM.
  - CDTrainer         : Joint physics + Contrastive Divergence + Langevin
                        training for EB-LNN.
  - PIDTrainer        : No gradient training. Performs a small grid search
                        over PID gains on the validation set.
"""
from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Re-use the EB-LNN losses + Langevin sampler from the eblnn package.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from eblnn.src.losses import JointLoss  # noqa: E402
from eblnn.src.sampler import LangevinSampler, ReplayBuffer  # noqa: E402

from .models import EBLNNWrapper  # noqa: E402
from .pid import PIDPredictor, fit_pid_gains  # noqa: E402


# ---------------------------------------------------------------------------
# Supervised MSE trainer (LNN, LSTM)
# ---------------------------------------------------------------------------


class SupervisedTrainer:
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

    def _train_epoch(self, loader: DataLoader) -> float:
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

    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred, _ = self.model(x_batch)
                total_loss += self.criterion(pred, y_batch).item()
        return total_loss / len(loader)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: str,
        epoch_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
        log_prefix: str = "",
        verbose: bool = True,
    ) -> None:
        os.makedirs(save_path, exist_ok=True)
        self.best_model_path = os.path.join(save_path, "best_model.pth")

        t0 = time.time()
        for epoch in range(self.epochs):
            ep_t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            improved = val_loss < self.best_val_loss - self.min_delta
            if improved:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.epochs_no_improve = 0
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_state_dict, self.best_model_path)
            else:
                self.epochs_no_improve += 1

            if verbose:
                tag = "*" if improved else " "
                print(
                    f"{log_prefix}epoch {epoch + 1:3d}/{self.epochs} {tag} "
                    f"train={train_loss:.5f}  val={val_loss:.5f}  "
                    f"best={self.best_val_loss:.5f}@{self.best_epoch}  "
                    f"pat={self.epochs_no_improve}/{self.patience}  "
                    f"dt={time.time() - ep_t0:.1f}s  elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )

            if epoch_callback is not None:
                epoch_callback(epoch + 1, {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": self.best_val_loss,
                    "epochs_no_improve": self.epochs_no_improve,
                })

            if self.early_stopping and self.epochs_no_improve >= self.patience:
                if verbose:
                    print(
                        f"{log_prefix}early stop at epoch {epoch + 1} "
                        f"(best val={self.best_val_loss:.5f} @ {self.best_epoch})",
                        flush=True,
                    )
                break

        torch.save(self.model.state_dict(), os.path.join(save_path, "last_model.pth"))

    def load_best_model(self) -> None:
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            return
        if self.best_model_path is None or not os.path.exists(self.best_model_path):
            raise FileNotFoundError("Best model checkpoint is not available.")
        state_dict = torch.load(
            self.best_model_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)

    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                pred, _ = self.model(x_batch)
                all_true.append(y_batch.cpu().numpy())
                all_pred.append(pred.cpu().numpy())
        y_true = np.concatenate(all_true).reshape(-1, 2)
        y_pred = np.concatenate(all_pred).reshape(-1, 2)
        return y_true, y_pred


# ---------------------------------------------------------------------------
# Contrastive-Divergence trainer (EB-LNN)
# ---------------------------------------------------------------------------


class CDTrainer:
    """
    Joint trainer for EB-LNN. Each step:

      1. Forward pass of positives (real batch) -> phys_pred, e_pos
      2. Sample negatives via Langevin from the replay buffer
      3. Forward pass of negatives -> e_neg
      4. JointLoss = MSE(phys_pred, y) + alpha * (e_pos - e_neg + reg)
      5. Backprop, gradient clip, Adam step

    Best-checkpoint criterion is the *physics validation MSE*
    (NOT the joint loss — CD can legitimately go negative).
    """

    def __init__(
        self,
        model: EBLNNWrapper,
        config: Dict,
        eblnn_cfg: Dict,
        seq_len: int,
        input_size: int,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.device = device
        self.config = config

        self.epochs = config.get("epochs", 200)
        self.lr = config.get("learning_rate", 1e-3)
        self.patience = config.get("patience", 20)
        self.min_delta = config.get("min_delta", 1e-4)
        self.early_stopping = config.get("early_stopping", True)

        self.alpha = eblnn_cfg.get("alpha", 1.0)
        self.l2_reg = eblnn_cfg.get("l2_reg", 0.01)
        self.margin = eblnn_cfg.get("margin", 0.0)
        self.energy_clamp = eblnn_cfg.get("energy_clamp", 20.0)
        self.buffer_prob = eblnn_cfg.get("buffer_prob", 0.95)

        # Replay buffer + Langevin sampler.
        self.replay_buffer = ReplayBuffer(
            capacity=eblnn_cfg.get("buffer_capacity", 10_000),
            input_shape=(seq_len, input_size),
            device=device,
        )
        lan_cfg = eblnn_cfg.get("langevin", {})
        self.sampler = LangevinSampler(
            energy_fn=model.energy,
            n_steps=lan_cfg.get("n_steps", 20),
            step_size=lan_cfg.get("step_size", 0.01),
            noise_scale=lan_cfg.get("noise_scale", 0.005),
            clip_x=lan_cfg.get("clip_x", 3.0),
        )

        self.criterion = JointLoss(
            alpha=self.alpha,
            l2_reg=self.l2_reg,
            margin=self.margin,
            energy_clamp=self.energy_clamp,
        )
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_phys": [], "val_phys": [],
            "train_cd": [],   "val_cd": [],
            "e_pos": [], "e_neg": [], "cd_gap": [],
        }
        self.best_val_loss = float("inf")    # tracks val *physics* MSE
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.best_state_dict: Optional[dict] = None
        self.best_model_path: Optional[str] = None

    # ------------------------------------------------------------------

    def _get_fantasy(self, batch_size: int) -> torch.Tensor:
        x_init, buf_idxs = self.replay_buffer.sample(
            batch_size,
            buffer_prob=self.buffer_prob,
        )
        x_init = x_init.to(self.device)

        self.model.eval()
        x_neg = self.sampler.sample(x_init)
        self.model.train()

        self.replay_buffer.update(buf_idxs, x_neg.cpu())
        return x_neg

    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        totals = {"loss": 0.0, "phys": 0.0, "cd": 0.0,
                  "e_pos": 0.0, "e_neg": 0.0, "cd_gap": 0.0}

        for x_pos, y_pos in loader:
            x_pos = x_pos.to(self.device)
            y_pos = y_pos.to(self.device)
            B = x_pos.size(0)

            phys_pred, e_pos_seq, _ = self.model.core(x_pos)
            e_pos = e_pos_seq.squeeze(-1).mean(dim=1)

            x_neg = self._get_fantasy(B)
            _, e_neg_seq, _ = self.model.core(x_neg)
            e_neg = e_neg_seq.squeeze(-1).mean(dim=1)

            loss, metrics, _ = self.criterion(phys_pred, y_pos, e_pos, e_neg)

            self.optimizer.zero_grad()
            loss.backward()
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

    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        totals = {"loss": 0.0, "phys": 0.0, "cd": 0.0}
        with torch.no_grad():
            for x_pos, y_pos in loader:
                x_pos = x_pos.to(self.device)
                y_pos = y_pos.to(self.device)

                phys_pred, e_pos_seq, _ = self.model.core(x_pos)
                e_pos = e_pos_seq.squeeze(-1).mean(dim=1)

                # Validation negatives: fresh noise (Langevin too slow here).
                x_neg = torch.randn_like(x_pos)
                _, e_neg_seq, _ = self.model.core(x_neg)
                e_neg = e_neg_seq.squeeze(-1).mean(dim=1)

                _, metrics, _ = self.criterion(phys_pred, y_pos, e_pos, e_neg)
                totals["loss"] += metrics["loss_total"]
                totals["phys"] += metrics["loss_physics"]
                totals["cd"]   += metrics["loss_cd"]

        n = len(loader)
        return {k: v / n for k, v in totals.items()}

    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: str,
        epoch_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
        log_prefix: str = "",
        verbose: bool = True,
    ) -> None:
        os.makedirs(save_path, exist_ok=True)
        self.best_model_path = os.path.join(save_path, "best_model.pth")

        t0 = time.time()
        for epoch in range(self.epochs):
            ep_t0 = time.time()
            tr = self._train_epoch(train_loader)
            va = self._validate(val_loader)

            self.history["train_loss"].append(tr["loss"])
            self.history["val_loss"].append(va["phys"])  # physics-only for plots
            self.history["train_phys"].append(tr["phys"])
            self.history["val_phys"].append(va["phys"])
            self.history["train_cd"].append(tr["cd"])
            self.history["val_cd"].append(va["cd"])
            self.history["e_pos"].append(tr["e_pos"])
            self.history["e_neg"].append(tr["e_neg"])
            self.history["cd_gap"].append(tr["cd_gap"])

            checkpoint_metric = va["phys"]
            improved = checkpoint_metric < self.best_val_loss - self.min_delta
            if improved:
                self.best_val_loss = checkpoint_metric
                self.best_epoch = epoch + 1
                self.epochs_no_improve = 0
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_state_dict, self.best_model_path)
            else:
                self.epochs_no_improve += 1

            if verbose:
                tag = "*" if improved else " "
                print(
                    f"{log_prefix}epoch {epoch + 1:3d}/{self.epochs} {tag} "
                    f"loss={tr['loss']:.4f}  phys(tr/va)={tr['phys']:.4f}/{va['phys']:.4f}  "
                    f"cd(tr/va)={tr['cd']:+.4f}/{va['cd']:+.4f}  "
                    f"e+={tr['e_pos']:+.3f} e-={tr['e_neg']:+.3f} gap={tr['cd_gap']:+.3f}  "
                    f"best={self.best_val_loss:.4f}@{self.best_epoch}  "
                    f"pat={self.epochs_no_improve}/{self.patience}  "
                    f"dt={time.time() - ep_t0:.1f}s  elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )

            if epoch_callback is not None:
                epoch_callback(epoch + 1, {
                    "train_loss":  tr["loss"],
                    "train_phys":  tr["phys"],
                    "train_cd":    tr["cd"],
                    "val_loss":    va["phys"],
                    "val_phys":    va["phys"],
                    "val_cd":      va["cd"],
                    "e_pos":       tr["e_pos"],
                    "e_neg":       tr["e_neg"],
                    "cd_gap":      tr["cd_gap"],
                    "best_val_phys": self.best_val_loss,
                    "epochs_no_improve": self.epochs_no_improve,
                })

            if self.early_stopping and self.epochs_no_improve >= self.patience:
                if verbose:
                    print(
                        f"{log_prefix}early stop at epoch {epoch + 1} "
                        f"(best val_phys={self.best_val_loss:.4f} @ {self.best_epoch})",
                        flush=True,
                    )
                break

        torch.save(self.model.state_dict(), os.path.join(save_path, "last_model.pth"))

    def load_best_model(self) -> None:
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            return
        if self.best_model_path is None or not os.path.exists(self.best_model_path):
            raise FileNotFoundError("Best model checkpoint is not available.")
        state_dict = torch.load(
            self.best_model_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)

    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                pred, _ = self.model(x_batch)
                all_true.append(y_batch.cpu().numpy())
                all_pred.append(pred.cpu().numpy())
        y_true = np.concatenate(all_true).reshape(-1, 2)
        y_pred = np.concatenate(all_pred).reshape(-1, 2)
        return y_true, y_pred


# ---------------------------------------------------------------------------
# PID grid-search "trainer"
# ---------------------------------------------------------------------------


class PIDTrainer:
    """
    No gradient training. Searches a small grid of PID gains against
    the validation set and stores the best gains.

    `history` carries one synthetic 'epoch' so downstream code that
    expects a non-empty history list does not break.
    """

    def __init__(
        self,
        model: PIDPredictor,
        pid_cfg: Dict,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.pid_cfg = pid_cfg
        self.device = device

        self.history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        self.best_val_loss = float("inf")
        self.best_epoch = 1
        self.best_state_dict: Optional[dict] = None
        self.best_model_path: Optional[str] = None
        self.best_gains = None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: str,
        epoch_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
        log_prefix: str = "",
        verbose: bool = True,
    ) -> None:
        os.makedirs(save_path, exist_ok=True)
        self.best_model_path = os.path.join(save_path, "best_model.pth")

        if verbose:
            print(f"{log_prefix}PID grid search starting ...", flush=True)
        t0 = time.time()

        gains, val_mse_phys = fit_pid_gains(
            self.model, train_loader, val_loader, self.pid_cfg, self.device
        )
        self.best_gains = gains
        self.best_val_loss = val_mse_phys

        # Single-point history just to keep the run script happy.
        self.history["train_loss"].append(val_mse_phys)
        self.history["val_loss"].append(val_mse_phys)

        if verbose:
            print(
                f"{log_prefix}PID grid search done in {time.time() - t0:.1f}s | "
                f"val_mse(phys)={val_mse_phys:.5f} | "
                f"Kp_t={gains.kp_temp} Ki_t={gains.ki_temp} Kd_t={gains.kd_temp} "
                f"Kp_o2={gains.kp_o2} | sp_t={gains.sp_temp:.2f} sp_o2={gains.sp_o2:.2f} "
                f"afr_sp={gains.afr_sp:.2f}",
                flush=True,
            )

        if epoch_callback is not None:
            epoch_callback(1, {
                "train_loss": val_mse_phys,
                "val_loss":   val_mse_phys,
                "best_val_loss": val_mse_phys,
            })

        self.best_state_dict = copy.deepcopy(self.model.state_dict())
        torch.save(self.best_state_dict, self.best_model_path)

    def load_best_model(self) -> None:
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            return
        if self.best_model_path is None or not os.path.exists(self.best_model_path):
            raise FileNotFoundError("Best PID checkpoint is not available.")
        state_dict = torch.load(
            self.best_model_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)

    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        all_true, all_pred = [], []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            pred, _ = self.model(x_batch)
            all_true.append(y_batch.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
        y_true = np.concatenate(all_true).reshape(-1, 2)
        y_pred = np.concatenate(all_pred).reshape(-1, 2)
        return y_true, y_pred


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_trainer(
    model_name: str,
    model: nn.Module,
    train_cfg: Dict,
    device: str,
    eblnn_cfg: Optional[Dict] = None,
    pid_cfg: Optional[Dict] = None,
    seq_len: Optional[int] = None,
    input_size: Optional[int] = None,
):
    name = model_name.lower()
    if name in {"lnn", "lstm"}:
        return SupervisedTrainer(model=model, config=train_cfg, device=device)
    if name == "eblnn":
        if eblnn_cfg is None or seq_len is None or input_size is None:
            raise ValueError("CDTrainer needs eblnn_cfg, seq_len, input_size.")
        return CDTrainer(
            model=model,
            config=train_cfg,
            eblnn_cfg=eblnn_cfg,
            seq_len=seq_len,
            input_size=input_size,
            device=device,
        )
    if name == "pid":
        return PIDTrainer(model=model, pid_cfg=pid_cfg or {}, device=device)
    raise ValueError(f"No trainer for model: {model_name}")
