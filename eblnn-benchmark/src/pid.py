"""
PID controller-style baseline predictor.

Motivation
----------
The thesis benchmark compares EB-LNN against an LSTM and a Liquid NN.
A PID controller is the workhorse baseline of industrial process control,
so it makes sense to include a controller-style predictor in the lineup
even though PID is not a learning model.

Formulation
-----------
We treat the next-step prediction as the output of two independent control
laws operating in *physical units*:

  Temperature loop (driven by the temperature error toward a setpoint):

      e_T(t)        = SP_T - current_temp(t)
      next_temp(t)  = current_temp(t)
                      + Kp_T * e_T(t)
                      + Ki_T * Σ_{τ≤t} e_T(τ)
                      + Kd_T * (e_T(t) - e_T(t-1))

  Excess-O₂ loop (driven by the deviation of the air-fuel ratio
  from a reference AFR — the variable that physically controls O₂):

      e_O(t)        = AFR(t) - AFR_SP
      next_o2(t)    = SP_O2 + Kp_O2 * e_O(t)

The setpoints (`SP_T`, `SP_O2`, `AFR_SP`) and gains
(`Kp_T`, `Ki_T`, `Kd_T`, `Kp_O2`) are chosen by a small grid search on the
validation set against the supervised next-state targets.

The PIDPredictor is wrapped as a `nn.Module` so it plugs straight into
the shared evaluation pipeline (`model(x)[0]` returns the prediction).
It carries the dataset's `input_scaler` and `target_scaler` so that the
control law operates in the original physical units even though the
benchmark tensors are standardised.

Note
----
This is intentionally a *naive* baseline. The point is to show how
much (or little) headroom the learned models gain over a tuned
classical controller as a next-step predictor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# Indices into the shared INPUT_COLS layout from data.py.
# (Kept here as constants to avoid an import cycle with data.py.)
IDX_FUEL_FLOW = 0
IDX_AFR = 1
IDX_CURRENT_TEMP = 2
IDX_INFLOW_TEMP = 3
IDX_INFLOW_RATE = 4

TARGET_TEMP = 0
TARGET_O2 = 1


@dataclass
class PIDGains:
    sp_temp: float = 0.0
    kp_temp: float = 0.0
    ki_temp: float = 0.0
    kd_temp: float = 0.0
    sp_o2: float = 0.0
    afr_sp: float = 0.0
    kp_o2: float = 0.0


class PIDPredictor(nn.Module):
    """PID-style next-state predictor (no learnable parameters)."""

    def __init__(
        self,
        input_scaler,
        target_scaler,
        gains: Optional[PIDGains] = None,
    ) -> None:
        super().__init__()
        self._input_scaler = input_scaler
        self._target_scaler = target_scaler
        self._gains = gains or PIDGains()

        # Pre-compute scaler tensors as buffers so .to(device) works.
        self.register_buffer(
            "_in_mean",
            torch.tensor(input_scaler.mean_, dtype=torch.float32),
        )
        self.register_buffer(
            "_in_scale",
            torch.tensor(input_scaler.scale_, dtype=torch.float32),
        )
        self.register_buffer(
            "_out_mean",
            torch.tensor(target_scaler.mean_, dtype=torch.float32),
        )
        self.register_buffer(
            "_out_scale",
            torch.tensor(target_scaler.scale_, dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # Gain management
    # ------------------------------------------------------------------

    def set_gains(self, gains: PIDGains) -> None:
        self._gains = gains

    @property
    def gains(self) -> PIDGains:
        return self._gains

    # ------------------------------------------------------------------
    # Forward (physical-unit control law, returns standardised tensor)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        hx: None = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Parameters
        ----------
        x : (B, T, input_size)  — standardised inputs

        Returns
        -------
        pred : (B, T, 2)        — standardised predictions for [next_temp, next_o2]
        """
        x_phys = x * self._in_scale + self._in_mean
        current_temp = x_phys[..., IDX_CURRENT_TEMP]
        afr = x_phys[..., IDX_AFR]

        g = self._gains

        # --- Temperature PID loop, vectorised over time ---
        err_t = g.sp_temp - current_temp                          # (B, T)
        # Cumulative sum gives the discrete integral term.
        integ_t = torch.cumsum(err_t, dim=1)
        # Discrete derivative (zero at t=0).
        deriv_t = torch.zeros_like(err_t)
        deriv_t[:, 1:] = err_t[:, 1:] - err_t[:, :-1]

        next_temp_phys = (
            current_temp
            + g.kp_temp * err_t
            + g.ki_temp * integ_t
            + g.kd_temp * deriv_t
        )

        # --- O₂ proportional loop driven by AFR deviation ---
        err_o2 = afr - g.afr_sp
        next_o2_phys = g.sp_o2 + g.kp_o2 * err_o2

        pred_phys = torch.stack([next_temp_phys, next_o2_phys], dim=-1)  # (B, T, 2)
        pred_std = (pred_phys - self._out_mean) / self._out_scale
        return pred_std, None


# ---------------------------------------------------------------------------
# Gain search
# ---------------------------------------------------------------------------


def _val_mse_physical(model: PIDPredictor, val_loader, device: str) -> float:
    total_se = 0.0
    total_n = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            pred, _ = model(x_batch)
            # Convert both to physical units for a fair, dimension-balanced loss.
            pred_phys = pred * model._out_scale + model._out_mean
            y_phys = y_batch.to(device) * model._out_scale + model._out_mean
            err = pred_phys - y_phys
            total_se += float((err ** 2).sum().item())
            total_n += err.numel()
    return total_se / max(total_n, 1)


def fit_pid_gains(
    model: PIDPredictor,
    train_loader,
    val_loader,
    pid_cfg: dict,
    device: str,
) -> Tuple[PIDGains, float]:
    """
    Fit PID setpoints (from training-set means by default) and search
    over a small grid of gains to minimise validation MSE.

    Returns
    -------
    best_gains : PIDGains
    best_val_mse_phys : float
    """
    # --- Compute setpoints from the training data in physical units ---
    sp_temp = pid_cfg.get("setpoint_temp")
    sp_o2 = pid_cfg.get("setpoint_o2")
    afr_sp = pid_cfg.get("afr_setpoint")

    if sp_temp is None or sp_o2 is None or afr_sp is None:
        sum_y = np.zeros(2, dtype=np.float64)
        sum_afr = 0.0
        n_y = 0
        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_phys = (
                    x_batch.to(device) * model._in_scale + model._in_mean
                ).cpu().numpy()
                y_phys = (
                    y_batch.to(device) * model._out_scale + model._out_mean
                ).cpu().numpy()
                sum_y += y_phys.reshape(-1, 2).sum(axis=0)
                sum_afr += float(x_phys[..., IDX_AFR].sum())
                n_y += y_phys.shape[0] * y_phys.shape[1]
        if sp_temp is None:
            sp_temp = float(sum_y[TARGET_TEMP] / n_y)
        if sp_o2 is None:
            sp_o2 = float(sum_y[TARGET_O2] / n_y)
        if afr_sp is None:
            afr_sp = float(sum_afr / n_y)

    # --- Grid search ---
    kp_t_grid = pid_cfg.get("kp_temp_grid", [0.0, 0.5, 1.0])
    ki_t_grid = pid_cfg.get("ki_temp_grid", [0.0])
    kd_t_grid = pid_cfg.get("kd_temp_grid", [0.0])
    kp_o2_grid = pid_cfg.get("kp_o2_grid", [0.0, 0.5, 1.0])

    best = PIDGains(
        sp_temp=sp_temp,
        sp_o2=sp_o2,
        afr_sp=afr_sp,
    )
    best_mse = float("inf")

    for kp_t in kp_t_grid:
        for ki_t in ki_t_grid:
            for kd_t in kd_t_grid:
                for kp_o2 in kp_o2_grid:
                    gains = PIDGains(
                        sp_temp=sp_temp,
                        kp_temp=kp_t,
                        ki_temp=ki_t,
                        kd_temp=kd_t,
                        sp_o2=sp_o2,
                        afr_sp=afr_sp,
                        kp_o2=kp_o2,
                    )
                    model.set_gains(gains)
                    mse = _val_mse_physical(model, val_loader, device)
                    if mse < best_mse:
                        best_mse = mse
                        best = gains

    model.set_gains(best)
    return best, best_mse
