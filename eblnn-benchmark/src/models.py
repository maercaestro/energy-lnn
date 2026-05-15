"""
Models for the EB-LNN vs LNN vs LSTM vs PID benchmark.

All models expose the same minimal interface:

    pred, _ = model(x)            # x: (B, T, input_size)
                                  # pred: (B, T, target_size)

so they can be evaluated by the shared `evaluate.py` machinery
(which calls `model(x)[0]` to obtain the prediction tensor).

EB-LNN training is a special case (Contrastive Divergence + Langevin)
that is handled by `trainers.CDTrainer`, not by the supervised trainer.
PID has no learnable parameters — it is fitted by gain-search in
`trainers.PIDTrainer`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from ncps.torch import CfC
except ImportError:  # pragma: no cover
    CfC = None

# Re-use the EB-LNN core implementation from the eblnn package.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from eblnn.src.model import EBLNN_Generative  # noqa: E402

from .pid import PIDPredictor  # noqa: E402


# ---------------------------------------------------------------------------
# Heads / baselines
# ---------------------------------------------------------------------------


class PhysicsHead(nn.Module):
    def __init__(self, hidden_size: int, output_size: int = 2) -> None:
        super().__init__()
        self.net = nn.Linear(hidden_size, output_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class LNN(nn.Module):
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        phys_output_size: int = 2,
        mixed_memory: bool = True,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        if CfC is None:
            raise ImportError("ncps is required to use the LNN model.")
        self.cfc_body = CfC(
            input_size,
            hidden_size,
            mixed_memory=mixed_memory,
            batch_first=batch_first,
        )
        self.phys_head = PhysicsHead(hidden_size, phys_output_size)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_seq, last_h = self.cfc_body(x, hx)
        return self.phys_head(hidden_seq), last_h


class LSTMBaseline(nn.Module):
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        phys_output_size: int = 2,
        dropout: float = 0.1,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=batch_first,
        )
        self.phys_head = PhysicsHead(hidden_size, phys_output_size)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hidden_seq, last_h = self.lstm(x, hx)
        return self.phys_head(hidden_seq), last_h


# ---------------------------------------------------------------------------
# EB-LNN benchmark wrapper
# ---------------------------------------------------------------------------


class EBLNNWrapper(nn.Module):
    """
    Adapter around `EBLNN_Generative` that exposes the (pred, hidden) API
    expected by the shared evaluator.

    The full 3-tuple (phys_pred, energy, last_h) is still accessible via
    `wrapper.core(x)` for the contrastive-divergence trainer.
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        phys_output_size: int = 2,
        ebm_hidden_dims: Optional[list[int]] = None,
        mixed_memory: bool = True,
    ) -> None:
        super().__init__()
        self.core = EBLNN_Generative(
            input_size=input_size,
            hidden_size=hidden_size,
            phys_output_size=phys_output_size,
            ebm_hidden_dims=ebm_hidden_dims or [128, 64],
            mixed_memory=mixed_memory,
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        phys_pred, _energy, last_h = self.core(x, hx)
        return phys_pred, last_h

    # Convenience pass-through used by the CD trainer / Langevin sampler.
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return self.core.energy(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_model(
    model_name: str,
    input_size: int,
    target_size: int,
    model_config: dict,
    device: str,
    **kwargs,
) -> nn.Module:
    """
    Build a model by name.

    For PID, the caller MUST pass `input_scaler` and `target_scaler` via
    kwargs (they are needed because PID operates in physical units).
    """
    name = model_name.lower()
    if name == "lnn":
        model: nn.Module = LNN(
            input_size=input_size,
            hidden_size=model_config.get("hidden_size", 128),
            phys_output_size=target_size,
            mixed_memory=model_config.get("mixed_memory", True),
        )
    elif name == "lstm":
        model = LSTMBaseline(
            input_size=input_size,
            hidden_size=model_config.get("hidden_size", 128),
            num_layers=model_config.get("num_layers", 2),
            phys_output_size=target_size,
            dropout=model_config.get("dropout", 0.1),
        )
    elif name == "eblnn":
        model = EBLNNWrapper(
            input_size=input_size,
            hidden_size=model_config.get("hidden_size", 128),
            phys_output_size=target_size,
            ebm_hidden_dims=model_config.get("ebm_hidden_dims", [128, 64]),
            mixed_memory=model_config.get("mixed_memory", True),
        )
    elif name == "pid":
        input_scaler = kwargs.get("input_scaler")
        target_scaler = kwargs.get("target_scaler")
        if input_scaler is None or target_scaler is None:
            raise ValueError("PID model requires 'input_scaler' and 'target_scaler'.")
        model = PIDPredictor(
            input_scaler=input_scaler,
            target_scaler=target_scaler,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model.to(device)
