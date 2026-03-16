"""
Standalone Liquid Neural Network (LNN) — Pure CfC
===================================================
For the ablation study showing that PINN-augmented edge cases
improve LNN performance on furnace next-state prediction.

Architecture
------------
    (State_t, Action_t)
          ↓
    [CfC Body]  — Closed-form Continuous-time LNN core
          ↓
    Hidden State h_t
          └── [Physics Head]  →  Next_State_Pred   (MSE vs ground truth)

No EBM head, no contrastive divergence.  Simple supervised regression.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from ncps.torch import CfC


# ---------------------------------------------------------------------------
# Physics Head  (identical to EBLNN's, duplicated for independence)
# ---------------------------------------------------------------------------

class PhysicsHead(nn.Module):
    """Linear head: hidden state → [next_temp, next_excess_o2]."""

    def __init__(self, hidden_size: int, output_size: int = 2) -> None:
        super().__init__()
        self.net = nn.Linear(hidden_size, output_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ---------------------------------------------------------------------------
# Standalone LNN (CfC + PhysicsHead)
# ---------------------------------------------------------------------------

class LNN(nn.Module):
    """
    Pure Liquid Neural Network for next-state prediction.

    Parameters
    ----------
    input_size       : int   — number of input features (default 5)
    hidden_size      : int   — CfC hidden size (default 128)
    phys_output_size : int   — prediction outputs (default 2: temp, O2)
    mixed_memory     : bool  — CfC mixed-memory flag (default True)
    batch_first      : bool  — (default True)
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        phys_output_size: int = 2,
        mixed_memory: bool = True,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

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
        """
        Parameters
        ----------
        x  : (B, T, input_size)
        hx : optional initial hidden state

        Returns
        -------
        phys_pred : (B, T, phys_output_size)
        last_h    : final hidden state
        """
        h_seq, last_h = self.cfc_body(x, hx)
        phys_pred = self.phys_head(h_seq)
        return phys_pred, last_h

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> None:
        total = self.get_num_parameters()
        cfc_p = sum(p.numel() for p in self.cfc_body.parameters() if p.requires_grad)
        phys_p = sum(p.numel() for p in self.phys_head.parameters() if p.requires_grad)
        print(f"LNN — {total:,} total trainable parameters")
        print(f"  CfC backbone  : {cfc_p:,}")
        print(f"  Physics head  : {phys_p:,}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_lnn_model(
    input_size: int = 5,
    hidden_size: int = 128,
    phys_output_size: int = 2,
    device: str = "cpu",
) -> LNN:
    """Create a standalone LNN and move to device."""
    model = LNN(
        input_size=input_size,
        hidden_size=hidden_size,
        phys_output_size=phys_output_size,
    )
    model = model.to(device)
    model.parameter_summary()
    return model
