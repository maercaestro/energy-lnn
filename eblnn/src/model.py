"""
Energy-Based Liquid Neural Network — Generative Architecture
=============================================================
Upgrade over the pilot study:
  - Pilot:  EBM Head = nn.Linear  →  trained via MSE vs hardcoded energy
  - Here:   EBM Head = MLP         →  trained via Contrastive Divergence
                                      with Langevin-generated fantasy states.
             No hardcoded energy formula. The landscape is LEARNED.

Architecture
------------
    (State_t, Action_t)
          ↓
    [CfC Body]  — Closed-form Continuous-time LNN core
          ↓
    Hidden State h_t
          ├── [Physics Head]  →  Next_State_Pred   (MSE vs ground truth)
          └── [EBM Head MLP]  →  Scalar_Energy E   (CD vs fantasy states)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from ncps.torch import CfC


# ---------------------------------------------------------------------------
# Energy MLP Head
# ---------------------------------------------------------------------------

class EBMHead(nn.Module):
    """
    Multi-layer perceptron that maps a hidden state to a *scalar energy score*.

    The sign convention follows the standard EBM literature:
        - Low  energy  ↔  high-probability / desirable configuration
        - High energy  ↔  low-probability  / undesirable / fantasy state

    Unlike the pilot's single Linear layer supervised against a formula,
    this head's weights are shaped entirely by Contrastive Divergence.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the CfC hidden state fed in.
    ebm_hidden_dims : list[int]
        Widths of the intermediate layers (default [128, 64]).
    activation : nn.Module
        Non-linearity between layers (default SiLU for smooth gradients).
    """

    def __init__(
        self,
        hidden_size: int,
        ebm_hidden_dims: list[int] | None = None,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()

        if ebm_hidden_dims is None:
            ebm_hidden_dims = [128, 64]
        if activation is None:
            activation = nn.SiLU()

        layers: list[nn.Module] = []
        in_dim = hidden_size
        for out_dim in ebm_hidden_dims:
            layers += [nn.Linear(in_dim, out_dim), activation]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))   # scalar output

        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : (B, T, hidden_size)  — sequence of hidden states

        Returns
        -------
        energy : (B, T, 1)  — per-timestep energy scores
        """
        return self.net(h)


# ---------------------------------------------------------------------------
# Physics Head  (identical role to the pilot, kept separate for clarity)
# ---------------------------------------------------------------------------

class PhysicsHead(nn.Module):
    """
    Linear head that predicts the next physical state from the hidden state.
    [next_temp, next_excess_o2] — same as the pilot study.
    """

    def __init__(self, hidden_size: int, output_size: int = 2) -> None:
        super().__init__()
        self.net = nn.Linear(hidden_size, output_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : (B, T, hidden_size)

        Returns
        -------
        pred : (B, T, output_size)
        """
        return self.net(h)


# ---------------------------------------------------------------------------
# Full EB-LNN (Generative)
# ---------------------------------------------------------------------------

class EBLNN_Generative(nn.Module):
    """
    Energy-Based Liquid Neural Network with a *generative search space*.

    Components
    ----------
    cfc_body    : CfC (Closed-form Continuous-time) backbone
    phys_head   : PhysicsHead — predicts next physical state (MSE-trained)
    ebm_head    : EBMHead MLP — learns energy manifold (CD-trained)

    The EBM head is *not* supervised against any formula.  Its landscape is
    shaped by Contrastive Divergence: real states are pushed to low energy,
    Langevin-generated fantasy states are pushed to high energy.

    Parameters
    ----------
    input_size       : int   — number of input features
    hidden_size      : int   — CfC hidden size
    phys_output_size : int   — number of physics prediction outputs (default 2)
    ebm_hidden_dims  : list  — MLP widths for EBM head (default [128, 64])
    mixed_memory     : bool  — CfC mixed-memory flag (default True)
    batch_first      : bool  — (default True)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        phys_output_size: int = 2,
        ebm_hidden_dims: list[int] | None = None,
        mixed_memory: bool = True,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # --- Backbone ---
        self.cfc_body = CfC(
            input_size,
            hidden_size,
            mixed_memory=mixed_memory,
            batch_first=batch_first,
        )

        # --- Head 1: Physics (dynamics prediction) ---
        self.phys_head = PhysicsHead(hidden_size, phys_output_size)

        # --- Head 2: EBM (energy manifold — no formula, learned via CD) ---
        self.ebm_head = EBMHead(hidden_size, ebm_hidden_dims)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Parameters
        ----------
        x  : (B, T, input_size)
        hx : optional initial hidden state

        Returns
        -------
        phys_pred : (B, T, phys_output_size)  — next-state predictions
        energy    : (B, T, 1)                 — per-timestep energy scores
        last_h    : final hidden state
        """
        h_seq, last_h = self.cfc_body(x, hx)   # (B, T, H)
        phys_pred = self.phys_head(h_seq)        # (B, T, phys_output_size)
        energy = self.ebm_head(h_seq)            # (B, T, 1)
        return phys_pred, energy, last_h

    # ------------------------------------------------------------------
    # Energy-only path (used inside the Langevin sampler)
    # ------------------------------------------------------------------

    def energy(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scalar energy for input `x` — needed for Langevin gradients.

        Returns a *single scalar per batch item* (mean over time):
            shape (B,)
        """
        h_seq, _ = self.cfc_body(x, hx)
        e_seq = self.ebm_head(h_seq)          # (B, T, 1)
        return e_seq.squeeze(-1).mean(dim=1)  # (B,)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> None:
        total = self.get_num_parameters()
        cfc_p = sum(p.numel() for p in self.cfc_body.parameters() if p.requires_grad)
        phys_p = sum(p.numel() for p in self.phys_head.parameters() if p.requires_grad)
        ebm_p = sum(p.numel() for p in self.ebm_head.parameters() if p.requires_grad)
        print(f"EBLNN_Generative — {total:,} total trainable parameters")
        print(f"  CfC backbone  : {cfc_p:,}")
        print(f"  Physics head  : {phys_p:,}")
        print(f"  EBM head      : {ebm_p:,}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(
    input_size: int = 5,
    hidden_size: int = 128,
    phys_output_size: int = 2,
    ebm_hidden_dims: list[int] | None = None,
    device: str = "cpu",
) -> EBLNN_Generative:
    """
    Convenience factory — create and move model to device.

    Parameters
    ----------
    input_size       : number of input features
    hidden_size      : CfC hidden size
    phys_output_size : physics head outputs (default 2: temp + O2)
    ebm_hidden_dims  : MLP layers for EBM head (default [128, 64])
    device           : 'cpu' | 'cuda'

    Returns
    -------
    EBLNN_Generative instance on `device`
    """
    if ebm_hidden_dims is None:
        ebm_hidden_dims = [128, 64]

    model = EBLNN_Generative(
        input_size=input_size,
        hidden_size=hidden_size,
        phys_output_size=phys_output_size,
        ebm_hidden_dims=ebm_hidden_dims,
    ).to(device)

    model.parameter_summary()
    print(f"  Input size    : {input_size}")
    print(f"  Hidden size   : {hidden_size}")
    print(f"  Phys outputs  : {phys_output_size}  (temp, O2)")
    print(f"  EBM dims      : {ebm_hidden_dims}")
    print(f"  Device        : {device}")

    return model
