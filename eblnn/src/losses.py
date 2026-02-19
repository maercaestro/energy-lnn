"""
Loss Functions for Generative EB-LNN
=====================================

Two losses, trained jointly:

    L_total = L_physics  +  α · L_CD

1. Physics Loss  (L_physics)
   ──────────────────────────
   Standard MSE between the physics head's predicted next state and
   the ground truth.  Identical to the pilot study.

       L_physics = MSE(ŷ_t+1 , y_t+1)

2. Contrastive Divergence Loss  (L_CD)
   ────────────────────────────────────
   Shapes the energy manifold so that:
     • real data   →  LOW  energy  (model assigns low cost)
     • fantasy data→  HIGH energy  (generated negatives are costly)

   Base CD objective:

       L_CD = E_θ(x_pos) − E_θ(x_neg)

   Minimising L_CD pushes E(x_pos) down and E(x_neg) up.

   Optional regularisation term (controlled by `l2_reg`):

       L_CD += λ · [ E_θ(x_pos)² + E_θ(x_neg)² ]

   This prevents the energy from collapsing to ±∞ and stabilises
   training, especially early in the run.

   Reference: Hinton (2002); Du & Mordatch, NeurIPS 2019.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Physics Loss
# ---------------------------------------------------------------------------

class PhysicsLoss(nn.Module):
    """
    MSE between the physics head's prediction and the ground-truth next state.

    Parameters
    ----------
    reduction : str — 'mean' | 'sum' (default 'mean')
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        predicted : (B, T, phys_output_size)
        target    : (B, T, phys_output_size)

        Returns
        -------
        loss : scalar tensor
        """
        return self.mse(predicted, target)


# ---------------------------------------------------------------------------
# Contrastive Divergence Loss
# ---------------------------------------------------------------------------

class ContrastiveDivergenceLoss(nn.Module):
    """
    Contrastive Divergence loss for EBM head training.

    Drives the energy surface so that:
        E(real states)    is minimised
        E(fantasy states) is maximised

    Parameters
    ----------
    l2_reg  : float — coefficient for energy-magnitude regularisation
                      (set to 0 to disable).  Typical range: 0.0 – 0.1.
    margin  : float — optional hinge margin.  If > 0, the loss only
                      penalises when E(neg) − E(pos) < margin, giving
                      a safety buffer.  Set 0 for the standard CD.
    """

    def __init__(self, l2_reg: float = 0.01, margin: float = 0.0) -> None:
        super().__init__()
        self.l2_reg = l2_reg
        self.margin = margin

    def forward(
        self,
        energy_pos: torch.Tensor,
        energy_neg: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute contrastive divergence loss.

        Parameters
        ----------
        energy_pos : (B,)  — energies of real (positive) samples
        energy_neg : (B,)  — energies of fantasy (negative) samples

        Returns
        -------
        loss    : scalar tensor
        metrics : dict with 'e_pos', 'e_neg', 'cd_gap' for logging
        """
        e_pos = energy_pos.mean()
        e_neg = energy_neg.mean()

        if self.margin > 0:
            # Hinge variant: only penalise when gap is too small
            cd_loss = F.relu(e_pos - e_neg + self.margin)
        else:
            # Standard CD
            cd_loss = e_pos - e_neg

        # L2 regularisation on energy magnitude
        if self.l2_reg > 0:
            reg = self.l2_reg * (energy_pos.pow(2).mean() + energy_neg.pow(2).mean())
            cd_loss = cd_loss + reg

        metrics = {
            "e_pos": e_pos.item(),
            "e_neg": e_neg.item(),
            "cd_gap": (e_neg - e_pos).item(),   # we want this > 0
        }

        return cd_loss, metrics


# ---------------------------------------------------------------------------
# Joint Loss
# ---------------------------------------------------------------------------

class JointLoss(nn.Module):
    """
    Combines physics loss and contrastive divergence loss:

        L_total = L_physics  +  α · L_CD

    Parameters
    ----------
    alpha   : float — weight for the CD term (default 1.0)
    l2_reg  : float — passed to ContrastiveDivergenceLoss
    margin  : float — passed to ContrastiveDivergenceLoss
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l2_reg: float = 0.01,
        margin: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.phys_loss = PhysicsLoss()
        self.cd_loss = ContrastiveDivergenceLoss(l2_reg=l2_reg, margin=margin)

    def forward(
        self,
        phys_pred: torch.Tensor,
        phys_target: torch.Tensor,
        energy_pos: torch.Tensor,
        energy_neg: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Parameters
        ----------
        phys_pred   : (B, T, phys_output_size)  — physics head output
        phys_target : (B, T, phys_output_size)  — ground truth next state
        energy_pos  : (B,)  — EBM energy of real sequences
        energy_neg  : (B,)  — EBM energy of fantasy sequences

        Returns
        -------
        total_loss : scalar tensor (gradient-enabled)
        metrics    : dict for WandB / console logging
        """
        l_phys = self.phys_loss(phys_pred, phys_target)
        l_cd, cd_metrics = self.cd_loss(energy_pos, energy_neg)

        total = l_phys + self.alpha * l_cd

        metrics = {
            "loss_total": total.item(),
            "loss_physics": l_phys.item(),
            "loss_cd": l_cd.item(),
            **cd_metrics,
        }

        return total, metrics
