"""
Langevin Dynamics Sampler
=========================
Generates "fantasy" (negative) samples via Markov-Chain Monte Carlo
guided by gradients of the learned energy function.

Theory
------
Standard Langevin update rule:

    x_{t+1} = x_t  −  (step_size / 2) · ∇_x E_θ(x_t)
                    +  sqrt(step_size) · ε,    ε ~ N(0, I)

Iterating from random noise (or a buffer-stored past sample), this
drives x toward the low-energy regions of the learned manifold.
The resulting x_T are the "fantasy states" used as negative samples
in Contrastive Divergence training.

Replay Buffer
-------------
Instead of initialising every Langevin chain from pure noise (expensive),
we maintain a buffer of previously-generated negatives.  Each call samples
from the buffer with probability `buffer_prob`; the rest are fresh noise.
This is the standard trick from the JEM / IGEBM papers that greatly reduces
the number of Langevin steps needed per training iteration.

Reference: Du & Mordatch, "Implicit Generation and Modeling with Energy
           Based Models", NeurIPS 2019.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-capacity FIFO/random-replacement buffer for Langevin fantasy states.

    Each stored item has the same shape as the model's input tensor:
        (seq_len, input_size)   — one sequence per slot.

    Parameters
    ----------
    capacity   : int   — maximum number of sequences stored
    input_shape: tuple — (seq_len, input_size) of one sequence
    device     : str
    """

    def __init__(
        self,
        capacity: int,
        input_shape: tuple[int, int],
        device: str = "cpu",
    ) -> None:
        self.capacity = capacity
        self.device = device

        # Pre-allocate storage — initialised with random noise
        self._buffer = torch.randn(capacity, *input_shape, device=device)
        self._size = 0          # number of valid entries written so far
        self._ptr = 0           # write pointer

    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        noise_init: Optional[torch.Tensor] = None,
        buffer_prob: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Draw `batch_size` sequences.

        Each sample is:
          - drawn from the buffer with probability `buffer_prob`
          - replaced with fresh Gaussian noise otherwise

        Returns
        -------
        samples  : (batch_size, *input_shape)  — initial states for Langevin
        buf_idxs : (batch_size,)               — buffer indices (for update)
        """
        # Random buffer indices
        buf_idxs = torch.randint(
            0, max(self._size, 1), (batch_size,), device=self.device
        )
        samples = self._buffer[buf_idxs].clone()

        # Replace some with fresh noise
        if buffer_prob < 1.0:
            fresh_mask = (torch.rand(batch_size) > buffer_prob).to(self.device)
            if fresh_mask.any():
                if noise_init is not None:
                    samples[fresh_mask] = noise_init[fresh_mask]
                else:
                    samples[fresh_mask] = torch.randn_like(samples[fresh_mask])

        return samples, buf_idxs

    def update(self, buf_idxs: torch.Tensor, new_samples: torch.Tensor) -> None:
        """Write back the refined Langevin samples to the buffer."""
        self._buffer[buf_idxs] = new_samples.detach().to(self.device)
        # Update size / pointer bookkeeping
        self._size = min(self._size + len(buf_idxs), self.capacity)

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Langevin Sampler
# ---------------------------------------------------------------------------

class LangevinSampler:
    """
    MCMC sampler that drives candidate states toward energy minima.

    The sampler operates in the *input space* of the model (sequence space).
    Gradients flow through the full CfC → EBM head pipeline, so the
    resulting fantasy states are meaningful in the input distribution.

    Parameters
    ----------
    energy_fn   : callable  — function(x) → scalar energy per batch item (B,)
                              Use ``model.energy`` from EBLNN_Generative.
    n_steps     : int       — number of Langevin update steps (default 20)
    step_size   : float     — gradient step size  α  (default 0.01)
    noise_scale : float     — noise coefficient   σ  (default 0.005)
    clip_x      : float|None— hard clip for |x| after each step (optional)
    projector   : callable|None
                            — optional projection after each step
                              e.g. clamp to normalised data range
    """

    def __init__(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        n_steps: int = 20,
        step_size: float = 0.01,
        noise_scale: float = 0.005,
        clip_x: Optional[float] = None,
        projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.energy_fn = energy_fn
        self.n_steps = n_steps
        self.step_size = step_size
        self.noise_scale = noise_scale
        self.clip_x = clip_x
        self.projector = projector

    # ------------------------------------------------------------------

    def sample(
        self,
        x_init: torch.Tensor,
        requires_grad_output: bool = False,
    ) -> torch.Tensor:
        """
        Run `n_steps` of Langevin dynamics from `x_init`.

        Parameters
        ----------
        x_init : (B, T, input_size)  — initial states (from buffer or noise)

        Returns
        -------
        x_fantasy : (B, T, input_size)  — fantasy (negative) states
        """
        x = x_init.clone().detach()
        x.requires_grad_(True)

        for _ in range(self.n_steps):
            # --- forward: compute energy ---
            energy = self.energy_fn(x)          # (B,)
            energy_sum = energy.sum()

            # --- backward: gradient w.r.t. input ---
            if x.grad is not None:
                x.grad.zero_()
            energy_sum.backward()
            grad = x.grad.detach()              # (B, T, input_size)

            # --- Langevin update ---
            noise = torch.randn_like(x) * self.noise_scale
            x_new = x.detach() - self.step_size * grad + noise

            # --- optional projection / clipping ---
            if self.clip_x is not None:
                x_new = x_new.clamp(-self.clip_x, self.clip_x)
            if self.projector is not None:
                x_new = self.projector(x_new)

            x = x_new.detach()
            x.requires_grad_(True)

        return x.detach().requires_grad_(requires_grad_output)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_no_grad(self, x_init: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper that runs Langevin purely for inference/viz —
        gradients are not tracked (much faster, less memory).

        Note: gradients w.r.t. the *model parameters* are still used
        internally during forward passes; this wrapper only skips
        tracking gradients of the output w.r.t. computation graph.
        Use ``sample()`` (with grad tracking) during training.
        """
        return self.sample(x_init, requires_grad_output=False)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_sampler(
    model: nn.Module,
    n_steps: int = 20,
    step_size: float = 0.01,
    noise_scale: float = 0.005,
    clip_x: Optional[float] = 3.0,
) -> LangevinSampler:
    """
    Build a LangevinSampler wired to ``model.energy``.

    Parameters
    ----------
    model      : EBLNN_Generative instance
    n_steps    : Langevin steps per call
    step_size  : gradient step size
    noise_scale: Gaussian noise magnitude
    clip_x     : hard-clip range for x (None to disable)

    Returns
    -------
    LangevinSampler
    """
    return LangevinSampler(
        energy_fn=model.energy,
        n_steps=n_steps,
        step_size=step_size,
        noise_scale=noise_scale,
        clip_x=clip_x,
    )
