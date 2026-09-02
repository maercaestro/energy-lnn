"""
compute.py
==========
Energy computation functions for the landscape analysis.

All functions return plain numpy arrays so they are framework-agnostic
for downstream plotting and W&B logging.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


# ── per-sequence energy ────────────────────────────────────────────────────

@torch.no_grad()
def compute_energies(
    model: nn.Module,
    sequences: np.ndarray,
    device: str = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    """
    Run model.energy() on a (N, T, D) array and return energies (N,).

    Parameters
    ----------
    model     : EBLNNWrapper or any module exposing .energy(x) -> (B,)
    sequences : float32 array (N, seq_len, input_size)
    device    : torch device string
    batch_size: forward-pass batch size

    Returns
    -------
    energies : float32 array (N,)
    """
    model.eval()
    all_energies = []
    x_tensor = torch.from_numpy(sequences).float()

    for start in range(0, len(x_tensor), batch_size):
        batch = x_tensor[start : start + batch_size].to(device)
        e = model.energy(batch)          # (B,)
        all_energies.append(e.cpu().numpy())

    return np.concatenate(all_energies).astype(np.float32)


def compute_energies_per_scenario(
    model: nn.Module,
    scenario_sequences: Dict[str, np.ndarray],
    device: str = "cpu",
    batch_size: int = 512,
) -> Dict[str, np.ndarray]:
    """
    Compute energies for every scenario group.

    Returns
    -------
    Dict: scenario_name -> float32 array (N,)
    """
    results: Dict[str, np.ndarray] = {}
    for scenario, seqs in scenario_sequences.items():
        print(f"[compute] energy for {scenario:<24s} ({len(seqs):5d} seqs) ...", end=" ")
        e = compute_energies(model, seqs, device=device, batch_size=batch_size)
        results[scenario] = e
        print(f"mean={e.mean():.3f}  std={e.std():.3f}  "
              f"min={e.min():.3f}  max={e.max():.3f}")
    return results


# ── 2-D energy grid ────────────────────────────────────────────────────────

@torch.no_grad()
def compute_energy_grid(
    model: nn.Module,
    probes: np.ndarray,
    grid_res: int,
    device: str = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    """
    Compute the energy for a (G², seq_len, D) probe batch and reshape to (G, G).

    Parameters
    ----------
    probes   : float32 array (G², seq_len, D) — from loader.build_grid_probe
    grid_res : int — G, so output shape is (grid_res, grid_res)

    Returns
    -------
    E_grid : float32 array (grid_res, grid_res)
    """
    energies = compute_energies(model, probes, device=device, batch_size=batch_size)
    return energies.reshape(grid_res, grid_res)


# ── 1-D energy profile ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_energy_profile(
    model: nn.Module,
    x_mean: np.ndarray,
    x_std:  np.ndarray,
    dim_i:  int,
    seq_len: int,
    n_sigma: float = 2.5,
    n_points: int = 100,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sweep feature `dim_i` from -n_sigma to +n_sigma (normalised units)
    while holding all other features at x_mean.

    Returns
    -------
    xi_vals  : (n_points,) — feature values in normalised space
    energies : (n_points,) — corresponding energy
    """
    xi_vals = np.linspace(
        x_mean[dim_i] - n_sigma * x_std[dim_i],
        x_mean[dim_i] + n_sigma * x_std[dim_i],
        n_points,
    )
    D = len(x_mean)
    # Each probe: one sequence of seq_len identical time steps
    probes = np.tile(x_mean, (n_points, seq_len, 1)).astype(np.float32)
    probes[:, :, dim_i] = xi_vals[:, None]

    energies = compute_energies(model, probes, device=device, batch_size=n_points)
    return xi_vals, energies


def compute_all_1d_profiles(
    model: nn.Module,
    x_mean: np.ndarray,
    x_std:  np.ndarray,
    seq_len: int,
    input_labels: list,
    n_sigma: float = 2.5,
    n_points: int = 100,
    device: str = "cpu",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute 1-D energy profiles for every input feature.

    Returns
    -------
    Dict: feature_name -> (xi_vals, energies)
    """
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for i, name in enumerate(input_labels):
        xi, e = compute_energy_profile(
            model, x_mean, x_std, i, seq_len,
            n_sigma=n_sigma, n_points=n_points, device=device,
        )
        profiles[name] = (xi, e)
        grad_approx = np.abs(np.gradient(e)).mean()
        print(f"[compute] 1D profile {name:<20s}  |∂E/∂x| ≈ {grad_approx:.4f}")
    return profiles


# ── energy sensitivity ranking ─────────────────────────────────────────────

def compute_sensitivity_ranking(
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    """
    Rank features by the mean absolute gradient of their 1-D energy profile.

    A higher value means the energy is more sensitive to that feature —
    i.e., the EBM treats it as more structurally important.

    Returns
    -------
    Dict: feature_name -> mean |∂E/∂x| (descending order)
    """
    ranking: Dict[str, float] = {}
    for name, (xi, e) in profiles.items():
        ranking[name] = float(np.abs(np.gradient(e)).mean())
    return dict(sorted(ranking.items(), key=lambda kv: kv[1], reverse=True))


# ── summary statistics ─────────────────────────────────────────────────────

def compute_energy_stats(
    scenario_energies: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean, std, median, min, max per scenario.
    """
    stats: Dict[str, Dict[str, float]] = {}
    for scenario, e in scenario_energies.items():
        stats[scenario] = {
            "mean":   float(e.mean()),
            "std":    float(e.std()),
            "median": float(np.median(e)),
            "min":    float(e.min()),
            "max":    float(e.max()),
            "n":      int(len(e)),
        }
    return stats


def compute_separation_margin(
    scenario_energies: Dict[str, np.ndarray],
    reference: str = "real",
) -> Dict[str, float]:
    """
    For each edge-case scenario, compute the mean energy gap vs the real data.

      margin = mean(E_scenario) - mean(E_real)

    Positive margin means the model assigns higher energy (less safe) to
    that scenario than to normal operation — the EBM is working correctly.

    Returns
    -------
    Dict: scenario -> margin value
    """
    e_ref = scenario_energies[reference].mean()
    margins: Dict[str, float] = {}
    for scenario, e in scenario_energies.items():
        if scenario == reference:
            continue
        margins[scenario] = float(e.mean() - e_ref)
    return dict(sorted(margins.items(), key=lambda kv: kv[1], reverse=True))
