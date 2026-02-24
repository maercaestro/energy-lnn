"""
Energy Landscape Visualisation
================================
Plots the learned energy function E_θ(x) from the trained EBM head.

Two view types
--------------
1. 2-D contour grids  — sweep two input dimensions on a grid, hold the
   remaining three at their mean value.  Produces a 5×5 panel of
   pairwise landscapes (upper triangle only).

2. 1-D energy profiles — sweep each input independently ±3σ while
   holding others at their mean.

Usage
-----
    cd energy-lnn/eblnn
    python experiments/energy_landscape.py
    python experiments/energy_landscape.py \\
        --model_path results/models/best_model.pth \\
        --data_path  data/synthetic_temperature_data.csv \\
        --grid_res   80 \\
        --out_dir    results/plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import create_model
from src.data import DataPipeline


INPUT_LABELS = ["fuel_flow", "AFR", "cur_temp", "inflow_T", "inflow_rate"]
D = len(INPUT_LABELS)


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_energy_grid(
    model,
    x_mean: np.ndarray,   # (D,)  mean values in normalised space
    x_std: np.ndarray,    # (D,)
    dim_i: int,
    dim_j: int,
    grid_res: int,
    seq_len: int,
    n_sigma: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep dim_i and dim_j on a grid; hold others at x_mean.

    Returns
    -------
    xi_vals : (grid_res,)
    xj_vals : (grid_res,)
    E_grid  : (grid_res, grid_res)
    """
    xi_vals = np.linspace(
        x_mean[dim_i] - n_sigma * x_std[dim_i],
        x_mean[dim_i] + n_sigma * x_std[dim_i],
        grid_res,
    )
    xj_vals = np.linspace(
        x_mean[dim_j] - n_sigma * x_std[dim_j],
        x_mean[dim_j] + n_sigma * x_std[dim_j],
        grid_res,
    )

    XI, XJ = np.meshgrid(xi_vals, xj_vals)   # (G, G)
    G = grid_res * grid_res

    # Build flat batch: (G, seq_len, D)
    # Every timestep in the sequence is the same point (steady-state probe)
    x_probe = np.tile(x_mean, (G, seq_len, 1)).astype(np.float32)
    x_probe[:, :, dim_i] = np.repeat(XI.ravel(), seq_len).reshape(G, seq_len)
    x_probe[:, :, dim_j] = np.repeat(XJ.ravel(), seq_len).reshape(G, seq_len)

    x_t = torch.from_numpy(x_probe)          # (G, T, D)
    energies = model.energy(x_t).numpy()     # (G,)

    E_grid = energies.reshape(grid_res, grid_res)
    return xi_vals, xj_vals, E_grid


@torch.no_grad()
def compute_energy_1d(
    model,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    dim: int,
    grid_res: int,
    seq_len: int,
    n_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep a single dimension; hold others at mean."""
    x_vals = np.linspace(
        x_mean[dim] - n_sigma * x_std[dim],
        x_mean[dim] + n_sigma * x_std[dim],
        grid_res,
    )
    x_probe = np.tile(x_mean, (grid_res, seq_len, 1)).astype(np.float32)
    x_probe[:, :, dim] = x_vals[:, None]

    x_t = torch.from_numpy(x_probe)
    energies = model.energy(x_t).numpy()
    return x_vals, energies


# ──────────────────────────────────────────────────────────────────────────
# plot helpers
# ──────────────────────────────────────────────────────────────────────────

def plot_2d_landscapes(
    model,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    grid_res: int,
    seq_len: int,
    out_path: Path,
) -> None:
    """Upper-triangle pairwise 2-D energy contour plots."""
    n_pairs = D * (D - 1) // 2
    fig, axes = plt.subplots(D - 1, D - 1, figsize=(3.5 * (D - 1), 3.0 * (D - 1)))

    # Global energy range for consistent colour scale
    E_min, E_max = np.inf, -np.inf
    cache: dict[tuple, tuple] = {}

    for i in range(D):
        for j in range(i + 1, D):
            xi_vals, xj_vals, E_grid = compute_energy_grid(
                model, x_mean, x_std, i, j, grid_res, seq_len
            )
            cache[(i, j)] = (xi_vals, xj_vals, E_grid)
            E_min = min(E_min, E_grid.min())
            E_max = max(E_max, E_grid.max())

    # Draw
    for i in range(D):
        for j in range(i + 1, D):
            row = i
            col = j - 1
            ax = axes[row][col] if D > 2 else axes

            xi_vals, xj_vals, E_grid = cache[(i, j)]
            XI, XJ = np.meshgrid(xi_vals, xj_vals)

            cf = ax.contourf(XI, XJ, E_grid, levels=20,
                             cmap="RdYlBu_r", vmin=E_min, vmax=E_max)
            ax.contour(XI, XJ, E_grid, levels=10,
                       colors="white", linewidths=0.4, alpha=0.5)

            # Mark minimum energy point
            ij_min = np.unravel_index(E_grid.argmin(), E_grid.shape)
            ax.plot(XI[ij_min], XJ[ij_min], "w*", markersize=10, label="E min")

            ax.set_xlabel(INPUT_LABELS[j], fontsize=8)
            ax.set_ylabel(INPUT_LABELS[i], fontsize=8)
            ax.set_title(f"{INPUT_LABELS[i]} vs {INPUT_LABELS[j]}", fontsize=8)
            ax.tick_params(labelsize=7)

    # Hide unused axes (lower triangle)
    for i in range(D - 1):
        for j in range(D - 1):
            if j < i:
                axes[i][j].set_visible(False)

    # Shared colourbar
    fig.colorbar(
        plt.cm.ScalarMappable(
            cmap="RdYlBu_r",
            norm=plt.Normalize(E_min, E_max),
        ),
        ax=axes, shrink=0.6, label="Energy E(x)",
    )
    fig.suptitle(
        "Pairwise Energy Landscape  (others held at mean)\n"
        "Blue = low energy (preferred region) · Red = high energy (rejected region)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"2-D landscapes saved → {out_path}")
    plt.close(fig)


def plot_1d_profiles(
    model,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    grid_res: int,
    seq_len: int,
    out_path: Path,
) -> None:
    """1-D energy profiles: one subplot per input feature."""
    fig, axes = plt.subplots(1, D, figsize=(3.5 * D, 3.5), sharey=False)

    colors = plt.cm.tab10(np.linspace(0, 1, D))

    for dim in range(D):
        x_vals, energies = compute_energy_1d(
            model, x_mean, x_std, dim, grid_res, seq_len
        )
        ax = axes[dim]
        ax.plot(x_vals, energies, color=colors[dim], linewidth=2)
        ax.fill_between(x_vals, energies, energies.min(),
                        color=colors[dim], alpha=0.15)

        # Minimum marker
        idx_min = energies.argmin()
        ax.axvline(x_vals[idx_min], color="black", linestyle="--",
                   linewidth=0.8, alpha=0.6)
        ax.scatter([x_vals[idx_min]], [energies[idx_min]],
                   color="black", zorder=5, s=50, label=f"min={x_vals[idx_min]:.2f}")

        # Mark ±1σ from mean
        ax.axvline(x_mean[dim] - x_std[dim], color="grey",
                   linestyle=":", linewidth=0.8, alpha=0.7)
        ax.axvline(x_mean[dim] + x_std[dim], color="grey",
                   linestyle=":", linewidth=0.8, alpha=0.7, label="±1σ")

        ax.set_xlabel(INPUT_LABELS[dim], fontsize=9)
        ax.set_ylabel("E(x)" if dim == 0 else "", fontsize=9)
        ax.set_title(f"{INPUT_LABELS[dim]}", fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "1-D Energy Profiles  (others held at mean)\n"
        "Dashes = energy minimum · Dotted = ±1σ bounds",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"1-D profiles saved → {out_path}")
    plt.close(fig)


def plot_energy_distribution(
    model,
    x_real: torch.Tensor,
    out_path: Path,
) -> None:
    """
    Histogram of E(x_real) over the test set.
    A good EBM should have a tight low-energy mode for real data.
    """
    with torch.no_grad():
        energies = model.energy(x_real).numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(energies, bins=50, color="steelblue", edgecolor="white",
            alpha=0.85, density=True)
    ax.axvline(energies.mean(), color="tomato", linewidth=1.5,
               linestyle="--", label=f"mean={energies.mean():.3f}")
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":",
               label="E=0 reference")
    ax.set_xlabel("Energy E(x)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        f"Energy Distribution  (real test data, n={len(energies)})\n"
        f"μ={energies.mean():.3f}  σ={energies.std():.3f}  "
        f"[{energies.min():.2f}, {energies.max():.2f}]",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Energy distribution saved → {out_path}")
    plt.close(fig)


def plot_energy_surface_3d(
    model,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    dim_i: int,
    dim_j: int,
    grid_res: int,
    seq_len: int,
    out_path: Path,
) -> None:
    """3-D surface plot for the most important pair (highest variance)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    xi_vals, xj_vals, E_grid = compute_energy_grid(
        model, x_mean, x_std, dim_i, dim_j, grid_res, seq_len
    )
    XI, XJ = np.meshgrid(xi_vals, xj_vals)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        XI, XJ, E_grid,
        cmap="RdYlBu_r", linewidth=0, antialiased=True, alpha=0.9,
    )
    ax.set_xlabel(INPUT_LABELS[dim_j], fontsize=9, labelpad=8)
    ax.set_ylabel(INPUT_LABELS[dim_i], fontsize=9, labelpad=8)
    ax.set_zlabel("E(x)", fontsize=9)
    ax.set_title(
        f"3-D Energy Surface\n{INPUT_LABELS[dim_i]} × {INPUT_LABELS[dim_j]}",
        fontsize=10,
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, label="Energy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"3-D surface saved → {out_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  default=str(ROOT / "results/models/best_model.pth"))
    p.add_argument("--data_path",   default=str(ROOT / "data/synthetic_temperature_data.csv"))
    p.add_argument("--grid_res",    type=int,   default=60)
    p.add_argument("--n_samples",   type=int,   default=500)
    p.add_argument("--hidden_size", type=int,   default=128)
    p.add_argument("--ebm_hidden_dims", default="[128,64]")
    p.add_argument("--out_dir",     default=str(ROOT / "results/plots"))
    return p.parse_args()


def main() -> None:
    import ast
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ─────────────────────────────────────────────────────
    print(f"Loading model from  {args.model_path}")
    ebm_dims = ast.literal_eval(args.ebm_hidden_dims)
    model = create_model(
        input_size=5,
        hidden_size=args.hidden_size,
        phys_output_size=2,
        ebm_hidden_dims=ebm_dims,
        device="cpu",
    )
    state = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.")

    # ── 2. Load data ──────────────────────────────────────────────────────
    print(f"Loading data from   {args.data_path}")
    pipeline = DataPipeline(
        data_path=args.data_path,
        num_sequences=max(1000, args.n_samples * 2),
        force_regenerate=False,
    ).build()

    x_batches = []
    for xb, _ in pipeline.test_loader:
        x_batches.append(xb)
        if sum(b.shape[0] for b in x_batches) >= args.n_samples:
            break
    x_real = torch.cat(x_batches, dim=0)[: args.n_samples]   # (N, T, 5)
    print(f"Test samples: {x_real.shape}")

    # ── 3. Compute statistics in normalised space ─────────────────────────
    x_flat = x_real.reshape(-1, D).numpy()          # (N*T, 5)
    x_mean = x_flat.mean(axis=0)
    x_std  = x_flat.std(axis=0).clip(min=1e-4)

    seq_len = x_real.shape[1]
    print(f"x_mean: {np.round(x_mean, 3)}")
    print(f"x_std : {np.round(x_std,  3)}")

    # ── 4. Quick sanity: print energy stats ───────────────────────────────
    with torch.no_grad():
        e_real = model.energy(x_real).numpy()
    print(f"\nEnergy on real test data:  μ={e_real.mean():.4f}  σ={e_real.std():.4f}"
          f"  range=[{e_real.min():.3f}, {e_real.max():.3f}]")

    # ── 5. Plots ──────────────────────────────────────────────────────────
    print("\nGenerating plots …")

    plot_energy_distribution(
        model, x_real,
        out_dir / "energy_distribution.png",
    )

    plot_1d_profiles(
        model, x_mean, x_std, args.grid_res, seq_len,
        out_dir / "energy_1d_profiles.png",
    )

    plot_2d_landscapes(
        model, x_mean, x_std, args.grid_res, seq_len,
        out_dir / "energy_2d_landscapes.png",
    )

    # 3-D surface: choose pair with largest Jacobian variance as "most interesting"
    x_t = x_real.detach().clone().requires_grad_(True)
    model.energy(x_t).sum().backward()
    jac_mean = x_t.grad.detach().abs().mean(dim=(0, 1)).numpy()  # (D,)
    top2 = np.argsort(jac_mean)[::-1][:2]
    di, dj = int(top2[0]), int(top2[1])
    print(f"Top-2 energy-sensitive features: {INPUT_LABELS[di]}, {INPUT_LABELS[dj]}")

    plot_energy_surface_3d(
        model, x_mean, x_std, di, dj, args.grid_res, seq_len,
        out_dir / f"energy_surface_3d_{INPUT_LABELS[di]}_{INPUT_LABELS[dj]}.png",
    )

    print("\nAll plots saved to", out_dir)


if __name__ == "__main__":
    main()
