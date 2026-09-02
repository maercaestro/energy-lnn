"""
plots.py
========
All plotting functions for the H1 energy landscape analysis.

Five figures:
  fig1  — Energy distribution by scenario (violin + strip)
  fig2  — Energy CDF per scenario
  fig3  — 2-D energy contour maps (pairwise feature sweeps)
  fig4  — 1-D energy sensitivity profiles
  fig5  — Energy vs safety boundary (scatter)

Each function returns a matplotlib Figure so the caller can either save it
locally or log it to W&B with wandb.Image(fig).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── colour palette ─────────────────────────────────────────────────────────
# Ordered to match SCENARIO_ORDER: normal (green), then 6 danger levels
PALETTE = [
    "#2ecc71",   # real      — green (safe)
    "#e74c3c",   # S1        — red
    "#e67e22",   # S2        — orange
    "#f39c12",   # S3        — amber
    "#9b59b6",   # S4        — purple
    "#3498db",   # S5        — blue
    "#1abc9c",   # S6        — teal
]

SCENARIO_ORDER = [
    "real",
    "S1_flame_out",
    "S2_air_leak",
    "S3_tube_rupture",
    "S4_positive_pressure",
    "S5_fuel_trip",
    "S6_fuel_contamination",
]

SCENARIO_LABELS = {
    "real":                  "Normal",
    "S1_flame_out":          "Flame-Out",
    "S2_air_leak":           "Air Leak",
    "S3_tube_rupture":       "Tube Rupture",
    "S4_positive_pressure":  "Pos. Pressure",
    "S5_fuel_trip":          "Fuel Trip",
    "S6_fuel_contamination": "Fuel Contam.",
}

INPUT_LABELS = ["fuel_flow", "AFR", "cur_temp", "inflow_T", "inflow_rate"]


# ── fig1: violin + strip ───────────────────────────────────────────────────

def plot_energy_distributions(
    scenario_energies: Dict[str, np.ndarray],
    title: str = "Learned Energy Distribution by Scenario",
) -> plt.Figure:
    """
    Violin plot of energy values per scenario, ordered by SCENARIO_ORDER.

    The key expected result: real (normal) data has the lowest energy;
    dangerous edge-case scenarios have progressively higher energy.
    This is visual evidence for H1.
    """
    present = [s for s in SCENARIO_ORDER if s in scenario_energies]
    labels  = [SCENARIO_LABELS.get(s, s) for s in present]
    colors  = [PALETTE[SCENARIO_ORDER.index(s)] for s in present]
    data    = [scenario_energies[s] for s in present]

    fig, ax = plt.subplots(figsize=(12, 6))

    parts = ax.violinplot(data, positions=range(len(present)),
                          showmedians=True, showextrema=True)

    for i, (pc, col) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    for part_name in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[part_name].set_color("0.3")
        parts[part_name].set_linewidth(1.5)

    # overlay individual points (subsample for speed)
    for i, (d, col) in enumerate(zip(data, colors)):
        sub = d if len(d) <= 300 else np.random.choice(d, 300, replace=False)
        jitter = np.random.uniform(-0.12, 0.12, size=len(sub))
        ax.scatter(i + jitter, sub, s=6, alpha=0.35, color=col, zorder=3)

    # annotate mean
    for i, d in enumerate(data):
        ax.text(i, d.mean(), f"μ={d.mean():.2f}", ha="center",
                va="bottom", fontsize=8, color="0.2")

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Energy  $E_\\theta(x)$", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axhline(scenario_energies.get("real", np.array([0])).mean(),
               color="#2ecc71", linestyle="--", linewidth=1.2, alpha=0.8,
               label="Normal mean")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ── fig2: CDF ──────────────────────────────────────────────────────────────

def plot_energy_cdf(
    scenario_energies: Dict[str, np.ndarray],
    title: str = "Cumulative Energy Distribution by Scenario",
) -> plt.Figure:
    """
    Overlaid CDFs per scenario.

    A large horizontal shift between normal and edge-case CDFs indicates
    a clear energy separation — the EBM has learned to distinguish them.
    """
    present = [s for s in SCENARIO_ORDER if s in scenario_energies]

    fig, ax = plt.subplots(figsize=(10, 5))

    for s in present:
        e = np.sort(scenario_energies[s])
        cdf = np.arange(1, len(e) + 1) / len(e)
        col = PALETTE[SCENARIO_ORDER.index(s)]
        lw  = 2.5 if s == "real" else 1.5
        ls  = "-"  if s == "real" else "--"
        ax.plot(e, cdf, color=col, linewidth=lw, linestyle=ls,
                label=SCENARIO_LABELS.get(s, s))

    ax.set_xlabel("Energy  $E_\\theta(x)$", fontsize=12)
    ax.set_ylabel("Cumulative probability", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


# ── fig3: 2-D contour grid ─────────────────────────────────────────────────

def plot_2d_contours(
    grids: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    scenario_sequences: Dict[str, np.ndarray],
    input_labels: List[str] = INPUT_LABELS,
    title: str = "2-D Energy Contour Maps",
    max_scatter: int = 200,
) -> plt.Figure:
    """
    One subplot per (dim_i, dim_j) pair in `grids`.

    Parameters
    ----------
    grids : dict (dim_i, dim_j) -> (xi_vals, xj_vals, E_grid)
    scenario_sequences : dict scenario -> (N, T, D) normalised sequences
                         Used to scatter-plot real and edge-case positions.
    """
    n_pairs = len(grids)
    ncols   = min(n_pairs, 3)
    nrows   = (n_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             squeeze=False)

    for ax_idx, ((di, dj), (xi, xj, Egrid)) in enumerate(grids.items()):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]

        xi_label = input_labels[di] if di < len(input_labels) else f"dim{di}"
        xj_label = input_labels[dj] if dj < len(input_labels) else f"dim{dj}"

        cf = ax.contourf(xi, xj, Egrid, levels=30, cmap="RdYlGn_r", alpha=0.85)
        plt.colorbar(cf, ax=ax, shrink=0.85, label="Energy")
        ax.contour(xi, xj, Egrid, levels=10, colors="k", linewidths=0.4, alpha=0.4)

        # scatter real and edge-case sequence means
        for s in SCENARIO_ORDER:
            if s not in scenario_sequences:
                continue
            seqs = scenario_sequences[s]          # (N, T, D)
            pts  = seqs[:, 0, :]                  # use first timestep as point
            if len(pts) > max_scatter:
                idx = np.random.choice(len(pts), max_scatter, replace=False)
                pts = pts[idx]
            col_s  = PALETTE[SCENARIO_ORDER.index(s)]
            marker = "o" if s == "real" else "x"
            size   = 14 if s == "real" else 18
            ax.scatter(pts[:, di], pts[:, dj],
                       s=size, c=col_s, marker=marker, alpha=0.55,
                       label=SCENARIO_LABELS.get(s, s), edgecolors="none",
                       linewidths=0.8)

        ax.set_xlabel(xi_label, fontsize=10)
        ax.set_ylabel(xj_label, fontsize=10)
        ax.set_title(f"E({xi_label}, {xj_label})", fontsize=11)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="upper right", markerscale=1.5)

    # hide unused axes
    for ax_idx in range(n_pairs, nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── fig4: 1-D profiles ─────────────────────────────────────────────────────

def plot_1d_profiles(
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray]],
    sensitivity_ranking: Dict[str, float],
    title: str = "1-D Energy Sensitivity Profiles",
) -> plt.Figure:
    """
    One subplot per feature showing how energy changes as that feature varies.
    Features ordered by sensitivity (most sensitive first).

    A steep profile means the EBM assigns a strong energy penalty to
    deviations in that feature — a signature of structural importance.
    """
    ordered_names = list(sensitivity_ranking.keys())   # desc order
    n = len(ordered_names)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for idx, name in enumerate(ordered_names):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        xi, e = profiles[name]
        rank_val = sensitivity_ranking[name]

        ax.plot(xi, e, color="#2c3e50", linewidth=2)
        ax.fill_between(xi, e, alpha=0.15, color="#2c3e50")

        # Mark minimum (safe operating point)
        min_idx = np.argmin(e)
        ax.axvline(xi[min_idx], color="#2ecc71", linestyle="--",
                   linewidth=1.3, label=f"min E @ {xi[min_idx]:.2f}")
        ax.scatter([xi[min_idx]], [e[min_idx]], s=60, color="#2ecc71", zorder=5)

        ax.set_xlabel(f"{name}  (normalised)", fontsize=9)
        ax.set_ylabel("Energy $E_\\theta$", fontsize=9)
        ax.set_title(f"{name}\n|∂E/∂x| ≈ {rank_val:.4f}",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── fig5: energy vs safety ─────────────────────────────────────────────────

def plot_energy_vs_safety(
    scenario_energies: Dict[str, np.ndarray],
    scenario_sequences: Dict[str, np.ndarray],
    target_scaler,
    title: str = "Energy vs Predicted Safety Violation",
    o2_low_crit:   float = 1.5,
    temp_high_crit: float = 500.0,
) -> plt.Figure:
    """
    Scatter: learned energy (x-axis) vs the raw (unscaled) current_temp
    and inflow_rate, coloured by scenario, with safety thresholds annotated.

    We use current_temp (input dim 2) as a proxy for the physical state.
    This shows whether high-energy sequences correspond to physically unsafe
    input states.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for s in SCENARIO_ORDER:
        if s not in scenario_energies or s not in scenario_sequences:
            continue
        e    = scenario_energies[s]
        seqs = scenario_sequences[s]   # (N, T, D) normalised
        col  = PALETTE[SCENARIO_ORDER.index(s)]

        # use first timestep, feature dim 2 = current_temp (normalised)
        cur_temp_norm = seqs[:, 0, 2]
        afr_norm      = seqs[:, 0, 1]

        sub_idx = slice(None) if len(e) <= 500 else \
                  np.random.choice(len(e), 500, replace=False)

        axes[0].scatter(e[sub_idx], cur_temp_norm[sub_idx],
                        s=12, alpha=0.45, color=col,
                        label=SCENARIO_LABELS.get(s, s))
        axes[1].scatter(e[sub_idx], afr_norm[sub_idx],
                        s=12, alpha=0.45, color=col,
                        label=SCENARIO_LABELS.get(s, s))

    for ax, ylabel, feat in zip(
        axes,
        ["Current Temp (normalised)", "Air-Fuel Ratio (normalised)"],
        ["cur_temp", "AFR"],
    ):
        ax.set_xlabel("Energy  $E_\\theta(x)$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Energy vs {feat}", fontsize=11)
        ax.legend(fontsize=7, loc="upper left", markerscale=1.8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ── fig6: separation summary bar ───────────────────────────────────────────

def plot_separation_margins(
    margins: Dict[str, float],
    title: str = "Energy Separation Margin (vs Normal)",
) -> plt.Figure:
    """
    Horizontal bar chart: mean(E_scenario) - mean(E_real) for each scenario.

    Positive bars = model assigns higher energy to that scenario than to
    normal operation. This is the most concise H1 evidence figure.
    """
    scenarios = list(margins.keys())
    values    = [margins[s] for s in scenarios]
    colors    = [PALETTE[SCENARIO_ORDER.index(s)] if s in SCENARIO_ORDER
                 else "#95a5a6" for s in scenarios]
    labels    = [SCENARIO_LABELS.get(s, s) for s in scenarios]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, values, color=colors, edgecolor="0.3", linewidth=0.6)
    ax.axvline(0, color="0.3", linewidth=1.0)

    for bar, v in zip(bars, values):
        xpos = v + (0.01 if v >= 0 else -0.01)
        ha   = "left" if v >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{v:+.3f}", va="center", ha=ha, fontsize=9)

    ax.set_xlabel("ΔE  (scenario mean − normal mean)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
