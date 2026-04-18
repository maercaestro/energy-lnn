"""
Publication-quality plots for the lnn-test benchmark.

Generates seven figures:
  1. Grouped bar chart — clean RMSE per model (temp and O₂).
  2. Line chart — disturbance degradation ratio per noise level.
  3. Grouped bar chart — safety violation rates (clean vs noise).
  4. Horizontal bar — composite ranking score.
  5. Grouped bar — parameter count vs accuracy (efficiency).
  6. Grouped bar — per-disturbance category breakdown.
  7. Training convergence curves (loss vs epoch).

Usage:
    python lnn-test/scripts/plot_results.py
    python lnn-test/scripts/plot_results.py --results_dir lnn-test/results/default
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"lnn": "#2176AE", "lstm": "#F26419", "mlp": "#86BA90"}
MODEL_LABELS = {"lnn": "LNN (CfC)", "lstm": "LSTM", "mlp": "MLP"}
NOISE_LEVELS = ["0.1", "0.25", "0.5", "1.0", "2.0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot lnn-test benchmark results")
    parser.add_argument("--results_dir", type=str, default=None)
    return parser.parse_args()


def load_runs(results_dir: Path) -> Dict[str, List[dict]]:
    runs: Dict[str, List[dict]] = {}
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        model_runs = []
        for seed_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            rpath = seed_dir / "results.json"
            if rpath.exists():
                with open(rpath) as f:
                    model_runs.append(json.load(f))
        if model_runs:
            runs[model_dir.name] = model_runs
    return runs


def _mean_metric(runs: List[dict], *keys: str) -> float:
    vals = []
    for run in runs:
        obj = run
        for k in keys:
            obj = obj[k]
        vals.append(float(obj))
    return mean(vals)


def _setup_figure():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })


# ─────────────────────────────────────────────────────────────────────
# Figure 1: Clean RMSE bar chart
# ─────────────────────────────────────────────────────────────────────

def plot_clean_rmse(runs: Dict[str, List[dict]], out_dir: Path) -> None:
    models = sorted(runs.keys())
    temp_vals = [_mean_metric(runs[m], "test_metrics", "rmse_temp") for m in models]
    o2_vals = [_mean_metric(runs[m], "test_metrics", "rmse_o2") for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, temp_vals, width, label="Temp RMSE (°C)", color=[COLORS.get(m, "#999") for m in models], edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, o2_vals, width, label="O₂ RMSE (%)", color=[COLORS.get(m, "#999") for m in models], alpha=0.6, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m.upper()) for m in models])
    ax.set_ylabel("RMSE")
    ax.set_title("Clean Test-Set RMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_clean_rmse.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 2: Noise degradation across sigma levels
# ─────────────────────────────────────────────────────────────────────

def plot_noise_degradation(runs: Dict[str, List[dict]], out_dir: Path) -> None:
    models = sorted(runs.keys())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for col_idx, (target, label) in enumerate([("rmse_temp", "Temp RMSE Degradation"), ("rmse_o2", "O₂ RMSE Degradation")]):
        ax = axes[col_idx]
        for model in models:
            deg_vals = []
            for sigma in NOISE_LEVELS:
                key = f"noise_sigma_{sigma}"
                vals = []
                for run in runs[model]:
                    vals.append(run["disturbance"][key]["degradation"][target])
                deg_vals.append(mean(vals))
            ax.plot(NOISE_LEVELS, deg_vals, marker="o", label=MODEL_LABELS.get(model, model.upper()), color=COLORS.get(model, "#999"), linewidth=2)

        ax.set_xlabel("Noise σ")
        ax.set_ylabel("Degradation (×)")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_noise_degradation.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 3: Safety violations (clean vs noise mean)
# ─────────────────────────────────────────────────────────────────────

def plot_safety_violations(runs: Dict[str, List[dict]], out_dir: Path) -> None:
    models = sorted(runs.keys())

    clean_crit = [_mean_metric(runs[m], "safety_summary", "clean_critical_rate") * 100 for m in models]
    noise_crit = [_mean_metric(runs[m], "safety_summary", "noise_critical_rate") * 100 for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, clean_crit, width, label="Clean", color=[COLORS.get(m, "#999") for m in models], edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, noise_crit, width, label="Mean under noise", color=[COLORS.get(m, "#999") for m in models], alpha=0.55, edgecolor="black", linewidth=0.5, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m.upper()) for m in models])
    ax.set_ylabel("Critical Violations (%)")
    ax.set_title("Safety-Boundary Violations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_safety_violations.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 4: Composite ranking bar
# ─────────────────────────────────────────────────────────────────────

def plot_composite_ranking(results_dir: Path, out_dir: Path) -> None:
    summary_path = results_dir / "comparison_summary.json"
    if not summary_path.exists():
        return
    with open(summary_path) as f:
        data = json.load(f)

    scores = data.get("composite_scores", {})
    if not scores:
        return

    ranked = sorted(scores.items(), key=lambda item: item[1]["rank"])
    models = [m for m, _ in ranked]
    vals = [info["composite_score"] for _, info in ranked]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(
        [MODEL_LABELS.get(m, m.upper()) for m in models],
        vals,
        color=[COLORS.get(m, "#999") for m in models],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Composite Score (lower = better)")
    ax.set_title("Overall Model Ranking")
    ax.invert_yaxis()

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_composite_ranking.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 5: Parameter efficiency — twin-axis bar (param count + RMSE)
# ─────────────────────────────────────────────────────────────────────

def plot_parameter_efficiency(runs: Dict[str, List[dict]], out_dir: Path) -> None:
    models = sorted(runs.keys())

    param_counts = []
    temp_rmse = []
    for m in models:
        pcs = [r.get("param_count", 0) for r in runs[m]]
        param_counts.append(mean(pcs))
        temp_rmse.append(_mean_metric(runs[m], "test_metrics", "rmse_temp"))

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(7, 4))
    bars = ax1.bar(x - width / 2, [p / 1000 for p in param_counts], width,
                   color=[COLORS.get(m, "#999") for m in models], edgecolor="black",
                   linewidth=0.5, label="Parameters (K)")
    ax1.set_ylabel("Parameters (×1000)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_LABELS.get(m, m.upper()) for m in models])

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, temp_rmse, width,
            color=[COLORS.get(m, "#999") for m in models], alpha=0.5,
            edgecolor="black", linewidth=0.5, hatch="//", label="Temp RMSE (°C)")
    ax2.set_ylabel("Temp RMSE (°C)")

    # Annotate efficiency
    for i, m in enumerate(models):
        eff = temp_rmse[i] / (param_counts[i] / 10000) if param_counts[i] > 0 else 0
        ax1.text(x[i], max(p / 1000 for p in param_counts) * 1.05, f"{eff:.4f}\nRMSE/10K",
                 ha="center", fontsize=8, style="italic")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Parameter Efficiency")
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_parameter_efficiency.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 6: Per-disturbance category breakdown (grouped bar)
# ─────────────────────────────────────────────────────────────────────

CATEGORIES = {
    "Noise": lambda k: k.startswith("noise_sigma_"),
    "Dropout": lambda k: k.startswith("dropout_"),
    "Spike": lambda k: k.startswith("spike_"),
    "Shuffle": lambda k: k == "temporal_shuffle",
    "Extrap": lambda k: k.startswith("extrap_"),
    "Scale": lambda k: k.startswith("scale_"),
}


def plot_disturbance_categories(runs: Dict[str, List[dict]], out_dir: Path) -> None:
    models = sorted(runs.keys())
    cat_names = list(CATEGORIES.keys())

    data = {m: [] for m in models}
    for m in models:
        for cat_name, cat_filter in CATEGORIES.items():
            cat_vals = []
            for run in runs[m]:
                dist = run["disturbance"]
                keys = [k for k in dist if k != "clean" and cat_filter(k)]
                if keys:
                    cat_vals.append(mean(dist[k]["degradation"]["rmse_temp"] for k in keys))
            data[m].append(mean(cat_vals) if cat_vals else 0.0)

    x = np.arange(len(cat_names))
    n_models = len(models)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(models):
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, data[m], width, label=MODEL_LABELS.get(m, m.upper()),
               color=COLORS.get(m, "#999"), edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_names)
    ax.set_ylabel("Mean Temp RMSE Degradation (×)")
    ax.set_title("Robustness by Perturbation Category")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_disturbance_categories.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 7: Training convergence curves
# ─────────────────────────────────────────────────────────────────────

def plot_training_convergence(results_dir: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        model_name = model_dir.name
        # Use the first seed for cleaner plot
        seed_dirs = sorted(p for p in model_dir.iterdir() if p.is_dir())
        if not seed_dirs:
            continue
        history_path = seed_dirs[0] / "history.npz"
        if not history_path.exists():
            continue
        history = np.load(history_path)
        val_loss = history["val_loss"]
        ax.plot(range(1, len(val_loss) + 1), val_loss,
                label=MODEL_LABELS.get(model_name, model_name.upper()),
                color=COLORS.get(model_name, "#999"), linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (MSE)")
    ax.set_title("Training Convergence (seed 42)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_dir / "fig7_training_convergence.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    _setup_figure()

    root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir) if args.results_dir else (root / "results" / "default")
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(results_dir)
    if not runs:
        raise ValueError(f"No saved runs found under {results_dir}")

    plot_clean_rmse(runs, plot_dir)
    print(f"  Saved fig1_clean_rmse.png")

    plot_noise_degradation(runs, plot_dir)
    print(f"  Saved fig2_noise_degradation.png")

    plot_safety_violations(runs, plot_dir)
    print(f"  Saved fig3_safety_violations.png")

    plot_composite_ranking(results_dir, plot_dir)
    print(f"  Saved fig4_composite_ranking.png")

    plot_parameter_efficiency(runs, plot_dir)
    print(f"  Saved fig5_parameter_efficiency.png")

    plot_disturbance_categories(runs, plot_dir)
    print(f"  Saved fig6_disturbance_categories.png")

    plot_training_convergence(results_dir, plot_dir)
    print(f"  Saved fig7_training_convergence.png")

    print(f"\nAll plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
