"""
analyze_ablations.py
====================
Collect all ablation results, build comparison tables and plots.

Usage
-----
    # Analyse results (from eblnn/ directory)
    python experiments/analyze_ablations.py

    # Custom results directory
    python experiments/analyze_ablations.py --results_dir results/ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "ablation"


# =====================================================================
# Helpers
# =====================================================================

def collect_results(results_dir: Path) -> pd.DataFrame:
    """Walk all experiment sub-directories and collect results.json."""
    records = []
    for rj in sorted(results_dir.glob("*/results.json")):
        with open(rj) as f:
            data = json.load(f)

        rec = {
            "experiment": data["experiment"],
            "description": data.get("description", ""),
            "axis": data["experiment"][0],  # A, B, C, …
            "rmse_temp": data["test_metrics"].get("test_rmse_temp"),
            "rmse_o2": data["test_metrics"].get("test_rmse_o2"),
            "best_val_phys": data["training"]["best_val_phys_loss"],
            "best_epoch": data["training"]["best_epoch"],
            "epochs_run": data["training"]["epochs_run"],
            "final_cd_gap": data["training"].get("final_cd_gap"),
            "final_e_pos": data["training"].get("final_e_pos"),
            "final_e_neg": data["training"].get("final_e_neg"),
            "wall_time_sec": data.get("wall_time_sec"),
            "total_sequences": data["data_summary"].get("total_sequences"),
            "real_sequences": data["data_summary"].get("real_sequences"),
            "edge_sequences": data["data_summary"].get("edge_sequences"),
        }

        # Attach key config values for grouping
        cfg = data.get("config", {})
        rec["alpha"] = cfg.get("alpha")
        rec["hidden_size"] = cfg.get("hidden_size")
        rec["seq_len"] = cfg.get("seq_len")
        rec["edge_fraction"] = cfg.get("edge_fraction")

        records.append(rec)

    if not records:
        print("⚠  No results.json files found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df


# =====================================================================
# Tables
# =====================================================================

def print_summary_table(df: pd.DataFrame) -> None:
    """Print a compact comparison table grouped by axis."""
    if df.empty:
        return

    cols = [
        "experiment", "rmse_temp", "rmse_o2",
        "best_val_phys", "best_epoch", "final_cd_gap",
        "total_sequences", "wall_time_sec",
    ]
    display = df[cols].copy()
    display = display.sort_values("experiment")

    # Formatting
    for c in ["rmse_temp", "rmse_o2", "best_val_phys"]:
        display[c] = display[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    display["final_cd_gap"] = display["final_cd_gap"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "—"
    )
    display["wall_time_sec"] = display["wall_time_sec"].map(
        lambda x: f"{x:.0f}s" if pd.notna(x) else "—"
    )

    print("\n" + "═" * 105)
    print("  ABLATION RESULTS SUMMARY")
    print("═" * 105)
    print(display.to_string(index=False))
    print("═" * 105 + "\n")


def print_axis_comparison(df: pd.DataFrame) -> None:
    """Print per-axis comparison tables with baseline delta."""
    if df.empty:
        return

    baseline_row = df[df["experiment"] == "A2_baseline"]
    if baseline_row.empty:
        print("⚠  No A2_baseline found — cannot compute deltas.\n")
        return

    bl_temp = baseline_row["rmse_temp"].values[0]
    bl_o2 = baseline_row["rmse_o2"].values[0]

    for axis in sorted(df["axis"].unique()):
        sub = df[df["axis"] == axis].sort_values("experiment")
        print(f"\n── Axis {axis} ──")
        for _, row in sub.iterrows():
            dt = row["rmse_temp"] - bl_temp if pd.notna(row["rmse_temp"]) else None
            do = row["rmse_o2"] - bl_o2 if pd.notna(row["rmse_o2"]) else None
            dt_str = f"{dt:+.4f}" if dt is not None else "—"
            do_str = f"{do:+.4f}" if do is not None else "—"
            print(
                f"  {row['experiment']:30s}  "
                f"Temp RMSE={row['rmse_temp']:.4f} (Δ{dt_str})  "
                f"O₂ RMSE={row['rmse_o2']:.4f} (Δ{do_str})"
            )


# =====================================================================
# Plots
# =====================================================================

def plot_axis_bar(df: pd.DataFrame, output_dir: Path) -> None:
    """Create grouped bar charts for each ablation axis."""
    if not HAS_MPL or df.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for axis in sorted(df["axis"].unique()):
        sub = df[df["axis"] == axis].sort_values("experiment")
        if sub.empty:
            continue

        names = sub["experiment"].str.replace(f"{axis}\\d+_", "", regex=True).tolist()
        temp_vals = sub["rmse_temp"].values
        o2_vals = sub["rmse_o2"].values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        x = np.arange(len(names))
        w = 0.6

        ax1.bar(x, temp_vals, w, color="#4CAF50", alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax1.set_ylabel("RMSE (°C)")
        ax1.set_title(f"Axis {axis} — Temperature RMSE")
        ax1.grid(axis="y", alpha=0.3)

        ax2.bar(x, o2_vals, w, color="#2196F3", alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax2.set_ylabel("RMSE (%)")
        ax2.set_title(f"Axis {axis} — Excess O₂ RMSE")
        ax2.grid(axis="y", alpha=0.3)

        fig.suptitle(f"Ablation Axis {axis}", fontsize=14, fontweight="bold")
        fig.tight_layout()

        figpath = output_dir / f"axis_{axis}_bar.png"
        fig.savefig(figpath, dpi=150)
        plt.close(fig)
        print(f"  Saved {figpath}")


def plot_training_curves(results_dir: Path, output_dir: Path) -> None:
    """Plot training loss curves from saved history.npz files."""
    if not HAS_MPL:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(results_dir.glob("*/history.npz"))
    if not npz_files:
        return

    # Overlay all physics losses
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for npz_path in npz_files:
        name = npz_path.parent.name
        data = np.load(npz_path)

        if "val_phys" in data:
            ax1.plot(data["val_phys"], label=name, alpha=0.7, linewidth=1)
        if "cd_gap" in data:
            ax2.plot(data["cd_gap"], label=name, alpha=0.7, linewidth=1)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val Physics Loss")
    ax1.set_title("Validation Physics Loss")
    ax1.legend(fontsize=6, ncol=2)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("CD Gap (E_neg − E_pos)")
    ax2.set_title("Contrastive Divergence Gap")
    ax2.legend(fontsize=6, ncol=2)
    ax2.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax2.grid(alpha=0.3)

    fig.suptitle("Ablation Training Curves", fontsize=14, fontweight="bold")
    fig.tight_layout()

    figpath = output_dir / "all_training_curves.png"
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {figpath}")


def plot_scatter_overview(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter: Temp RMSE vs O₂ RMSE, coloured by axis."""
    if not HAS_MPL or df.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    colours = {"A": "#E53935", "B": "#43A047", "C": "#1E88E5",
               "D": "#FB8C00", "E": "#8E24AA", "F": "#00897B"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for axis in sorted(df["axis"].unique()):
        sub = df[df["axis"] == axis]
        c = colours.get(axis, "#757575")
        ax.scatter(sub["rmse_temp"], sub["rmse_o2"],
                   label=f"Axis {axis}", c=c, s=80, alpha=0.8, edgecolors="white")
        for _, row in sub.iterrows():
            ax.annotate(row["experiment"], (row["rmse_temp"], row["rmse_o2"]),
                        fontsize=5, alpha=0.7, ha="center", va="bottom")

    ax.set_xlabel("Temperature RMSE (°C)")
    ax.set_ylabel("Excess O₂ RMSE (%)")
    ax.set_title("Ablation Overview — Test RMSE", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    figpath = output_dir / "scatter_overview.png"
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {figpath}")


# =====================================================================
# Entry point
# =====================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="Analyse ablation study results")
    p.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS))
    p.add_argument("--plots_dir", type=str, default=None,
                   help="Directory for plots (default: <results_dir>/plots)")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir) if args.plots_dir else results_dir / "plots"

    print(f"\nCollecting results from: {results_dir}\n")
    df = collect_results(results_dir)

    if df.empty:
        print("No results found. Run ablation experiments first.\n")
        return

    # Save CSV summary
    csv_path = results_dir / "ablation_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV: {csv_path}")

    # Tables
    print_summary_table(df)
    print_axis_comparison(df)

    # Plots
    if HAS_MPL:
        print("\nGenerating plots...")
        plot_axis_bar(df, plots_dir)
        plot_training_curves(results_dir, plots_dir)
        plot_scatter_overview(df, plots_dir)
        print(f"\nAll plots saved in {plots_dir}\n")
    else:
        print("\n⚠  matplotlib not installed — skipping plots\n")


if __name__ == "__main__":
    main()
