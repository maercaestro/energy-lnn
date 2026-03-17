"""
analyze_lnn_ablation.py
========================
Post-hoc statistical analysis of LNN ablation results.

Collects all ``results.json`` files from ``results/lnn_ablation/``,
aggregates multi-seed runs, computes mean ± std for every metric,
runs paired statistical tests (A1 vs A2, etc.), and outputs:
  1. Console summary
  2. LaTeX-ready table (results/lnn_ablation/summary_table.tex)
  3. Full JSON report  (results/lnn_ablation/analysis.json)

Usage
-----
    cd eblnn
    python experiments/analyze_lnn_ablation.py

    # Only specific axis
    python experiments/analyze_lnn_ablation.py --axis A
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "lnn_ablation"

# Metrics we care about (ordered for display)
METRICS = [
    "test_rmse_temp", "test_mae_temp", "test_mape_temp", "test_r2_temp", "test_max_ae_temp",
    "test_rmse_o2",   "test_mae_o2",   "test_mape_o2",   "test_r2_o2",   "test_max_ae_o2",
]

METRIC_LABELS = {
    "test_rmse_temp": "RMSE Temp (°C)",
    "test_mae_temp":  "MAE Temp (°C)",
    "test_mape_temp": "MAPE Temp (%)",
    "test_r2_temp":   "R² Temp",
    "test_max_ae_temp": "MaxAE Temp (°C)",
    "test_rmse_o2":   "RMSE O₂ (%)",
    "test_mae_o2":    "MAE O₂ (%)",
    "test_mape_o2":   "MAPE O₂ (%)",
    "test_r2_o2":     "R² O₂",
    "test_max_ae_o2": "MaxAE O₂ (%)",
}

# These metrics are "higher is better" (for Δ interpretation)
HIGHER_IS_BETTER = {"test_r2_temp", "test_r2_o2"}


# =====================================================================
# Data Collection
# =====================================================================

def collect_results(axis: str | None = None) -> dict[str, list[dict]]:
    """
    Scan results/lnn_ablation/ and group results by experiment name.

    Returns
    -------
    {experiment_name: [result_dict, ...]}   (one per seed)
    """
    grouped: dict[str, list[dict]] = defaultdict(list)

    if not RESULTS_DIR.exists():
        print(f"No results directory found: {RESULTS_DIR}")
        sys.exit(1)

    for result_file in sorted(RESULTS_DIR.rglob("results.json")):
        with open(result_file) as f:
            data = json.load(f)

        exp_name = data.get("experiment", result_file.parent.name)

        # Filter by axis if requested
        if axis and not exp_name.startswith(axis):
            continue

        grouped[exp_name].append(data)

    return dict(grouped)


# =====================================================================
# Aggregation
# =====================================================================

def aggregate(grouped: dict[str, list[dict]]) -> dict[str, dict]:
    """
    For each experiment, compute mean ± std over seeds for every metric.

    Returns
    -------
    {experiment: {metric: {"mean": ..., "std": ..., "values": [...]}, "n_seeds": ...}}
    """
    summary = {}

    for exp, runs in sorted(grouped.items()):
        entry: dict = {"n_seeds": len(runs)}

        for m in METRICS:
            values = []
            for run in runs:
                v = run.get("test_metrics", {}).get(m)
                if v is not None:
                    values.append(float(v))

            if values:
                entry[m] = {
                    "mean": float(np.mean(values)),
                    "std":  float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "values": values,
                }
            else:
                entry[m] = {"mean": float("nan"), "std": 0.0, "values": []}

        # Training info
        epochs = [r.get("training", {}).get("epochs_run", 0) for r in runs]
        entry["epochs_run"] = {"mean": float(np.mean(epochs)), "values": epochs}

        summary[exp] = entry

    return summary


# =====================================================================
# Statistical Tests
# =====================================================================

def significance_tests(
    grouped: dict[str, list[dict]],
    reference: str = "A1_real_only",
    target: str = "A2_baseline",
) -> dict[str, dict]:
    """
    Run paired significance tests comparing `reference` vs `target`.

    Uses Wilcoxon signed-rank test if n≥6 seeds, else Mann-Whitney U.
    Falls back to simple two-sample t-test if scipy is unavailable.

    Returns
    -------
    {metric: {"statistic": ..., "p_value": ..., "test": ..., "significant": bool}}
    """
    try:
        from scipy import stats as sp_stats
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    if reference not in grouped or target not in grouped:
        return {}

    ref_runs = grouped[reference]
    tgt_runs = grouped[target]
    results = {}

    for m in METRICS:
        ref_vals = [r.get("test_metrics", {}).get(m) for r in ref_runs]
        tgt_vals = [r.get("test_metrics", {}).get(m) for r in tgt_runs]

        # Filter None
        ref_vals = [v for v in ref_vals if v is not None]
        tgt_vals = [v for v in tgt_vals if v is not None]

        if len(ref_vals) < 2 or len(tgt_vals) < 2:
            results[m] = {
                "test": "insufficient_data",
                "p_value": float("nan"),
                "significant": False,
                "note": f"ref n={len(ref_vals)}, tgt n={len(tgt_vals)}",
            }
            continue

        if not HAS_SCIPY:
            # Simple t-test without scipy
            ref_a, tgt_a = np.array(ref_vals), np.array(tgt_vals)
            diff_mean = np.mean(tgt_a) - np.mean(ref_a)
            pooled_se = np.sqrt(np.var(ref_a, ddof=1)/len(ref_a) + np.var(tgt_a, ddof=1)/len(tgt_a))
            t_stat = diff_mean / pooled_se if pooled_se > 0 else 0
            results[m] = {
                "test": "welch_t_approx",
                "statistic": float(t_stat),
                "p_value": float("nan"),
                "significant": False,
                "note": "scipy unavailable — install for proper p-values",
            }
            continue

        # Paired test if same number of seeds, else independent
        if len(ref_vals) == len(tgt_vals):
            stat, p = sp_stats.wilcoxon(ref_vals, tgt_vals, alternative="two-sided")
            test_name = "wilcoxon"
        else:
            stat, p = sp_stats.mannwhitneyu(ref_vals, tgt_vals, alternative="two-sided")
            test_name = "mann_whitney_u"

        results[m] = {
            "test": test_name,
            "statistic": float(stat),
            "p_value": float(p),
            "significant": p < 0.05,
        }

    return results


# =====================================================================
# Display & Export
# =====================================================================

def print_summary(summary: dict[str, dict], sig_tests: dict) -> None:
    """Pretty-print aggregated results to console."""

    # Header
    print(f"\n{'═' * 90}")
    print("  LNN ABLATION — STATISTICAL SUMMARY")
    print(f"{'═' * 90}\n")

    # Key metrics table: RMSE + MAE + R²
    key_metrics = ["test_rmse_temp", "test_mae_temp", "test_r2_temp",
                   "test_rmse_o2",   "test_mae_o2",   "test_r2_o2"]

    header = f"{'Experiment':25s} │ {'n':>2s} │ "
    header += " │ ".join(f"{METRIC_LABELS[m]:>16s}" for m in key_metrics)
    print(header)
    print("─" * len(header))

    for exp, entry in sorted(summary.items()):
        row = f"{exp:25s} │ {entry['n_seeds']:2d} │ "
        cells = []
        for m in key_metrics:
            mean = entry[m]["mean"]
            std  = entry[m]["std"]
            if std > 0:
                cells.append(f"{mean:>8.4f}±{std:<6.4f}")
            else:
                cells.append(f"{mean:>15.4f} ")
        row += " │ ".join(cells)
        print(row)

    # Significance tests
    if sig_tests:
        print(f"\n{'─' * 90}")
        print("  SIGNIFICANCE TESTS: A1_real_only vs A2_baseline")
        print(f"{'─' * 90}")
        for m in key_metrics:
            if m in sig_tests:
                t = sig_tests[m]
                sig_mark = "***" if t.get("significant") else "n.s."
                p_str = f"p={t['p_value']:.4f}" if not np.isnan(t["p_value"]) else "p=N/A"
                print(f"  {METRIC_LABELS[m]:20s}: {t['test']:18s}  {p_str:12s}  {sig_mark}")
        print()


def export_latex(summary: dict[str, dict], output_path: Path) -> None:
    """Export a LaTeX-ready table of RMSE + MAE + R² (mean ± std)."""

    key_metrics = ["test_rmse_temp", "test_mae_temp", "test_r2_temp",
                   "test_rmse_o2",   "test_mae_o2",   "test_r2_o2"]
    short_labels = ["RMSE$_T$", "MAE$_T$", "R$^2_T$",
                    "RMSE$_{O_2}$", "MAE$_{O_2}$", "R$^2_{O_2}$"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{LNN Ablation Study Results (mean $\pm$ std over seeds)}",
        r"\label{tab:lnn-ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l c " + "c " * len(key_metrics) + r"}",
        r"\toprule",
        r"Experiment & $n$ & " + " & ".join(short_labels) + r" \\",
        r"\midrule",
    ]

    for exp, entry in sorted(summary.items()):
        # Clean experiment name for LaTeX
        exp_clean = exp.replace("_", r"\_")
        cells = [exp_clean, str(entry["n_seeds"])]
        for m in key_metrics:
            mean = entry[m]["mean"]
            std  = entry[m]["std"]
            if std > 0:
                cells.append(f"${mean:.4f} \\pm {std:.4f}$")
            else:
                cells.append(f"${mean:.4f}$")
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"LaTeX table saved to {output_path}")


def export_json(summary: dict, sig_tests: dict, output_path: Path) -> None:
    """Save full analysis as JSON."""
    report = {
        "summary": {},
        "significance_tests": {},
    }

    # Convert numpy/float for JSON serialisation
    for exp, entry in summary.items():
        clean = {"n_seeds": entry["n_seeds"]}
        for m in METRICS:
            clean[m] = {
                "mean": entry[m]["mean"],
                "std":  entry[m]["std"],
                "n":    len(entry[m]["values"]),
            }
        report["summary"][exp] = clean

    for m, t in sig_tests.items():
        report["significance_tests"][m] = {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in t.items()
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Full analysis saved to {output_path}")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="Analyze LNN ablation results")
    p.add_argument("--axis", type=str, default=None,
                   help="Filter experiments by axis prefix (A/B/C/D/E)")
    p.add_argument("--reference", type=str, default="A1_real_only",
                   help="Reference experiment for significance tests")
    p.add_argument("--target", type=str, default="A2_baseline",
                   help="Target experiment for significance tests")
    args = p.parse_args()

    # 1. Collect
    grouped = collect_results(args.axis)
    if not grouped:
        print("No results found. Run ablation experiments first.")
        sys.exit(1)

    print(f"\nFound results for {len(grouped)} experiments:")
    for exp, runs in sorted(grouped.items()):
        print(f"  {exp:30s}  ({len(runs)} seed{'s' if len(runs) > 1 else ''})")

    # 2. Aggregate
    summary = aggregate(grouped)

    # 3. Significance tests
    sig_tests = significance_tests(grouped, args.reference, args.target)

    # 4. Display
    print_summary(summary, sig_tests)

    # 5. Export
    export_latex(summary, RESULTS_DIR / "summary_table.tex")
    export_json(summary, sig_tests, RESULTS_DIR / "analysis.json")


if __name__ == "__main__":
    main()
