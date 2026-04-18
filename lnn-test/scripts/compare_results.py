from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate lnn-test benchmark outputs")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Benchmark run directory. Defaults to lnn-test/results/default.",
    )
    return parser.parse_args()


def format_mean_std(values: List[float], digits: int = 4) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{mean(values):.{digits}f} $\\pm$ {stdev(values):.{digits}f}"


def format_mean_std_plain(values: List[float], digits: int = 4) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{mean(values):.{digits}f} +/- {stdev(values):.{digits}f}"


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else float("nan")


def load_model_runs(results_dir: Path) -> Dict[str, List[dict]]:
    runs: Dict[str, List[dict]] = {}
    for model_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        model_runs = []
        for seed_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            results_path = seed_dir / "results.json"
            if results_path.exists():
                with open(results_path) as handle:
                    model_runs.append(json.load(handle))
        if model_runs:
            runs[model_dir.name] = model_runs
    return runs


def build_summary(runs: Dict[str, List[dict]]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    for model_name, model_runs in runs.items():
        metrics = [run["test_metrics"] for run in model_runs]
        disturbance = [run["disturbance_summary"] for run in model_runs]
        safety = [run["safety_summary"] for run in model_runs]
        training = [run["training"] for run in model_runs]
        summary[model_name] = {
            "n_runs": len(model_runs),
            "temp_rmse": [row["rmse_temp"] for row in metrics],
            "o2_rmse": [row["rmse_o2"] for row in metrics],
            "temp_r2": [row["r2_temp"] for row in metrics],
            "o2_r2": [row["r2_o2"] for row in metrics],
            "temp_mae": [row["mae_temp"] for row in metrics],
            "o2_mae": [row["mae_o2"] for row in metrics],
            "temp_mape": [row["mape_temp"] for row in metrics],
            "o2_mape": [row["mape_o2"] for row in metrics],
            "noise_deg_temp": [row["noise_deg_rmse_temp"] for row in disturbance],
            "noise_deg_o2": [row["noise_deg_rmse_o2"] for row in disturbance],
            "overall_deg_temp": [row["overall_deg_rmse_temp"] for row in disturbance],
            "overall_deg_o2": [row["overall_deg_rmse_o2"] for row in disturbance],
            "clean_critical": [row["clean_critical_rate"] for row in safety],
            "clean_total": [row["clean_total_rate"] for row in safety],
            "noise_critical": [row["noise_critical_rate"] for row in safety],
            "noise_total": [row["noise_total_rate"] for row in safety],
            "overall_critical": [row["overall_critical_rate"] for row in safety],
            "overall_total": [row["overall_total_rate"] for row in safety],
            # New: training efficiency
            "param_count": [run.get("param_count", 0) for run in model_runs],
            "best_epoch": [row["best_epoch"] for row in training],
            "epochs_run": [row["epochs_run"] for row in training],
            "wall_time_sec": [run.get("wall_time_sec", 0) for run in model_runs],
            # New: inference
            "latency_ms": [run.get("inference", {}).get("latency_per_sample_ms", float("nan")) for run in model_runs],
            "throughput": [run.get("inference", {}).get("throughput_samples_per_sec", float("nan")) for run in model_runs],
        }

        # Extreme test summaries
        extreme_keys = [
            "multi_drop_abs_temp", "multi_drop_abs_o2",
            "drift_abs_temp", "drift_abs_o2",
            "stuck_abs_temp", "stuck_abs_o2",
            "oscillation_abs_temp", "oscillation_abs_o2",
            "intermittent_abs_temp", "intermittent_abs_o2",
            "combined_abs_temp", "combined_abs_o2",
            "extreme_extrap_abs_temp", "extreme_extrap_abs_o2",
            "extreme_noise_abs_temp", "extreme_noise_abs_o2",
            "multi_drop_critical", "drift_critical",
            "stuck_critical", "combined_critical",
            "overall_extreme_abs_temp", "overall_extreme_abs_o2",
            "overall_extreme_critical",
        ]
        for ek in extreme_keys:
            summary[model_name][ek] = [
                run.get("extreme_summary", {}).get(ek, float("nan")) for run in model_runs
            ]

        # Per-disturbance-category means (from raw disturbance dict)
        categories = {
            "noise": lambda k: k.startswith("noise_sigma_"),
            "dropout": lambda k: k.startswith("dropout_"),
            "spike": lambda k: k.startswith("spike_"),
            "shuffle": lambda k: k == "temporal_shuffle",
            "extrap": lambda k: k.startswith("extrap_"),
            "scale": lambda k: k.startswith("scale_"),
        }
        # Realistic = process-level (dropout, scale, extrapolation)
        # Synthetic = sensor-level (noise, spike, shuffle)
        realistic_filter = lambda k: (k.startswith("dropout_") or k.startswith("scale_")
                                      or k.startswith("extrap_"))
        synthetic_filter = lambda k: (k.startswith("noise_sigma_") or k.startswith("spike_")
                                      or k == "temporal_shuffle")

        for cat_name, cat_filter in categories.items():
            cat_deg_temp = []
            cat_deg_o2 = []
            cat_abs_temp = []
            cat_abs_o2 = []
            for run in model_runs:
                disturbance_raw = run["disturbance"]
                keys = [k for k in disturbance_raw if k != "clean" and cat_filter(k)]
                if keys:
                    cat_deg_temp.append(mean(disturbance_raw[k]["degradation"]["rmse_temp"] for k in keys))
                    cat_deg_o2.append(mean(disturbance_raw[k]["degradation"]["rmse_o2"] for k in keys))
                    cat_abs_temp.append(mean(disturbance_raw[k]["metrics"]["rmse_temp"] for k in keys))
                    cat_abs_o2.append(mean(disturbance_raw[k]["metrics"]["rmse_o2"] for k in keys))
            summary[model_name][f"cat_{cat_name}_deg_temp"] = cat_deg_temp
            summary[model_name][f"cat_{cat_name}_deg_o2"] = cat_deg_o2
            summary[model_name][f"cat_{cat_name}_abs_temp"] = cat_abs_temp
            summary[model_name][f"cat_{cat_name}_abs_o2"] = cat_abs_o2

        # Grouped absolute perturbed RMSE: realistic vs synthetic
        for group_name, group_filter in [("realistic", realistic_filter), ("synthetic", synthetic_filter)]:
            grp_abs_temp = []
            grp_abs_o2 = []
            for run in model_runs:
                disturbance_raw = run["disturbance"]
                keys = [k for k in disturbance_raw if k != "clean" and group_filter(k)]
                if keys:
                    grp_abs_temp.append(mean(disturbance_raw[k]["metrics"]["rmse_temp"] for k in keys))
                    grp_abs_o2.append(mean(disturbance_raw[k]["metrics"]["rmse_o2"] for k in keys))
            summary[model_name][f"{group_name}_abs_temp"] = grp_abs_temp
            summary[model_name][f"{group_name}_abs_o2"] = grp_abs_o2

    return summary


# ─────────────────────────────────────────────────────────────────────
# Composite ranking score
# ─────────────────────────────────────────────────────────────────────

RANKING_WEIGHTS = {
    # Accuracy (50%)
    "temp_rmse": 0.15,
    "o2_rmse": 0.15,
    "temp_r2": 0.10,       # inverted (higher = better)
    "o2_r2": 0.10,         # inverted
    # Disturbance robustness — absolute perturbed RMSE (30%)
    # Realistic process disturbances weighted 2× synthetic
    "realistic_abs_temp": 0.08,
    "realistic_abs_o2": 0.08,
    "synthetic_abs_temp": 0.04,
    "synthetic_abs_o2": 0.04,
    # Not used (keep for back-compat in normalised dict):
    # "noise_deg_temp", "overall_deg_temp" etc. removed
    # Safety stability (20%)
    "noise_critical": 0.08,
    "noise_total": 0.04,
    "overall_critical": 0.04,
    "overall_total": 0.04,
}

HIGHER_IS_BETTER = {"temp_r2", "o2_r2"}


def compute_composite_scores(summary: Dict[str, dict]) -> Dict[str, dict]:
    """Rank models using a min-max normalised weighted composite score.

    Lower composite score = better model.
    """
    model_names = list(summary.keys())
    metric_keys = list(RANKING_WEIGHTS.keys())

    raw: Dict[str, Dict[str, float]] = {}
    for model_name in model_names:
        raw[model_name] = {}
        for key in metric_keys:
            values = summary[model_name].get(key, [])
            raw[model_name][key] = _safe_mean(values)

    normalised: Dict[str, Dict[str, float]] = {m: {} for m in model_names}
    for key in metric_keys:
        vals = [raw[m][key] for m in model_names]
        lo, hi = min(vals), max(vals)
        rng = hi - lo if hi - lo > 1e-12 else 1.0
        for m in model_names:
            norm = (raw[m][key] - lo) / rng
            if key in HIGHER_IS_BETTER:
                norm = 1.0 - norm
            normalised[m][key] = norm

    scores: Dict[str, dict] = {}
    for m in model_names:
        weighted = sum(normalised[m][k] * RANKING_WEIGHTS[k] for k in metric_keys)
        scores[m] = {
            "composite_score": round(weighted, 6),
            "normalised": {k: round(normalised[m][k], 4) for k in metric_keys},
        }

    ranked = sorted(scores.items(), key=lambda item: item[1]["composite_score"])
    for rank_position, (model_name, score_dict) in enumerate(ranked, 1):
        score_dict["rank"] = rank_position

    return scores


# ─────────────────────────────────────────────────────────────────────
# Publication-style markdown report
# ─────────────────────────────────────────────────────────────────────

def _bold_best(values: Dict[str, str], best_model: str, model_name: str, value: str) -> str:
    return f"**{value}**" if model_name == best_model else value


def _best_model_for(summary: Dict[str, dict], key: str) -> str:
    higher = key in HIGHER_IS_BETTER
    best = None
    best_val = None
    for model_name, row in summary.items():
        val = _safe_mean(row.get(key, []))
        if best_val is None or (higher and val > best_val) or (not higher and val < best_val):
            best_val = val
            best = model_name
    return best or ""


def build_markdown(summary: Dict[str, dict], scores: Dict[str, dict] | None = None) -> str:
    lines: list[str] = []

    # ── Title ──
    lines.extend([
        "# LNN vs LSTM vs MLP — Benchmark Report",
        "",
        "Evaluated on the PINN-augmented industrial furnace dataset.",
        "Each model is trained with three random seeds; values are reported as mean $\\pm$ std.",
        "",
    ])

    # ── Table 1: Prediction accuracy ──
    lines.extend([
        "## Table 1 — Prediction Accuracy (physical units)",
        "",
        "| Model | Runs | Temp RMSE (°C) | O₂ RMSE (%) | Temp MAE (°C) | O₂ MAE (%) | Temp R² | O₂ R² |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    acc_keys = ["temp_rmse", "o2_rmse", "temp_mae", "o2_mae", "temp_r2", "o2_r2"]
    best_for = {k: _best_model_for(summary, k) for k in acc_keys}

    for model_name, row in sorted(summary.items()):
        def _cell(key: str, digits: int = 4) -> str:
            val = format_mean_std(row[key], digits=digits)
            return f"**{val}**" if model_name == best_for[key] else val

        lines.append(
            f"| {model_name.upper()} | {row['n_runs']} | "
            f"{_cell('temp_rmse')} | {_cell('o2_rmse')} | "
            f"{_cell('temp_mae')} | {_cell('o2_mae')} | "
            f"{_cell('temp_r2', 6)} | {_cell('o2_r2', 6)} |"
        )

    lines.append("")

    # ── Table 2: Disturbance robustness — absolute perturbed RMSE ──
    lines.extend([
        "## Table 2 — Disturbance Robustness (absolute perturbed RMSE)",
        "",
        "Realistic = process-level disturbances (sensor dropout, feature scaling, extrapolation).",
        "Synthetic = sensor-level disturbances (Gaussian noise, spike injection, temporal shuffle).",
        "Lower = more robust.",
        "",
        "| Model | Realistic Temp | Realistic O₂ | Synthetic Temp | Synthetic O₂ |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ])
    abs_keys = ["realistic_abs_temp", "realistic_abs_o2", "synthetic_abs_temp", "synthetic_abs_o2"]
    best_abs = {k: _best_model_for(summary, k) for k in abs_keys}

    for model_name, row in sorted(summary.items()):
        def _acell(key: str) -> str:
            val = format_mean_std(row[key])
            return f"**{val}**" if model_name == best_abs[key] else val

        lines.append(
            f"| {model_name.upper()} | {_acell('realistic_abs_temp')} | {_acell('realistic_abs_o2')} | "
            f"{_acell('synthetic_abs_temp')} | {_acell('synthetic_abs_o2')} |"
        )

    lines.append("")

    # ── Table 3: Safety stability ──
    lines.extend([
        "## Table 3 — Safety Stability (violation rates under disturbance)",
        "",
        "Critical = O₂ < 1.5% or Temp > 500 °C.",
        "",
        "| Model | Clean Critical | Clean Total | Noise Critical | Noise Total | Overall Critical | Overall Total |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    safety_keys = ["clean_critical", "clean_total", "noise_critical", "noise_total", "overall_critical", "overall_total"]
    best_safety = {k: _best_model_for(summary, k) for k in safety_keys}

    for model_name, row in sorted(summary.items()):
        def _scell(key: str) -> str:
            val = format_mean_std(row[key])
            return f"**{val}**" if model_name == best_safety[key] else val

        lines.append(
            f"| {model_name.upper()} | {_scell('clean_critical')} | {_scell('clean_total')} | "
            f"{_scell('noise_critical')} | {_scell('noise_total')} | "
            f"{_scell('overall_critical')} | {_scell('overall_total')} |"
        )

    lines.append("")

    # ── Table 4: Composite ranking ──
    if scores:
        ranked = sorted(scores.items(), key=lambda item: item[1]["rank"])
        lines.extend([
            "## Table 4 — Composite Ranking",
            "",
            "Weighted score combining accuracy (50%), disturbance robustness (30% — realistic 2×, absolute RMSE), and safety (20%).",
            "Lower score = better overall.",
            "",
            "| Rank | Model | Composite Score |",
            "| ---: | :--- | ---: |",
        ])
        for model_name, info in ranked:
            lines.append(f"| {info['rank']} | {model_name.upper()} | {info['composite_score']:.4f} |")
        lines.append("")

    # ── Table 5: Model Efficiency ──
    lines.extend([
        "## Table 5 — Model Efficiency",
        "",
        "Parameter count, convergence speed, inference latency, and accuracy-per-parameter.",
        "",
        "| Model | Parameters | Best Epoch | Epochs Run | Wall Time (s) | Latency (ms/sample) | Throughput (samples/s) | RMSE / 10K Params |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    eff_best_for = {
        "param_count": _best_model_for(summary, "param_count"),
        "best_epoch": _best_model_for(summary, "best_epoch"),
        "latency_ms": _best_model_for(summary, "latency_ms"),
        "throughput": None,  # higher is better — find manually
    }
    # throughput: higher = better
    best_thr = None
    best_thr_val = -1.0
    for mn, rr in summary.items():
        v = _safe_mean(rr.get("throughput", []))
        if v > best_thr_val:
            best_thr_val = v
            best_thr = mn
    eff_best_for["throughput"] = best_thr

    for model_name, row in sorted(summary.items()):
        pc = int(_safe_mean(row["param_count"]))
        rmse_per_10k = _safe_mean(row["temp_rmse"]) / (pc / 10000) if pc > 0 else float("nan")
        pc_str = f"**{pc:,}**" if model_name == eff_best_for["param_count"] else f"{pc:,}"
        be_str = format_mean_std(row["best_epoch"], digits=0)
        be_str = f"**{be_str}**" if model_name == eff_best_for["best_epoch"] else be_str
        er_str = format_mean_std(row["epochs_run"], digits=0)
        wt_str = format_mean_std(row["wall_time_sec"], digits=1)
        lat_str = format_mean_std(row["latency_ms"], digits=3)
        lat_str = f"**{lat_str}**" if model_name == eff_best_for["latency_ms"] else lat_str
        thr_str = format_mean_std(row["throughput"], digits=0)
        thr_str = f"**{thr_str}**" if model_name == eff_best_for["throughput"] else thr_str
        rp_str = f"{rmse_per_10k:.4f}"
        lines.append(
            f"| {model_name.upper()} | {pc_str} | {be_str} | {er_str} | {wt_str} | {lat_str} | {thr_str} | {rp_str} |"
        )
    lines.append("")

    # ── Table 6: Seed Consistency ──
    lines.extend([
        "## Table 6 — Seed Consistency (Coefficient of Variation)",
        "",
        "Lower CoV = more reproducible across random seeds.",
        "",
        "| Model | Temp RMSE CoV | O₂ RMSE CoV | Noise Deg Temp CoV | Overall Deg Temp CoV |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ])
    cov_keys = ["temp_rmse", "o2_rmse", "noise_deg_temp", "overall_deg_temp"]
    cov_best = {}
    for key in cov_keys:
        best_m = None
        best_cov = float("inf")
        for mn, rr in summary.items():
            vals = rr.get(key, [])
            if len(vals) >= 2:
                mu = mean(vals)
                cov = stdev(vals) / mu if abs(mu) > 1e-12 else float("inf")
            else:
                cov = 0.0
            if cov < best_cov:
                best_cov = cov
                best_m = mn
        cov_best[key] = best_m

    for model_name, row in sorted(summary.items()):
        cells = []
        for key in cov_keys:
            vals = row.get(key, [])
            if len(vals) >= 2:
                mu = mean(vals)
                cov = stdev(vals) / mu * 100 if abs(mu) > 1e-12 else float("nan")
                cell = f"{cov:.2f}%"
            else:
                cell = "n/a"
            if model_name == cov_best[key]:
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append(f"| {model_name.upper()} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    lines.append("")

    # ── Table 7: Per-Disturbance Category Breakdown ──
    cat_names = ["noise", "dropout", "spike", "shuffle", "extrap", "scale"]
    cat_labels = ["Noise", "Dropout", "Spike", "Shuffle", "Extrapolation", "Scaling"]
    lines.extend([
        "## Table 7 — Per-Disturbance Category Breakdown (Temp RMSE degradation)",
        "",
        "Mean degradation ratio per perturbation category. Lower = more robust.",
        "",
        "| Model | " + " | ".join(cat_labels) + " |",
        "| :--- | " + " | ".join(["---:"] * len(cat_labels)) + " |",
    ])
    cat_best: Dict[str, str] = {}
    for cat in cat_names:
        key = f"cat_{cat}_deg_temp"
        cat_best[cat] = _best_model_for(summary, key)

    for model_name, row in sorted(summary.items()):
        cells = []
        for cat in cat_names:
            key = f"cat_{cat}_deg_temp"
            val = format_mean_std(row.get(key, []))
            if model_name == cat_best[cat]:
                val = f"**{val}**"
            cells.append(val)
        lines.append(f"| {model_name.upper()} | " + " | ".join(cells) + " |")
    lines.append("")

    # ── Table 8: Per-Disturbance Category Breakdown (O₂) ──
    lines.extend([
        "## Table 8 — Per-Disturbance Category Breakdown (O₂ RMSE degradation)",
        "",
        "| Model | " + " | ".join(cat_labels) + " |",
        "| :--- | " + " | ".join(["---:"] * len(cat_labels)) + " |",
    ])
    cat_best_o2: Dict[str, str] = {}
    for cat in cat_names:
        key = f"cat_{cat}_deg_o2"
        cat_best_o2[cat] = _best_model_for(summary, key)

    for model_name, row in sorted(summary.items()):
        cells = []
        for cat in cat_names:
            key = f"cat_{cat}_deg_o2"
            val = format_mean_std(row.get(key, []))
            if model_name == cat_best_o2[cat]:
                val = f"**{val}**"
            cells.append(val)
        lines.append(f"| {model_name.upper()} | " + " | ".join(cells) + " |")
    lines.append("")

    # ── Table 9: Extreme Stress Tests — Absolute Perturbed RMSE ──
    extreme_cats = [
        ("Multi-Sensor Drop", "multi_drop_abs_temp", "multi_drop_abs_o2"),
        ("Sensor Drift", "drift_abs_temp", "drift_abs_o2"),
        ("Stuck Sensor", "stuck_abs_temp", "stuck_abs_o2"),
        ("Oscillation", "oscillation_abs_temp", "oscillation_abs_o2"),
        ("Intermittent", "intermittent_abs_temp", "intermittent_abs_o2"),
        ("Combined Attack", "combined_abs_temp", "combined_abs_o2"),
        ("Extreme Extrap (5–10σ)", "extreme_extrap_abs_temp", "extreme_extrap_abs_o2"),
        ("Extreme Noise (3–5σ)", "extreme_noise_abs_temp", "extreme_noise_abs_o2"),
    ]
    # Only emit if extreme data exists
    has_extreme = any(
        not all(v != v for v in summary[m].get("overall_extreme_abs_temp", [float("nan")]))
        for m in summary
    )
    if has_extreme:
        lines.extend([
            "## Table 9 — Extreme Stress Tests (absolute perturbed RMSE)",
            "",
            "Realistic refinery failure scenarios. Lower = more robust.",
            "",
        ])
        # Build header
        hdr_cols = []
        for label, _, _ in extreme_cats:
            hdr_cols.extend([f"{label} Temp", f"{label} O₂"])
        lines.append("| Model | " + " | ".join(hdr_cols) + " |")
        lines.append("| :--- | " + " | ".join(["---:"] * len(hdr_cols)) + " |")

        # Best per column
        all_keys = []
        for _, kt, ko in extreme_cats:
            all_keys.extend([kt, ko])
        ext_best = {k: _best_model_for(summary, k) for k in all_keys}

        for model_name, row in sorted(summary.items()):
            cells = []
            for _, kt, ko in extreme_cats:
                for k in [kt, ko]:
                    val = format_mean_std(row.get(k, []))
                    if model_name == ext_best[k]:
                        val = f"**{val}**"
                    cells.append(val)
            lines.append(f"| {model_name.upper()} | " + " | ".join(cells) + " |")
        lines.append("")

        # ── Table 10: Extreme Safety ──
        lines.extend([
            "## Table 10 — Safety Under Extreme Conditions (critical violation rate)",
            "",
            "| Model | Multi-Sensor Drop | Sensor Drift | Stuck Sensor | Combined Attack | Overall Extreme |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ])
        safety_ext_keys = ["multi_drop_critical", "drift_critical", "stuck_critical",
                           "combined_critical", "overall_extreme_critical"]
        ext_safety_best = {k: _best_model_for(summary, k) for k in safety_ext_keys}

        for model_name, row in sorted(summary.items()):
            cells = []
            for k in safety_ext_keys:
                val = format_mean_std(row.get(k, []))
                if model_name == ext_safety_best[k]:
                    val = f"**{val}**"
                cells.append(val)
            lines.append(f"| {model_name.upper()} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend([
        "---",
        "",
        "Lower is better for RMSE, MAE, degradation ratio, violation rates, composite score, CoV, and latency.",
        "Higher is better for R² and throughput.",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir) if args.results_dir else (root / "results" / "default")
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    runs = load_model_runs(results_dir)
    if not runs:
        raise ValueError(f"No saved runs found under {results_dir}")

    summary = build_summary(runs)
    scores = compute_composite_scores(summary)
    report = build_markdown(summary, scores)

    with open(results_dir / "comparison_summary.json", "w") as handle:
        json.dump({"summary": summary, "composite_scores": scores}, handle, indent=2)
    with open(results_dir / "comparison_report.md", "w") as handle:
        handle.write(report)

    print(report)
    print(f"\nSaved comparison_summary.json and comparison_report.md to {results_dir}")


if __name__ == "__main__":
    main()
