"""
evaluate_robustness.py
======================
Out-of-Distribution (OOD) and robustness evaluation for trained LNN models.

Loads a trained model + its data pipeline, then runs a battery of
perturbation tests on the test set to measure how gracefully the model
degrades.  This is the *key evidence* for the paper narrative:

    "A model trained WITH PINN-augmented edge cases is more robust
     to OOD inputs than one trained WITHOUT."

Tests
-----
1. **Gaussian Noise**      — escalating σ on all inputs
2. **Sensor Dropout**      — zero out each of the 5 features one at a time
3. **Spike Injection**     — random large spikes at p% of timesteps
4. **Temporal Shuffle**    — shuffle timestep order within each sequence
5. **Extrapolation Shift** — shift all inputs by +kσ beyond training range
6. **Feature Scaling**     — scale individual features by 2× and 0.5×

For each test, RMSE and MAE are measured and the *degradation ratio*
(perturbed / clean) is reported.

Usage
-----
    cd eblnn

    # Evaluate a single experiment
    python experiments/evaluate_robustness.py -e A2_baseline

    # Compare two experiments head-to-head
    python experiments/evaluate_robustness.py --compare A1_real_only A2_baseline

    # Custom data directory
    python experiments/evaluate_robustness.py -e A2_baseline --data_dir /path/to/dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lnn_model import LNN
from src.data_real import RealDataPipeline, INPUT_SIZE, TARGET_SIZE


# =====================================================================
# Core evaluation helpers
# =====================================================================

def load_model_and_pipeline(
    experiment_dir: Path,
    data_dir: Path,
    device: str,
) -> Tuple[LNN, RealDataPipeline, dict]:
    """Load a trained model and rebuild its data pipeline from saved config."""

    results_path = experiment_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No results.json in {experiment_dir}")

    with open(results_path) as f:
        results = json.load(f)

    cfg = results["config"]

    # Rebuild pipeline (same seed/split → identical test set)
    real_csv = (
        str(data_dir / "real_furnace_eblnn.csv")
        if cfg.get("use_real", True) else None
    )
    edge_csv = (
        str(data_dir / "edge_cases_v2_eblnn.csv")
        if cfg.get("use_edge", True) else None
    )

    pipeline = RealDataPipeline(
        real_csv=real_csv,
        edge_csv=edge_csv,
        scenarios=cfg.get("scenarios"),
        confidence_filter=cfg.get("confidence_filter"),
        edge_fraction=cfg.get("edge_fraction", 1.0),
        seq_len=cfg["seq_len"],
        stride=cfg.get("stride", cfg["seq_len"]),
        batch_size=cfg["batch_size"],
        test_size=cfg["test_size"],
        val_size=cfg["val_size"],
        seed=cfg["seed"],
    ).build()

    # Load model
    model = LNN(
        input_size=INPUT_SIZE,
        hidden_size=cfg["hidden_size"],
        phys_output_size=TARGET_SIZE,
    ).to(device)

    model_path = experiment_dir / "models" / "best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    return model, pipeline, cfg


def predict(
    model: LNN,
    loader: DataLoader,
    device: str,
    target_scaler=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference, return (y_true, y_pred) in physical units."""
    all_true, all_pred = [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            pred, _ = model(x_batch)
            all_true.append(y_batch.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_true).reshape(-1, 2)
    y_pred = np.concatenate(all_pred).reshape(-1, 2)

    if target_scaler is not None:
        y_true = target_scaler.inverse_transform(y_true)
        y_pred = target_scaler.inverse_transform(y_pred)

    return y_true, y_pred


def predict_perturbed(
    model: LNN,
    loader: DataLoader,
    device: str,
    target_scaler,
    perturb_fn,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference with a perturbation applied to each input batch."""
    all_true, all_pred = [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_pert = perturb_fn(x_batch.clone())
            x_pert = x_pert.to(device)
            pred, _ = model(x_pert)
            all_true.append(y_batch.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_true).reshape(-1, 2)
    y_pred = np.concatenate(all_pred).reshape(-1, 2)

    if target_scaler is not None:
        y_true = target_scaler.inverse_transform(y_true)
        y_pred = target_scaler.inverse_transform(y_pred)

    return y_true, y_pred


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE and MAE per column."""
    err = y_pred - y_true
    return {
        "rmse_temp": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "rmse_o2":   float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "mae_temp":  float(np.mean(np.abs(err[:, 0]))),
        "mae_o2":    float(np.mean(np.abs(err[:, 1]))),
    }


# =====================================================================
# Perturbation functions (operate on scaled tensors)
# =====================================================================

def make_gaussian_noise(sigma: float):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * sigma
    return fn


def make_sensor_dropout(feature_idx: int):
    """Zero out a single feature across all timesteps."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        x[:, :, feature_idx] = 0.0
        return x
    return fn


def make_spike_injection(fraction: float = 0.1, magnitude: float = 5.0):
    """Inject large spikes at random timesteps."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        mask = torch.rand(x.shape[0], x.shape[1], 1) < fraction
        spikes = torch.randn(x.shape[0], x.shape[1], x.shape[2]) * magnitude
        return x + spikes * mask.float()
    return fn


def make_temporal_shuffle():
    """Shuffle timestep order within each sequence."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        for i in range(B):
            perm = torch.randperm(T)
            x[i] = x[i, perm]
        return x
    return fn


def make_extrapolation_shift(shift_sigma: float):
    """Shift all features by +k standard deviations (data is already scaled)."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        return x + shift_sigma
    return fn


def make_feature_scale(feature_idx: int, scale: float):
    """Scale a single feature by a factor."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        x[:, :, feature_idx] = x[:, :, feature_idx] * scale
        return x
    return fn


# =====================================================================
# Full test battery
# =====================================================================

FEATURE_NAMES = ["fuel_flow", "air_fuel_ratio", "current_temp",
                 "inflow_temp", "inflow_rate"]


def run_robustness_tests(
    model: LNN,
    pipeline: RealDataPipeline,
    device: str,
) -> Dict[str, dict]:
    """
    Run the full battery of OOD/robustness tests.

    Returns
    -------
    {test_name: {"metrics": {...}, "degradation": {...}}}
    """
    loader = pipeline.test_loader
    scaler = pipeline.target_scaler

    # Baseline (clean)
    y_true, y_pred = predict(model, loader, device, scaler)
    clean = compute_metrics(y_true, y_pred)

    results: Dict[str, dict] = {"clean": {"metrics": clean}}

    def _run(name: str, perturb_fn):
        _, y_p = predict_perturbed(model, loader, device, scaler, perturb_fn)
        m = compute_metrics(y_true, y_p)
        deg = {
            k: m[k] / clean[k] if clean[k] > 1e-12 else float("nan")
            for k in m
        }
        results[name] = {"metrics": m, "degradation": deg}

    # 1. Gaussian noise at escalating levels
    for sigma in [0.1, 0.25, 0.5, 1.0, 2.0]:
        _run(f"noise_sigma_{sigma}", make_gaussian_noise(sigma))

    # 2. Sensor dropout (one feature at a time)
    for i, fname in enumerate(FEATURE_NAMES):
        _run(f"dropout_{fname}", make_sensor_dropout(i))

    # 3. Spike injection
    for frac in [0.05, 0.10, 0.20]:
        _run(f"spike_frac_{frac}", make_spike_injection(frac, magnitude=5.0))

    # 4. Temporal shuffle
    _run("temporal_shuffle", make_temporal_shuffle())

    # 5. Extrapolation shift
    for k in [1.0, 2.0, 3.0]:
        _run(f"extrap_shift_{k}sigma", make_extrapolation_shift(k))

    # 6. Feature scaling (2× and 0.5×)
    for i, fname in enumerate(FEATURE_NAMES):
        _run(f"scale_2x_{fname}", make_feature_scale(i, 2.0))
        _run(f"scale_0.5x_{fname}", make_feature_scale(i, 0.5))

    return results


# =====================================================================
# Display
# =====================================================================

def print_results(name: str, results: Dict[str, dict]) -> None:
    """Pretty-print robustness results for one experiment."""
    clean = results["clean"]["metrics"]
    print(f"\n{'═' * 80}")
    print(f"  ROBUSTNESS REPORT: {name}")
    print(f"{'═' * 80}")
    print(f"  Clean baseline — RMSE Temp: {clean['rmse_temp']:.4f}°C  "
          f"RMSE O₂: {clean['rmse_o2']:.4f}%")
    print(f"{'─' * 80}")
    print(f"  {'Test':35s} │ {'RMSE_T':>8s} │ {'RMSE_O2':>8s} │ {'Deg_T':>6s} │ {'Deg_O2':>6s}")
    print(f"  {'─' * 34} ┼ {'─' * 8} ┼ {'─' * 8} ┼ {'─' * 6} ┼ {'─' * 6}")

    for test_name, entry in sorted(results.items()):
        if test_name == "clean":
            continue
        m = entry["metrics"]
        d = entry["degradation"]
        print(f"  {test_name:35s} │ {m['rmse_temp']:8.4f} │ {m['rmse_o2']:8.4f} │ "
              f"{d['rmse_temp']:5.2f}× │ {d['rmse_o2']:5.2f}×")

    print()


def print_comparison(
    name_a: str, res_a: Dict[str, dict],
    name_b: str, res_b: Dict[str, dict],
) -> None:
    """Side-by-side comparison: which model degrades less?"""
    print(f"\n{'═' * 95}")
    print(f"  ROBUSTNESS COMPARISON: {name_a} vs {name_b}")
    print(f"{'═' * 95}")

    clean_a = res_a["clean"]["metrics"]
    clean_b = res_b["clean"]["metrics"]
    print(f"  Clean RMSE Temp — {name_a}: {clean_a['rmse_temp']:.4f}  "
          f"{name_b}: {clean_b['rmse_temp']:.4f}")
    print(f"  Clean RMSE O₂   — {name_a}: {clean_a['rmse_o2']:.4f}  "
          f"{name_b}: {clean_b['rmse_o2']:.4f}")
    print(f"{'─' * 95}")
    print(f"  {'Test':30s} │ {'Deg_T(' + name_a[:6] + ')':>12s} │ {'Deg_T(' + name_b[:6] + ')':>12s} │ "
          f"{'Deg_O2(' + name_a[:6] + ')':>12s} │ {'Deg_O2(' + name_b[:6] + ')':>12s} │ {'Winner':>8s}")
    print(f"  {'─' * 29} ┼ {'─' * 12} ┼ {'─' * 12} ┼ {'─' * 12} ┼ {'─' * 12} ┼ {'─' * 8}")

    wins = {name_a: 0, name_b: 0, "tie": 0}

    common_tests = set(res_a.keys()) & set(res_b.keys()) - {"clean"}
    for test_name in sorted(common_tests):
        da = res_a[test_name]["degradation"]
        db = res_b[test_name]["degradation"]

        # Average degradation across both targets
        avg_a = (da["rmse_temp"] + da["rmse_o2"]) / 2
        avg_b = (db["rmse_temp"] + db["rmse_o2"]) / 2

        if avg_a < avg_b - 0.01:
            winner = name_a[:8]
            wins[name_a] += 1
        elif avg_b < avg_a - 0.01:
            winner = name_b[:8]
            wins[name_b] += 1
        else:
            winner = "tie"
            wins["tie"] += 1

        print(f"  {test_name:30s} │ {da['rmse_temp']:11.2f}× │ {db['rmse_temp']:11.2f}× │ "
              f"{da['rmse_o2']:11.2f}× │ {db['rmse_o2']:11.2f}× │ {winner:>8s}")

    print(f"\n  Score: {name_a} wins {wins[name_a]}, "
          f"{name_b} wins {wins[name_b]}, ties {wins['tie']}")
    print()


# =====================================================================
# Main
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OOD/Robustness evaluation for trained LNN models"
    )
    p.add_argument("-e", "--experiment", type=str, default=None,
                   help="Single experiment to evaluate")
    p.add_argument("--compare", nargs=2, metavar=("EXP_A", "EXP_B"),
                   help="Compare two experiments head-to-head")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--results_dir", type=str, default=None,
                   help="Results root (default: results/lnn_ablation/)")
    return p.parse_args()


def _resolve_dir(name: str, results_root: Path) -> Path:
    """Find experiment directory, handling seed suffixes."""
    direct = results_root / name
    if direct.exists():
        return direct
    # Try to find any seed variant
    candidates = sorted(results_root.glob(f"{name}*"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No results for '{name}' in {results_root}")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else (ROOT.parent / "dataset")
    results_root = Path(args.results_dir) if args.results_dir else (ROOT / "results" / "lnn_ablation")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.compare:
        name_a, name_b = args.compare
        dir_a = _resolve_dir(name_a, results_root)
        dir_b = _resolve_dir(name_b, results_root)

        print(f"\nLoading {name_a} from {dir_a} ...")
        model_a, pipe_a, _ = load_model_and_pipeline(dir_a, data_dir, device)
        print(f"\nLoading {name_b} from {dir_b} ...")
        model_b, pipe_b, _ = load_model_and_pipeline(dir_b, data_dir, device)

        print(f"\nRunning robustness tests for {name_a} ...")
        res_a = run_robustness_tests(model_a, pipe_a, device)
        print(f"Running robustness tests for {name_b} ...")
        res_b = run_robustness_tests(model_b, pipe_b, device)

        print_results(name_a, res_a)
        print_results(name_b, res_b)
        print_comparison(name_a, res_a, name_b, res_b)

        # Save
        out = results_root / f"robustness_{name_a}_vs_{name_b}.json"
        with open(out, "w") as f:
            json.dump({name_a: res_a, name_b: res_b}, f, indent=2, default=str)
        print(f"Results saved to {out}")

    elif args.experiment:
        exp_dir = _resolve_dir(args.experiment, results_root)
        print(f"\nLoading {args.experiment} from {exp_dir} ...")
        model, pipeline, _ = load_model_and_pipeline(exp_dir, data_dir, device)

        print(f"Running robustness tests ...")
        res = run_robustness_tests(model, pipeline, device)
        print_results(args.experiment, res)

        out = exp_dir / "robustness.json"
        with open(out, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"Results saved to {out}")

    else:
        print("Specify --experiment or --compare. Use -h for help.")
        sys.exit(1)


if __name__ == "__main__":
    main()
