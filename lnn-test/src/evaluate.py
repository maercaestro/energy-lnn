from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


FEATURE_NAMES = [
    "fuel_flow",
    "air_fuel_ratio",
    "current_temp",
    "inflow_temp",
    "inflow_rate",
]

O2_LOW_CRIT = 1.5
O2_HIGH_WARN = 15.0
TEMP_HIGH_CRIT = 500.0
TEMP_LOW_WARN = 25.0


def _predict(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    target_scaler=None,
    perturb_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    all_true = []
    all_pred = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            if perturb_fn is not None:
                x_batch = perturb_fn(x_batch.clone())
            x_batch = x_batch.to(device)
            pred = model(x_batch)[0]
            all_true.append(y_batch.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_true).reshape(-1, 2)
    y_pred = np.concatenate(all_pred).reshape(-1, 2)

    if target_scaler is not None:
        y_true = target_scaler.inverse_transform(y_true)
        y_pred = target_scaler.inverse_transform(y_pred)

    return y_true, y_pred


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    names = ["temp", "o2"]
    for index, name in enumerate(names):
        err = y_pred[:, index] - y_true[:, index]
        abs_err = np.abs(err)
        nonzero = np.abs(y_true[:, index]) > 1e-8
        mape = (
            float(np.mean(abs_err[nonzero] / np.abs(y_true[:, index][nonzero])) * 100)
            if nonzero.any()
            else float("nan")
        )
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((y_true[:, index] - np.mean(y_true[:, index])) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        metrics[f"rmse_{name}"] = float(np.sqrt(np.mean(err ** 2)))
        metrics[f"mae_{name}"] = float(np.mean(abs_err))
        metrics[f"mape_{name}"] = mape
        metrics[f"r2_{name}"] = r2
        metrics[f"max_ae_{name}"] = float(np.max(abs_err))
    return metrics


def make_gaussian_noise(sigma: float):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * sigma
    return fn


def make_sensor_dropout(feature_idx: int):
    def fn(x: torch.Tensor) -> torch.Tensor:
        x[:, :, feature_idx] = 0.0
        return x
    return fn


def make_spike_injection(fraction: float = 0.1, magnitude: float = 5.0):
    def fn(x: torch.Tensor) -> torch.Tensor:
        mask = torch.rand(x.shape[0], x.shape[1], 1) < fraction
        spikes = torch.randn(x.shape[0], x.shape[1], x.shape[2]) * magnitude
        return x + spikes * mask.float()
    return fn


def make_temporal_shuffle():
    def fn(x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        for index in range(batch_size):
            perm = torch.randperm(seq_len)
            x[index] = x[index, perm]
        return x
    return fn


def make_extrapolation_shift(shift_sigma: float):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return x + shift_sigma
    return fn


def make_feature_scale(feature_idx: int, scale: float):
    def fn(x: torch.Tensor) -> torch.Tensor:
        x[:, :, feature_idx] = x[:, :, feature_idx] * scale
        return x
    return fn


# ─────────────────────────────────────────────────────────────────────
# Extreme / realistic refinery perturbations
# ─────────────────────────────────────────────────────────────────────

def make_multi_sensor_dropout(feature_indices: list[int]):
    """Simultaneous failure of multiple sensors (zeroed out)."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        for idx in feature_indices:
            x[:, :, idx] = 0.0
        return x
    return fn


def make_sensor_drift(feature_idx: int, drift_rate: float):
    """Gradual calibration drift: value increases linearly over the sequence.

    drift_rate is in std-units per timestep (data is standardised).
    """
    def fn(x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        ramp = torch.linspace(0, drift_rate * seq_len, seq_len, device=x.device)
        x[:, :, feature_idx] = x[:, :, feature_idx] + ramp.unsqueeze(0)
        return x
    return fn


def make_stuck_sensor(feature_idx: int):
    """Sensor freezes at its first-timestep value for entire sequence."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        x[:, :, feature_idx] = x[:, 0:1, feature_idx]
        return x
    return fn


def make_oscillation(feature_idx: int, amplitude: float, period: int = 5):
    """Rapid oscillation injected into a sensor (control-loop instability)."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        t = torch.arange(seq_len, dtype=x.dtype, device=x.device)
        osc = amplitude * torch.sin(2 * 3.14159 * t / period)
        x[:, :, feature_idx] = x[:, :, feature_idx] + osc.unsqueeze(0)
        return x
    return fn


def make_intermittent_dropout(feature_idx: int, drop_prob: float = 0.3):
    """Random intermittent sensor dropouts (sensor cuts in/out)."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        mask = torch.rand(x.shape[0], x.shape[1], device=x.device) < drop_prob
        x[:, :, feature_idx] = x[:, :, feature_idx] * (~mask).float()
        return x
    return fn


def make_combined_attack(noise_sigma: float, dropout_idx: int, spike_frac: float):
    """Simultaneous noise + sensor dropout + spike injection."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        x = x + torch.randn_like(x) * noise_sigma
        x[:, :, dropout_idx] = 0.0
        mask = torch.rand(x.shape[0], x.shape[1], 1) < spike_frac
        spikes = torch.randn(x.shape[0], x.shape[1], x.shape[2]) * 5.0
        x = x + spikes * mask.float()
        return x
    return fn


def run_extreme_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    target_scaler=None,
) -> Dict[str, dict]:
    """Extreme refinery-realistic stress tests."""
    y_true, y_pred = _predict(model, loader, device, target_scaler)
    clean_metrics = compute_regression_metrics(y_true, y_pred)
    results: Dict[str, dict] = {"clean": {"metrics": clean_metrics}}

    def run_case(name: str, perturb_fn):
        _, perturbed_pred = _predict(model, loader, device, target_scaler, perturb_fn=perturb_fn)
        metrics = compute_regression_metrics(y_true, perturbed_pred)
        degradation = {
            key: (metrics[key] / clean_metrics[key] if clean_metrics[key] > 1e-12 else float("nan"))
            for key in ["rmse_temp", "rmse_o2", "mae_temp", "mae_o2"]
        }
        safety = compute_safety_violations(perturbed_pred)
        results[name] = {
            "metrics": metrics,
            "degradation": degradation,
            "safety": safety,
        }

    # --- Multi-sensor failures ---
    run_case("multi_drop_fuel+air", make_multi_sensor_dropout([0, 1]))
    run_case("multi_drop_fuel+temp", make_multi_sensor_dropout([0, 2]))
    run_case("multi_drop_3sensors", make_multi_sensor_dropout([0, 2, 3]))

    # --- Sensor drift (calibration loss) ---
    for idx, name in enumerate(FEATURE_NAMES):
        run_case(f"drift_{name}_slow", make_sensor_drift(idx, 0.05))
        run_case(f"drift_{name}_fast", make_sensor_drift(idx, 0.2))

    # --- Stuck sensors ---
    for idx, name in enumerate(FEATURE_NAMES):
        run_case(f"stuck_{name}", make_stuck_sensor(idx))

    # --- Rapid oscillations ---
    for idx, name in enumerate(FEATURE_NAMES):
        run_case(f"oscillation_{name}", make_oscillation(idx, amplitude=2.0, period=5))

    # --- Intermittent dropout ---
    for idx, name in enumerate(FEATURE_NAMES):
        run_case(f"intermittent_{name}_30pct", make_intermittent_dropout(idx, 0.3))
        run_case(f"intermittent_{name}_50pct", make_intermittent_dropout(idx, 0.5))

    # --- Combined attacks ---
    run_case("combined_mild", make_combined_attack(0.25, 0, 0.05))
    run_case("combined_moderate", make_combined_attack(0.5, 2, 0.10))
    run_case("combined_severe", make_combined_attack(1.0, 2, 0.20))

    # --- Extreme extrapolation ---
    for shift in [5.0, 10.0]:
        run_case(f"extrap_shift_{shift}sigma", make_extrapolation_shift(shift))

    # --- Extreme noise ---
    for sigma in [3.0, 5.0]:
        run_case(f"extreme_noise_{sigma}", make_gaussian_noise(sigma))

    return results


def summarize_extreme_results(results: Dict[str, dict]) -> Dict[str, float]:
    """Summarize extreme evaluation into category-level means."""
    perturbations = [k for k in results if k != "clean"]

    def _cat_mean(prefix: str, metric: str = "rmse_temp") -> float:
        keys = [k for k in perturbations if k.startswith(prefix)]
        if not keys:
            return float("nan")
        return _mean(results[k]["metrics"][metric] for k in keys)

    def _cat_safety(prefix: str) -> float:
        keys = [k for k in perturbations if k.startswith(prefix)]
        if not keys:
            return float("nan")
        return _mean(results[k]["safety"]["viol_critical"] for k in keys)

    return {
        "multi_drop_abs_temp": _cat_mean("multi_drop"),
        "multi_drop_abs_o2": _cat_mean("multi_drop", "rmse_o2"),
        "drift_abs_temp": _cat_mean("drift_"),
        "drift_abs_o2": _cat_mean("drift_", "rmse_o2"),
        "stuck_abs_temp": _cat_mean("stuck_"),
        "stuck_abs_o2": _cat_mean("stuck_", "rmse_o2"),
        "oscillation_abs_temp": _cat_mean("oscillation_"),
        "oscillation_abs_o2": _cat_mean("oscillation_", "rmse_o2"),
        "intermittent_abs_temp": _cat_mean("intermittent_"),
        "intermittent_abs_o2": _cat_mean("intermittent_", "rmse_o2"),
        "combined_abs_temp": _cat_mean("combined_"),
        "combined_abs_o2": _cat_mean("combined_", "rmse_o2"),
        "extreme_extrap_abs_temp": _cat_mean("extrap_shift_"),
        "extreme_extrap_abs_o2": _cat_mean("extrap_shift_", "rmse_o2"),
        "extreme_noise_abs_temp": _cat_mean("extreme_noise_"),
        "extreme_noise_abs_o2": _cat_mean("extreme_noise_", "rmse_o2"),
        # Safety under extreme conditions
        "multi_drop_critical": _cat_safety("multi_drop"),
        "drift_critical": _cat_safety("drift_"),
        "stuck_critical": _cat_safety("stuck_"),
        "combined_critical": _cat_safety("combined_"),
        "overall_extreme_abs_temp": _mean(results[k]["metrics"]["rmse_temp"] for k in perturbations),
        "overall_extreme_abs_o2": _mean(results[k]["metrics"]["rmse_o2"] for k in perturbations),
        "overall_extreme_critical": _mean(results[k]["safety"]["viol_critical"] for k in perturbations),
    }


def compute_safety_violations(y_pred: np.ndarray) -> Dict[str, float]:
    temp = y_pred[:, 0]
    o2 = y_pred[:, 1]
    n_samples = len(temp)

    o2_low = o2 < O2_LOW_CRIT
    o2_high = o2 > O2_HIGH_WARN
    temp_high = temp > TEMP_HIGH_CRIT
    temp_low = temp < TEMP_LOW_WARN

    critical = o2_low | temp_high
    total = o2_low | o2_high | temp_high | temp_low

    return {
        "viol_o2_low": float(o2_low.sum() / n_samples),
        "viol_o2_high": float(o2_high.sum() / n_samples),
        "viol_temp_high": float(temp_high.sum() / n_samples),
        "viol_temp_low": float(temp_low.sum() / n_samples),
        "viol_critical": float(critical.sum() / n_samples),
        "viol_total": float(total.sum() / n_samples),
        "n_critical": int(critical.sum()),
        "n_total_viol": int(total.sum()),
        "n_samples": n_samples,
    }


def run_disturbance_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    target_scaler=None,
) -> Dict[str, dict]:
    y_true, y_pred = _predict(model, loader, device, target_scaler)
    clean_metrics = compute_regression_metrics(y_true, y_pred)
    results: Dict[str, dict] = {"clean": {"metrics": clean_metrics}}

    def run_case(name: str, perturb_fn):
        _, perturbed_pred = _predict(model, loader, device, target_scaler, perturb_fn=perturb_fn)
        metrics = compute_regression_metrics(y_true, perturbed_pred)
        degradation = {
            key: (metrics[key] / clean_metrics[key] if clean_metrics[key] > 1e-12 else float("nan"))
            for key in ["rmse_temp", "rmse_o2", "mae_temp", "mae_o2"]
        }
        results[name] = {
            "metrics": metrics,
            "degradation": degradation,
        }

    for sigma in [0.1, 0.25, 0.5, 1.0, 2.0]:
        run_case(f"noise_sigma_{sigma}", make_gaussian_noise(sigma))

    for index, feature_name in enumerate(FEATURE_NAMES):
        run_case(f"dropout_{feature_name}", make_sensor_dropout(index))

    for fraction in [0.05, 0.10, 0.20]:
        run_case(f"spike_frac_{fraction}", make_spike_injection(fraction, magnitude=5.0))

    run_case("temporal_shuffle", make_temporal_shuffle())

    for shift in [1.0, 2.0, 3.0]:
        run_case(f"extrap_shift_{shift}sigma", make_extrapolation_shift(shift))

    for index, feature_name in enumerate(FEATURE_NAMES):
        run_case(f"scale_2x_{feature_name}", make_feature_scale(index, 2.0))
        run_case(f"scale_0.5x_{feature_name}", make_feature_scale(index, 0.5))

    return results


def run_safety_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    target_scaler=None,
) -> Dict[str, dict]:
    _, clean_pred = _predict(model, loader, device, target_scaler)
    clean_safety = compute_safety_violations(clean_pred)
    results: Dict[str, dict] = {
        "clean": {
            "safety": clean_safety,
            "thresholds": {
                "o2_low_critical": O2_LOW_CRIT,
                "o2_high_warning": O2_HIGH_WARN,
                "temp_high_critical": TEMP_HIGH_CRIT,
                "temp_low_warning": TEMP_LOW_WARN,
            },
        }
    }

    def run_case(name: str, perturb_fn):
        _, perturbed_pred = _predict(model, loader, device, target_scaler, perturb_fn=perturb_fn)
        safety = compute_safety_violations(perturbed_pred)
        ratio_critical = (
            safety["viol_critical"] / clean_safety["viol_critical"]
            if clean_safety["viol_critical"] > 1e-9
            else float("nan")
        )
        ratio_total = (
            safety["viol_total"] / clean_safety["viol_total"]
            if clean_safety["viol_total"] > 1e-9
            else float("nan")
        )
        results[name] = {
            "safety": safety,
            "ratio_critical": ratio_critical,
            "ratio_total": ratio_total,
        }

    for sigma in [0.1, 0.25, 0.5, 1.0, 2.0]:
        run_case(f"noise_sigma_{sigma}", make_gaussian_noise(sigma))

    for index, feature_name in enumerate(FEATURE_NAMES):
        run_case(f"dropout_{feature_name}", make_sensor_dropout(index))

    for fraction in [0.05, 0.10, 0.20]:
        run_case(f"spike_frac_{fraction}", make_spike_injection(fraction, magnitude=5.0))

    run_case("temporal_shuffle", make_temporal_shuffle())

    for shift in [1.0, 2.0, 3.0]:
        run_case(f"extrap_shift_{shift}sigma", make_extrapolation_shift(shift))

    for index, feature_name in enumerate(FEATURE_NAMES):
        run_case(f"scale_2x_{feature_name}", make_feature_scale(index, 2.0))
        run_case(f"scale_0.5x_{feature_name}", make_feature_scale(index, 0.5))

    return results


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def summarize_disturbance_results(results: Dict[str, dict]) -> Dict[str, float]:
    perturbations = [name for name in results if name != "clean"]
    noise_tests = [name for name in perturbations if name.startswith("noise_")]

    return {
        "noise_deg_rmse_temp": _mean(results[name]["degradation"]["rmse_temp"] for name in noise_tests),
        "noise_deg_rmse_o2": _mean(results[name]["degradation"]["rmse_o2"] for name in noise_tests),
        "overall_deg_rmse_temp": _mean(results[name]["degradation"]["rmse_temp"] for name in perturbations),
        "overall_deg_rmse_o2": _mean(results[name]["degradation"]["rmse_o2"] for name in perturbations),
    }


def summarize_safety_results(results: Dict[str, dict]) -> Dict[str, float]:
    perturbations = [name for name in results if name != "clean"]
    noise_tests = [name for name in perturbations if name.startswith("noise_")]

    return {
        "clean_critical_rate": results["clean"]["safety"]["viol_critical"],
        "clean_total_rate": results["clean"]["safety"]["viol_total"],
        "noise_critical_rate": _mean(results[name]["safety"]["viol_critical"] for name in noise_tests),
        "noise_total_rate": _mean(results[name]["safety"]["viol_total"] for name in noise_tests),
        "overall_critical_rate": _mean(results[name]["safety"]["viol_critical"] for name in perturbations),
        "overall_total_rate": _mean(results[name]["safety"]["viol_total"] for name in perturbations),
    }
