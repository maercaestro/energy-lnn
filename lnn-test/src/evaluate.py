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
