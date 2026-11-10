"""Evaluate a trained legacy EB-LNN across real and COP-informed scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_real import INPUT_COLS, TARGET_COLS  # noqa: E402
from src.model import create_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "dataset")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def block_sequences(values: np.ndarray, seq_len: int) -> np.ndarray:
    count = len(values) // seq_len
    return values[: count * seq_len].reshape(count, seq_len, values.shape[1])


@torch.no_grad()
def score_group(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    input_scaler,
    target_scaler,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_scaled = input_scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    energies = []
    predictions = []
    for start in range(0, len(x_scaled), batch_size):
        batch = torch.tensor(
            x_scaled[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        pred, _, _ = model(batch)
        energies.append(model.energy(batch).cpu().numpy())
        predictions.append(pred.cpu().numpy())
    pred_scaled = np.concatenate(predictions)
    pred_raw = target_scaler.inverse_transform(
        pred_scaled.reshape(-1, pred_scaled.shape[-1])
    ).reshape(pred_scaled.shape)
    return np.concatenate(energies), pred_raw, y


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    run_dir = args.run_dir.resolve()
    with (run_dir / "scalers.pkl").open("rb") as handle:
        scalers = pickle.load(handle)
    input_scaler = scalers["input_scaler"]
    target_scaler = scalers["target_scaler"]

    model = create_model(
        input_size=len(INPUT_COLS),
        hidden_size=config["hidden_size"],
        phys_output_size=len(TARGET_COLS),
        ebm_hidden_dims=config["ebm_hidden_dims"],
        device=args.device,
    )
    model.load_state_dict(
        torch.load(
            run_dir / "models" / "best_model.pth",
            map_location=args.device,
            weights_only=True,
        )
    )
    model.eval()

    real = pd.read_csv(args.data_dir / config["real_csv"])
    edge = pd.read_csv(args.data_dir / config["edge_csv"])
    groups = {"real": real}
    groups.update({
        scenario: frame.reset_index(drop=True)
        for scenario, frame in edge.groupby("scenario", sort=True)
    })

    raw = {}
    rows = []
    for name, frame in groups.items():
        x = block_sequences(
            frame[INPUT_COLS].to_numpy(dtype=np.float32), config["seq_len"]
        )
        y = block_sequences(
            frame[TARGET_COLS].to_numpy(dtype=np.float32), config["seq_len"]
        )
        energy, pred, target = score_group(
            model,
            x,
            y,
            input_scaler,
            target_scaler,
            args.device,
            config["batch_size"],
        )
        raw[name] = energy
        rows.append({
            "group": name,
            "n_sequences": len(energy),
            "energy_mean": float(np.mean(energy)),
            "energy_sample_sd": (
                float(np.std(energy, ddof=1)) if len(energy) > 1 else None
            ),
            "temperature_rmse": float(np.sqrt(np.mean(
                (pred[..., 0] - target[..., 0]) ** 2
            ))),
            "oxygen_rmse": float(np.sqrt(np.mean(
                (pred[..., 1] - target[..., 1]) ** 2
            ))),
        })

    real_energy = raw["real"]
    comparisons = {}
    for name, scenario_energy in raw.items():
        if name == "real":
            continue
        labels = np.concatenate([
            np.zeros(len(real_energy), dtype=np.int8),
            np.ones(len(scenario_energy), dtype=np.int8),
        ])
        scores = np.concatenate([real_energy, scenario_energy])
        comparisons[name] = {
            "auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "fraction_above_real_99th_percentile": float(np.mean(
                scenario_energy > np.quantile(real_energy, 0.99, method="higher")
            )),
        }

    with (run_dir / "scenario_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(run_dir / "scenario_energies.npz", **raw)
    summary = {
        "energy_direction": "Higher energy treated as more abnormal.",
        "groups": rows,
        "scenario_vs_real": comparisons,
    }
    (run_dir / "scenario_evaluation.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
