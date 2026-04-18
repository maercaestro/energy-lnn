from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import RealDataPipeline
from src.evaluate import (
    compute_regression_metrics,
    run_disturbance_evaluation,
    run_extreme_evaluation,
    run_safety_evaluation,
    summarize_disturbance_results,
    summarize_extreme_results,
    summarize_safety_results,
)
from src.models import create_model
from src.trainer import BenchmarkTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the lnn-test benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config" / "benchmark_config.yaml"),
        help="Path to benchmark YAML config.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=["lnn", "lstm", "mlp"],
        help="Override model list from config.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override seed list from config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override benchmark output directory.",
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def choose_device(config_device: str) -> str:
    if config_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if config_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return config_device


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    run_name = config.get("run_name", "default")
    models = args.models or config["experiment"]["models"]
    seeds = args.seeds or config["experiment"]["seeds"]
    device = choose_device(config["experiment"].get("device", "auto"))

    output_dir = Path(args.output_dir) if args.output_dir else (ROOT / "results" / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    train_cfg = config["training"]

    real_csv = resolve_path(ROOT, data_cfg.get("real_csv"))
    edge_csv = resolve_path(ROOT, data_cfg.get("edge_csv"))

    all_saved_runs = []

    for model_name in models:
        model_cfg = config["models"][model_name]
        for seed in seeds:
            set_seed(seed)
            start_time = time.time()

            pipeline = RealDataPipeline(
                real_csv=real_csv,
                edge_csv=edge_csv,
                scenarios=data_cfg.get("scenarios"),
                confidence_filter=data_cfg.get("confidence_filter"),
                edge_fraction=data_cfg.get("edge_fraction", 1.0),
                seq_len=data_cfg.get("seq_len", 30),
                stride=data_cfg.get("stride"),
                batch_size=data_cfg.get("batch_size", 64),
                test_size=data_cfg.get("test_size", 0.2),
                val_size=data_cfg.get("val_size", 0.1),
                seed=seed,
            ).build()

            model = create_model(
                model_name=model_name,
                input_size=pipeline.input_size,
                target_size=pipeline.target_size,
                model_config=model_cfg,
                device=device,
            )
            param_count = sum(p.numel() for p in model.parameters())

            trainer = BenchmarkTrainer(model=model, config=train_cfg, device=device)

            model_dir = output_dir / model_name / f"seed_{seed}" / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            trainer.train(
                train_loader=pipeline.train_loader,
                val_loader=pipeline.val_loader,
                save_path=str(model_dir),
            )
            trainer.load_best_model()

            y_true, y_pred = trainer.predict(pipeline.test_loader)
            y_true = pipeline.target_scaler.inverse_transform(y_true)
            y_pred = pipeline.target_scaler.inverse_transform(y_pred)
            test_metrics = compute_regression_metrics(y_true, y_pred)

            # --- Inference latency measurement ---
            model.eval()
            n_batches = len(pipeline.test_loader)
            # warm-up
            with torch.no_grad():
                for x_batch, _ in pipeline.test_loader:
                    model(x_batch.to(device))
                    break
            # timed pass
            t0 = time.time()
            n_samples_inf = 0
            with torch.no_grad():
                for x_batch, _ in pipeline.test_loader:
                    model(x_batch.to(device))
                    n_samples_inf += x_batch.shape[0]
            inference_wall = time.time() - t0
            latency_per_sample_ms = (inference_wall / n_samples_inf) * 1000 if n_samples_inf > 0 else float("nan")
            throughput_samples_per_sec = n_samples_inf / inference_wall if inference_wall > 0 else float("nan")

            disturbance = run_disturbance_evaluation(
                model=trainer.model,
                loader=pipeline.test_loader,
                device=device,
                target_scaler=pipeline.target_scaler,
            )
            safety = run_safety_evaluation(
                model=trainer.model,
                loader=pipeline.test_loader,
                device=device,
                target_scaler=pipeline.target_scaler,
            )
            disturbance_summary = summarize_disturbance_results(disturbance)
            safety_summary = summarize_safety_results(safety)

            extreme = run_extreme_evaluation(
                model=trainer.model,
                loader=pipeline.test_loader,
                device=device,
                target_scaler=pipeline.target_scaler,
            )
            extreme_summary = summarize_extreme_results(extreme)

            result = {
                "model": model_name,
                "seed": seed,
                "device": device,
                "param_count": param_count,
                "config": {
                    "data": data_cfg,
                    "training": train_cfg,
                    "model": model_cfg,
                },
                "data_summary": pipeline.data_summary,
                "training": {
                    "best_val_loss": trainer.best_val_loss,
                    "best_epoch": trainer.best_epoch,
                    "epochs_run": len(trainer.history["train_loss"]),
                    "final_train_loss": trainer.history["train_loss"][-1],
                    "final_val_loss": trainer.history["val_loss"][-1],
                },
                "inference": {
                    "latency_per_sample_ms": round(latency_per_sample_ms, 4),
                    "throughput_samples_per_sec": round(throughput_samples_per_sec, 1),
                    "n_batches": n_batches,
                    "n_samples": n_samples_inf,
                },
                "test_metrics": test_metrics,
                "disturbance": disturbance,
                "disturbance_summary": disturbance_summary,
                "safety": safety,
                "safety_summary": safety_summary,
                "extreme": extreme,
                "extreme_summary": extreme_summary,
                "wall_time_sec": round(time.time() - start_time, 2),
            }

            seed_dir = output_dir / model_name / f"seed_{seed}"
            with open(seed_dir / "results.json", "w") as handle:
                json.dump(result, handle, indent=2)
            np.savez(
                seed_dir / "history.npz",
                train_loss=np.array(trainer.history["train_loss"]),
                val_loss=np.array(trainer.history["val_loss"]),
            )

            all_saved_runs.append({
                "model": model_name,
                "seed": seed,
                "results_path": str(seed_dir / "results.json"),
            })

            print(
                f"{model_name.upper()} seed={seed} | "
                f"temp_rmse={test_metrics['rmse_temp']:.4f} | "
                f"o2_rmse={test_metrics['rmse_o2']:.4f} | "
                f"noise_deg_temp={disturbance_summary['noise_deg_rmse_temp']:.4f} | "
                f"clean_critical={safety_summary['clean_critical_rate']:.4f}"
            )

    with open(output_dir / "run_manifest.json", "w") as handle:
        json.dump(
            {
                "run_name": run_name,
                "models": models,
                "seeds": seeds,
                "device": device,
                "runs": all_saved_runs,
            },
            handle,
            indent=2,
        )

    print(f"\nBenchmark outputs saved to {output_dir}")
    print("Run compare_results.py to generate the final report.")


if __name__ == "__main__":
    main()