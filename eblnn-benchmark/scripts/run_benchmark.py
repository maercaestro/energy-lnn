"""
End-to-end runner for the EB-LNN vs LNN vs LSTM vs PID benchmark.

Usage:
    python eblnn-benchmark/scripts/run_benchmark.py
    python eblnn-benchmark/scripts/run_benchmark.py --models eblnn lstm pid --seeds 42
    python eblnn-benchmark/scripts/run_benchmark.py --no-wandb
    python eblnn-benchmark/scripts/run_benchmark.py --force         # rerun even if results exist

Designed to be safe to interrupt (Ctrl-C, SSH drop) and resume:
any (model, seed) combination whose results.json already exists is
skipped unless --force is passed.
"""
from __future__ import annotations

import argparse
import json
import os
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
from src.trainers import create_trainer, PIDTrainer

# wandb is optional. The runner still works (silently) if it is not installed.
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EB-LNN benchmark")
    parser.add_argument("--config", type=str,
                        default=str(ROOT / "config" / "benchmark_config.yaml"))
    parser.add_argument("--models", nargs="+", default=None,
                        choices=["eblnn", "lnn", "lstm", "pid"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging even if enabled in config.")
    parser.add_argument("--force", action="store_true",
                        help="Rerun (model, seed) combinations whose results.json already exists.")
    return parser.parse_args()


def resolve_path(root: Path, value: str | None) -> str | None:
    if value is None:
        return None
    p = Path(value)
    return str(p) if p.is_absolute() else str((root / p).resolve())


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

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else (ROOT / "results" / run_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    train_cfg = config["training"]
    eblnn_cfg = config.get("eblnn", {})
    pid_cfg = config.get("pid", {})

    wandb_cfg = config.get("wandb", {}) or {}
    use_wandb = (
        bool(wandb_cfg.get("enabled", False))
        and not args.no_wandb
        and _WANDB_AVAILABLE
    )
    if wandb_cfg.get("enabled") and not _WANDB_AVAILABLE and not args.no_wandb:
        print("[warn] wandb enabled in config but the package is not installed; "
              "continuing without wandb.")

    real_csv = resolve_path(ROOT, data_cfg.get("real_csv"))
    edge_csv = resolve_path(ROOT, data_cfg.get("edge_csv"))

    all_saved_runs = []

    for model_name in models:
        model_cfg = config["models"][model_name]
        for seed in seeds:
            seed_dir = output_dir / model_name / f"seed_{seed}"
            results_path = seed_dir / "results.json"
            if results_path.exists() and not args.force:
                print(f"[skip] {model_name} seed={seed} already done -> {results_path}")
                all_saved_runs.append({
                    "model": model_name, "seed": seed,
                    "results_path": str(results_path), "skipped": True,
                })
                continue

            set_seed(seed)
            start_time = time.time()

            # ---- wandb run (one per (model, seed)) ----
            wb_run = None
            if use_wandb:
                wb_settings = wandb.Settings(
                    console="wrap",          # capture stdout/stderr into wandb Logs
                    _disable_stats=False,
                )
                wb_run = wandb.init(
                    project=wandb_cfg.get("project", "eblnn-benchmark"),
                    entity=wandb_cfg.get("entity"),
                    mode=wandb_cfg.get("mode", "online"),
                    group=wandb_cfg.get("group") or run_name,
                    job_type=model_name,
                    name=f"{run_name}-{model_name}-seed{seed}",
                    tags=wandb_cfg.get("tags"),
                    reinit=True,
                    settings=wb_settings,
                    config={
                        "run_name": run_name,
                        "model": model_name,
                        "seed": seed,
                        "device": device,
                        "data": data_cfg,
                        "training": train_cfg,
                        "model_cfg": model_cfg,
                        "eblnn": eblnn_cfg if model_name == "eblnn" else None,
                        "pid": pid_cfg if model_name == "pid" else None,
                    },
                )

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
                input_scaler=pipeline.input_scaler,
                target_scaler=pipeline.target_scaler,
            )
            param_count = sum(p.numel() for p in model.parameters())

            trainer = create_trainer(
                model_name=model_name,
                model=model,
                train_cfg=train_cfg,
                device=device,
                eblnn_cfg=eblnn_cfg,
                pid_cfg=pid_cfg,
                seq_len=data_cfg.get("seq_len", 30),
                input_size=pipeline.input_size,
            )

            model_dir = output_dir / model_name / f"seed_{seed}" / "models"
            model_dir.mkdir(parents=True, exist_ok=True)

            def _epoch_cb(epoch: int, metrics: dict, _wb=wb_run, _m=model_name, _s=seed):
                if _wb is None:
                    return
                _wb.log({f"{_m}/{k}": v for k, v in metrics.items()}, step=epoch)

            log_prefix = f"[{model_name} seed={seed}] "
            print(
                f"\n>>> training {model_name} seed={seed} "
                f"(device={device}, params={param_count})",
                flush=True,
            )

            trainer.train(
                train_loader=pipeline.train_loader,
                val_loader=pipeline.val_loader,
                save_path=str(model_dir),
                epoch_callback=_epoch_cb,
                log_prefix=log_prefix,
                verbose=True,
            )
            trainer.load_best_model()

            y_true, y_pred = trainer.predict(pipeline.test_loader)
            y_true = pipeline.target_scaler.inverse_transform(y_true)
            y_pred = pipeline.target_scaler.inverse_transform(y_pred)
            test_metrics = compute_regression_metrics(y_true, y_pred)

            # ---- Inference latency ----
            model.eval()
            n_batches = len(pipeline.test_loader)
            with torch.no_grad():
                for x_batch, _ in pipeline.test_loader:
                    model(x_batch.to(device))
                    break
            t0 = time.time()
            n_samples_inf = 0
            with torch.no_grad():
                for x_batch, _ in pipeline.test_loader:
                    model(x_batch.to(device))
                    n_samples_inf += x_batch.shape[0]
            inference_wall = time.time() - t0
            latency_per_sample_ms = (
                (inference_wall / n_samples_inf) * 1000
                if n_samples_inf > 0 else float("nan")
            )
            throughput_samples_per_sec = (
                n_samples_inf / inference_wall
                if inference_wall > 0 else float("nan")
            )

            disturbance = run_disturbance_evaluation(
                model=trainer.model, loader=pipeline.test_loader,
                device=device, target_scaler=pipeline.target_scaler,
            )
            safety = run_safety_evaluation(
                model=trainer.model, loader=pipeline.test_loader,
                device=device, target_scaler=pipeline.target_scaler,
            )
            disturbance_summary = summarize_disturbance_results(disturbance)
            safety_summary = summarize_safety_results(safety)

            extreme = run_extreme_evaluation(
                model=trainer.model, loader=pipeline.test_loader,
                device=device, target_scaler=pipeline.target_scaler,
            )
            extreme_summary = summarize_extreme_results(extreme)

            # ---- Extra: PID-specific metadata ----
            extra: dict = {}
            if isinstance(trainer, PIDTrainer) and trainer.best_gains is not None:
                g = trainer.best_gains
                extra["pid_gains"] = {
                    "sp_temp": g.sp_temp, "sp_o2": g.sp_o2, "afr_sp": g.afr_sp,
                    "kp_temp": g.kp_temp, "ki_temp": g.ki_temp, "kd_temp": g.kd_temp,
                    "kp_o2": g.kp_o2,
                }

            result = {
                "model": model_name,
                "seed": seed,
                "device": device,
                "param_count": param_count,
                "config": {
                    "data": data_cfg,
                    "training": train_cfg,
                    "model": model_cfg,
                    "eblnn": eblnn_cfg if model_name == "eblnn" else None,
                    "pid": pid_cfg if model_name == "pid" else None,
                },
                "data_summary": pipeline.data_summary,
                "training": {
                    "best_val_loss": trainer.best_val_loss,
                    "best_epoch": trainer.best_epoch,
                    "epochs_run": len(trainer.history["train_loss"]),
                    "final_train_loss": (
                        trainer.history["train_loss"][-1]
                        if trainer.history["train_loss"] else float("nan")
                    ),
                    "final_val_loss": (
                        trainer.history["val_loss"][-1]
                        if trainer.history["val_loss"] else float("nan")
                    ),
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
                **extra,
            }

            seed_dir = output_dir / model_name / f"seed_{seed}"
            with open(seed_dir / "results.json", "w") as handle:
                json.dump(result, handle, indent=2)
            np.savez(
                seed_dir / "history.npz",
                train_loss=np.array(trainer.history["train_loss"]),
                val_loss=np.array(trainer.history["val_loss"]),
            )

            # ---- log final summary metrics + finish wandb ----
            if wb_run is not None:
                wb_summary = {
                    f"summary/{model_name}/temp_rmse": test_metrics["rmse_temp"],
                    f"summary/{model_name}/o2_rmse":   test_metrics["rmse_o2"],
                    f"summary/{model_name}/temp_mae":  test_metrics["mae_temp"],
                    f"summary/{model_name}/o2_mae":    test_metrics["mae_o2"],
                    f"summary/{model_name}/temp_r2":   test_metrics["r2_temp"],
                    f"summary/{model_name}/o2_r2":     test_metrics["r2_o2"],
                    f"summary/{model_name}/noise_deg_rmse_temp": disturbance_summary["noise_deg_rmse_temp"],
                    f"summary/{model_name}/clean_critical_rate": safety_summary["clean_critical_rate"],
                    f"summary/{model_name}/latency_ms": result["inference"]["latency_per_sample_ms"],
                    f"summary/{model_name}/throughput": result["inference"]["throughput_samples_per_sec"],
                    f"summary/{model_name}/wall_time_sec": result["wall_time_sec"],
                    f"summary/{model_name}/best_val_loss": trainer.best_val_loss,
                    f"summary/{model_name}/best_epoch":   trainer.best_epoch,
                    f"summary/{model_name}/param_count":  param_count,
                }
                for k, v in wb_summary.items():
                    wb_run.summary[k] = v
                # Upload the artifacts as a wandb artifact for archival.
                try:
                    art = wandb.Artifact(
                        name=f"{run_name}-{model_name}-seed{seed}",
                        type="benchmark-results",
                    )
                    art.add_file(str(seed_dir / "results.json"))
                    art.add_file(str(seed_dir / "history.npz"))
                    wb_run.log_artifact(art)
                except Exception as exc:
                    print(f"[warn] could not log wandb artifact: {exc}")
                wb_run.finish()

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
                "run_name": run_name, "models": models, "seeds": seeds,
                "device": device, "runs": all_saved_runs,
            },
            handle, indent=2,
        )

    print(f"\nBenchmark outputs saved to {output_dir}")
    print("Run compare_results.py to generate the final report.")


if __name__ == "__main__":
    main()
