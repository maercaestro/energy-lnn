"""
run_lnn_ablation.py
===================
Run a single ablation experiment for the standalone LNN (pure CfC).

Reads ``lnn_ablation_config.yaml``, merges overrides onto baseline,
trains and evaluates the LNN model.

Usage
-----
    cd eblnn

    # Run baseline
    python experiments/run_lnn_ablation.py --experiment A2_baseline

    # Without W&B
    python experiments/run_lnn_ablation.py --experiment A1_real_only --no_wandb

    # Custom data directory
    python experiments/run_lnn_ablation.py --experiment B2_edge_25pct \\
        --data_dir /path/to/dataset

    # List available experiments
    python experiments/run_lnn_ablation.py --list
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

# ── Resolve imports from eblnn/ root ─────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lnn_model import create_lnn_model
from src.lnn_train import LNNTrainer
from src.data_real import RealDataPipeline

# Robustness evaluation (forward-pass only, runs after training)
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_robustness import run_robustness_tests

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =====================================================================
# Config helpers
# =====================================================================

def load_ablation_config(path: str | None = None) -> dict:
    if path is None:
        path = ROOT / "config" / "lnn_ablation_config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_experiment(raw_cfg: dict, experiment: str) -> dict:
    baseline = dict(raw_cfg["baseline"])
    experiments = raw_cfg.get("experiments", {})

    if experiment not in experiments:
        avail = ", ".join(sorted(experiments.keys()))
        raise KeyError(
            f"Unknown experiment '{experiment}'.  Available:\n  {avail}"
        )

    overrides = experiments[experiment] or {}
    description = overrides.pop("description", experiment)

    cfg = {**baseline, **overrides}
    cfg["experiment"] = experiment
    cfg["description"] = description
    cfg["data_files"] = raw_cfg["data_files"]
    return cfg


# =====================================================================
# Main
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one ablation experiment for standalone LNN"
    )
    p.add_argument(
        "--experiment", "-e", type=str, default="A2_baseline",
        help="Experiment key from lnn_ablation_config.yaml",
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Path to lnn_ablation_config.yaml",
    )
    p.add_argument(
        "--data_dir", type=str, default=None,
        help="Directory containing the CSV data files (default: ../dataset)",
    )
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to save results (default: results/lnn_ablation/<experiment>)",
    )
    p.add_argument("--no_wandb", action="store_true", help="Disable W&B")
    p.add_argument("--list", action="store_true", help="List experiments and exit")
    p.add_argument(
        "--seed", type=int, default=None,
        help="Override seed (for multi-seed runs)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────
    raw_cfg = load_ablation_config(args.config)

    if args.list:
        print("\n═══ Available LNN ablation experiments ═══\n")
        for name, info in raw_cfg["experiments"].items():
            desc = (info or {}).get("description", "")
            print(f"  {name:30s}  {desc}")
        print()
        return

    # ── Resolve experiment ───────────────────────────────────────────
    cfg = resolve_experiment(raw_cfg, args.experiment)

    # ── Seed override (for multi-seed runs) ──────────────────────────
    if args.seed is not None:
        cfg["seed"] = args.seed

    # ── Resolve data paths ───────────────────────────────────────────
    data_dir = Path(args.data_dir) if args.data_dir else (ROOT.parent / "dataset")
    real_csv = (
        str(data_dir / cfg["data_files"]["real"])
        if cfg.get("use_real", True)
        else None
    )
    edge_csv = (
        str(data_dir / cfg["data_files"]["edge"])
        if cfg.get("use_edge", True)
        else None
    )

    # ── Resolve output path ──────────────────────────────────────────
    seed_suffix = f"_seed{cfg['seed']}" if args.seed is not None else ""
    output_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "results" / "lnn_ablation" / (cfg["experiment"] + seed_suffix)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device

    # ── W&B ──────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb and WANDB_AVAILABLE
    run_name = cfg["experiment"] + (f"_s{cfg['seed']}" if args.seed is not None else "")
    if use_wandb:
        wandb.init(
            project="lnn-ablation",
            name=run_name,
            group="lnn-ablation",
            tags=["lnn-ablation", cfg["experiment"].split("_")[0]],
            config=cfg,
        )

    # ── Print config ─────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print(f"  LNN ABLATION EXPERIMENT: {cfg['experiment']}")
    print(f"  {cfg['description']}")
    print("═" * 65)
    skip_keys = {"data_files", "experiment", "description"}
    for k, v in sorted(cfg.items()):
        if k not in skip_keys:
            print(f"  {k:25s}: {v}")
    print(f"  {'device':25s}: {device}")
    print(f"  {'output_dir':25s}: {output_dir}")
    print("═" * 65 + "\n")

    # ── Set global seed ──────────────────────────────────────────────
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    t0 = time.time()

    # ── 1. Data pipeline ─────────────────────────────────────────────
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

    # ── 2. Model ─────────────────────────────────────────────────────
    model = create_lnn_model(
        input_size=pipeline.input_size,
        hidden_size=cfg["hidden_size"],
        phys_output_size=pipeline.target_size,
        device=device,
    )

    # ── 3. Trainer ───────────────────────────────────────────────────
    trainer_cfg = {
        "epochs": cfg["epochs"],
        "learning_rate": cfg["learning_rate"],
        "patience": cfg["patience"],
        "min_delta": cfg["min_delta"],
        "early_stopping": cfg.get("early_stopping", True),
    }

    trainer = LNNTrainer(
        model=model,
        config=trainer_cfg,
        device=device,
        use_wandb=use_wandb,
    )

    # ── 4. Train ─────────────────────────────────────────────────────
    model_dir = str(output_dir / "models")
    trainer.train(
        train_loader=pipeline.train_loader,
        val_loader=pipeline.val_loader,
        save_path=model_dir,
    )

    # ── 5. Evaluate ──────────────────────────────────────────────────
    test_metrics = trainer.evaluate(
        test_loader=pipeline.test_loader,
        target_scaler=pipeline.target_scaler,
    )

    # ── 6. Robustness / OOD tests ────────────────────────────────────
    print("\nRunning robustness tests (OOD perturbations) ...")
    robustness = run_robustness_tests(model, pipeline, device)
    print(f"  {len(robustness) - 1} perturbation tests completed.")

    # Log robustness to W&B (single call + summary for table/bar charts)
    if use_wandb:
        rob_flat = {}
        for test_name, entry in robustness.items():
            if test_name == "clean":
                # Log clean baseline metrics
                for k, v in entry.get("metrics", {}).items():
                    rob_flat[f"robustness/clean/{k}"] = v
                continue
            for k, v in entry.get("degradation", {}).items():
                rob_flat[f"robustness/{test_name}/{k}"] = v
            for k, v in entry.get("metrics", {}).items():
                rob_flat[f"robustness/{test_name}/{k}"] = v
        wandb.log(rob_flat)
        wandb.run.summary.update(rob_flat)

    elapsed = time.time() - t0

    # ── 7. Collect & save results ────────────────────────────────────
    results = {
        "experiment": cfg["experiment"],
        "description": cfg["description"],
        "config": {k: v for k, v in cfg.items() if k != "data_files"},
        "data_summary": pipeline.data_summary,
        "test_metrics": test_metrics,
        "training": {
            "best_val_loss": trainer.best_val_loss,
            "best_epoch": trainer.best_epoch,
            "final_train_loss": (
                trainer.history["train_loss"][-1]
                if trainer.history["train_loss"]
                else None
            ),
            "final_val_loss": (
                trainer.history["val_loss"][-1]
                if trainer.history["val_loss"]
                else None
            ),
            "epochs_run": len(trainer.history["train_loss"]),
        },
        "wall_time_sec": round(elapsed, 1),
        "device": device,
        "robustness": robustness,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅  Results saved to {results_path}")

    history_path = output_dir / "history.npz"
    np.savez(
        history_path,
        **{k: np.array(v) for k, v in trainer.history.items()},
    )
    print(f"📊  Training history saved to {history_path}")

    # ── 8. Close W&B ─────────────────────────────────────────────────
    if use_wandb:
        wandb.log({"wall_time_sec": elapsed})
        wandb.run.summary.update(test_metrics)
        wandb.finish()

    print(f"\n⏱  Total wall time: {elapsed:.0f}s")
    print("Experiment complete.\n")


if __name__ == "__main__":
    main()
