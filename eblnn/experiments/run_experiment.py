"""
run_experiment.py
-----------------
Entry-point for a single Generative EB-LNN experiment.

Usage
-----

# Run with default config
$ python experiments/run_experiment.py

# Override specific keys
$ python experiments/run_experiment.py \
      --alpha 2.0 \
      --hidden_size 256 \
      --n_steps 30 \
      --no_wandb

# Run as part of a W&B sweep
$ wandb sweep config/sweep_config.yaml
$ wandb agent <sweep_id>
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

import torch
import yaml

# --- Resolve imports relative to eblnn/ root ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import create_model
from src.sampler import build_sampler
from src.data import DataPipeline
from src.train import GenerativeTrainer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str = None) -> dict:
    """Load YAML config and flatten nested keys for easy access."""
    if path is None:
        path = ROOT / "config" / "base_config.yaml"

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Flatten nested dicts with a single-level view (nested access still works)
    cfg: dict = {}

    # data
    cfg.update(raw.get("data", {}))
    # model
    cfg.update(raw.get("model", {}))
    # training
    cfg.update(raw.get("training", {}))
    # sampler → prefix to avoid collision
    for k, v in raw.get("sampler", {}).items():
        cfg[k] = v
    # buffer
    cfg.update(raw.get("buffer", {}))

    cfg["device"] = raw.get("device", "cpu")
    cfg["wandb"] = raw.get("wandb", {})
    cfg["paths"] = raw.get("paths", {})

    return cfg


# ---------------------------------------------------------------------------
# Argument parsing (W&B sweep overrides come via wandb.config)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generative EB-LNN experiment")
    p.add_argument("--config", default=None)
    p.add_argument("--alpha", type=float)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--hidden_size", type=int)
    p.add_argument("--ebm_hidden_dims", type=str,
                   help='Python list literal, e.g. "[128,64]"')
    p.add_argument("--n_steps", type=int)
    p.add_argument("--step_size", type=float)
    p.add_argument("--noise_scale", type=float)
    p.add_argument("--l2_reg", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 1. Load base config
    cfg = load_config(args.config)

    # 2. Command-line overrides
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    overrides.pop("config", None)
    overrides.pop("no_wandb", None)
    cfg.update(overrides)

    # 3. Parse ebm_hidden_dims if given as string
    if isinstance(cfg.get("ebm_hidden_dims"), str):
        cfg["ebm_hidden_dims"] = ast.literal_eval(cfg["ebm_hidden_dims"])

    # 4. Device
    device_req = cfg.get("device", "cpu")
    device = device_req if torch.cuda.is_available() else "cpu"
    if device != device_req:
        print(f"[warn] requested {device_req}, using cpu (CUDA not available)")
    cfg["device"] = device

    # 5. W&B initialisation
    use_wandb = not args.no_wandb and WANDB_AVAILABLE
    if use_wandb:
        wb_cfg = cfg.get("wandb", {})
        run = wandb.init(
            project=wb_cfg.get("project", "energy-based-lnn"),
            entity=wb_cfg.get("entity") or None,
            tags=wb_cfg.get("tags", []),
            notes=wb_cfg.get("notes", ""),
            config=cfg,
        )
        # Sweep may override config keys
        cfg.update(dict(wandb.config))
        if isinstance(cfg.get("ebm_hidden_dims"), str):
            cfg["ebm_hidden_dims"] = ast.literal_eval(cfg["ebm_hidden_dims"])

    print("\n=== Generative EB-LNN Experiment ===")
    for k, v in cfg.items():
        if k not in ("wandb", "paths"):
            print(f"  {k}: {v}")

    # 6. Data pipeline
    paths = cfg.get("paths", {})
    data_dir = ROOT / paths.get("results", "results")
    data_dir.mkdir(parents=True, exist_ok=True)

    pipeline = DataPipeline(
        data_path=str(ROOT / cfg["data_path"]),
        num_sequences=cfg["num_sequences"],
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
        test_size=cfg["test_size"],
        val_size=cfg["val_size"],
        seed=cfg["seed"],
        force_regenerate=cfg["force_regenerate"],
    ).build()

    # 7. Model
    model = create_model(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        phys_output_size=cfg["phys_output_size"],
        ebm_hidden_dims=cfg["ebm_hidden_dims"],
        device=device,
    )

    # 8. Sampler
    sampler = build_sampler(
        model=model,
        n_steps=cfg["n_steps"],
        step_size=cfg["step_size"],
        noise_scale=cfg["noise_scale"],
        clip_x=cfg.get("clip_x", 3.0),
    )

    # 9. Trainer
    trainer = GenerativeTrainer(
        model=model,
        sampler=sampler,
        config=cfg,
        device=device,
        use_wandb=use_wandb,
    )

    # 10. Train
    model_dir = str(ROOT / paths.get("models", "results/models"))
    trainer.train(
        train_loader=pipeline.train_loader,
        val_loader=pipeline.val_loader,
        save_path=model_dir,
    )

    # 11. Evaluate
    trainer.evaluate(
        test_loader=pipeline.test_loader,
        target_scaler=pipeline.target_scaler,
    )

    # 12. Close W&B
    if use_wandb:
        wandb.finish()

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
