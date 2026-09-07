"""Run one validation-only EB-LNN CD/Langevin tuning trial."""

from __future__ import annotations

import argparse
import ast
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

from src.data_real import RealDataPipeline
from src.model import create_model
from src.sampler import build_sampler
from src.train import GenerativeTrainer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only EB-LNN CD/Langevin tuning trial"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("base_config.yaml")),
    )
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    use_wandb = WANDB_AVAILABLE and not args.no_wandb

    if use_wandb:
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            raise RuntimeError(
                "WANDB_API_KEY must be set in the operating-system environment. "
                "Use --no-wandb only for local debugging."
            )
        wandb.login(key=api_key, relogin=False)
        run = wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"].get("entity"),
            tags=config["wandb"].get("tags", []),
            config=config,
        )
        config.update(dict(wandb.config))
    else:
        run = None

    if isinstance(config.get("ebm_hidden_dims"), str):
        config["ebm_hidden_dims"] = ast.literal_eval(config["ebm_hidden_dims"])

    set_seed(config["seed"])
    device = choose_device(config.get("device", "auto"))
    data_dir = ROOT.parent / "dataset"
    output_dir = ROOT / "results" / "ebm_tuning" / (run.id if run else "local")
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = RealDataPipeline(
        real_csv=str(data_dir / config["real_csv"]),
        edge_csv=str(data_dir / config["edge_csv"]),
        scenarios=config.get("scenarios"),
        confidence_filter=config.get("confidence_filter"),
        edge_fraction=config["edge_fraction"],
        seq_len=config["seq_len"],
        stride=config["stride"],
        batch_size=config["batch_size"],
        test_size=config["test_size"],
        val_size=config["val_size"],
        seed=config["seed"],
    ).build()

    model = create_model(
        input_size=pipeline.input_size,
        hidden_size=config["hidden_size"],
        phys_output_size=pipeline.target_size,
        ebm_hidden_dims=config["ebm_hidden_dims"],
        device=device,
    )
    sampler = build_sampler(
        model=model,
        n_steps=config["n_steps"],
        step_size=config["step_size"],
        noise_scale=config["noise_scale"],
        clip_x=config["clip_x"],
    )
    trainer = GenerativeTrainer(
        model=model,
        sampler=sampler,
        config={
            **config,
            "input_size": pipeline.input_size,
        },
        device=device,
        use_wandb=use_wandb,
    )

    started_at = time.time()
    trainer.train(
        train_loader=pipeline.train_loader,
        val_loader=pipeline.val_loader,
        save_path=str(output_dir / "models"),
    )
    result = {
        "config": config,
        "device": device,
        "data_summary": pipeline.data_summary,
        "best_val_physics": trainer.best_val_loss,
        "best_epoch": trainer.best_epoch,
        "epochs_run": len(trainer.history["train_loss"]),
        "final_cd_gap": trainer.history["cd_gap"][-1],
        "wall_time_sec": round(time.time() - started_at, 1),
    }
    summary_path = output_dir / "trial.json"
    history_path = output_dir / "history.npz"
    with open(summary_path, "w") as handle:
        json.dump(result, handle, indent=2, default=str)
    np.savez(
        history_path,
        **{name: np.asarray(values) for name, values in trainer.history.items()},
    )

    if run:
        wandb.run.summary.update(result)
        artifact = wandb.Artifact(
            name=f"eblnn-ebm-tuning-{run.id}",
            type="ebm-tuning-trial",
            description="Validation-only summary and training history for one EBM/CD tuning trial.",
        )
        artifact.add_file(str(summary_path), name="trial.json")
        artifact.add_file(str(history_path), name="history.npz")
        wandb.log_artifact(artifact)
        wandb.finish()

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()