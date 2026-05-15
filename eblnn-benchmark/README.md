# eblnn-benchmark

Benchmark folder for the final research comparison:

> **Energy-Based Liquid Neural Network (EB-LNN)** vs **Liquid NN (CfC)** vs **LSTM** vs a tuned **PID baseline**

on the augmented industrial-furnace dataset (real data + PINN-generated edge cases).

It mirrors the structure of [`lnn-test/`](../lnn-test/README.md) — same data
pipeline, same disturbance/safety/extreme tests, same composite ranking — but
adds the EB-LNN model (Contrastive-Divergence trained) and a PID baseline.

## What is being compared

| Model | Description | Trainer |
| :--- | :--- | :--- |
| **EB-LNN** | CfC backbone + physics head + EBM head; learns an energy manifold via Contrastive Divergence with Langevin-sampled negatives. | `CDTrainer` (joint physics MSE + CD loss) |
| **LNN**  | Plain CfC backbone + linear physics head. | `SupervisedTrainer` (MSE) |
| **LSTM** | 2-layer LSTM + linear physics head. | `SupervisedTrainer` (MSE) |
| **PID**  | Controller-style next-state predictor: `next_temp = current_temp + Kp·e + Ki·∫e + Kd·de`, with an analogous proportional loop on excess O₂ driven by AFR deviation. | `PIDTrainer` (validation grid search) |

The PID baseline is intentionally a *naive* baseline: it operates in physical
units and exposes the gap between a tuned classical controller and the
learning models on the same next-step prediction task.

## Dataset

Same as `lnn-test/`:
- real data: `../dataset/real_furnace_eblnn.csv`
- PINN edge cases: `../dataset/edge_cases_v2_eblnn.csv`

## Quick start

```bash
pip install -r eblnn-benchmark/requirements.txt

# Full benchmark (all four models, three seeds)
python eblnn-benchmark/scripts/run_benchmark.py

# Single seed of EB-LNN vs LSTM vs PID for a quick check
python eblnn-benchmark/scripts/run_benchmark.py \
    --models eblnn lstm pid --seeds 42

# Aggregate to comparison_report.md and comparison_summary.json
python eblnn-benchmark/scripts/compare_results.py

# Publication figures
python eblnn-benchmark/scripts/plot_results.py
```

## Outputs

For each seed of each model:
- `results/<run>/<model>/seed_<seed>/results.json`
- `results/<run>/<model>/seed_<seed>/history.npz`
- `results/<run>/<model>/seed_<seed>/models/best_model.pth`

Aggregated:
- `results/<run>/comparison_summary.json`
- `results/<run>/comparison_report.md`
- `results/<run>/plots/fig1..fig7_*.png`

For PID runs, `results.json` additionally carries the chosen `pid_gains`.

## Running on a VM (tmux + wandb)

The full sweep is designed for an unattended weekend run on a remote VM.

```bash
# On the VM:
ssh user@vm
git pull
python -m venv venv && source venv/bin/activate
pip install -r eblnn-benchmark/requirements.txt

# Open a tmux session that survives SSH disconnects.
tmux new -s eblnn

# (Optional) point at a non-default Python.
export EBLNN_PYTHON=$(which python)

# wandb authentication — pick ONE:
export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxx     # easiest
# or:  wandb login                                  # writes ~/.netrc
# or:  export NO_WANDB=1                            # disable entirely
# or:  set wandb.mode=offline in benchmark_config.yaml and sync later

bash eblnn-benchmark/scripts/run_vm.sh
# detach with Ctrl-b d ; reattach with `tmux attach -t eblnn`
```

`run_vm.sh`:
- Mirrors stdout/stderr to `results/<run>/logs/run-<timestamp>.log`.
- Runs the full sweep, then `compare_results.py`, then `plot_results.py`.
- **Resumable** — interrupt or disconnect, relaunch, and any `(model, seed)`
  whose `results.json` already exists is skipped. Pass `FORCE=1` to override.

Environment variables understood by `run_vm.sh`:

| Var | Default | Purpose |
| :-- | :-- | :-- |
| `EBLNN_PYTHON` | `python` | Python interpreter to use. |
| `RUN_NAME` | `default` | Run name → results live under `results/$RUN_NAME/`. |
| `CONFIG_PATH` | `eblnn-benchmark/config/benchmark_config.yaml` | Override config. |
| `WANDB_API_KEY` | (unset) | wandb auth. |
| `NO_WANDB` | `0` | Set to `1` to disable wandb logging. |
| `FORCE` | `0` | Set to `1` to ignore existing `results.json` and rerun. |

### What gets logged to wandb

One run per `(model, seed)`, grouped under `RUN_NAME`:

- **Per-epoch** metrics (`train_loss`, `val_loss`, plus EB-LNN diagnostics
  `train_phys`, `val_phys`, `train_cd`, `val_cd`, `e_pos`, `e_neg`, `cd_gap`).
- **Summary** scalars (test RMSE/MAE/R² for temp & O₂, noise-disturbance
  degradation, clean safety-violation rate, latency, throughput, wall time,
  parameter count).
- **Artifacts**: `results.json` and `history.npz` per seed, archived in wandb.

## Notes

- The EB-LNN core model and its loss/sampler implementations are imported from
  the [`eblnn`](../eblnn/) package; this folder only adds the benchmark
  wrapper, the PID baseline, and the comparison scripts.
- Best-checkpoint criterion for EB-LNN is the *physics* validation MSE, not
  the joint loss (the CD term can legitimately be negative).
- Validation negatives during EB-LNN training are fresh Gaussian noise (no
  Langevin) for speed.
- wandb is optional: the runner falls back to silent operation if `wandb`
  is not installed or `--no-wandb` is passed.
