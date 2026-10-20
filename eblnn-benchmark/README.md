# eblnn-benchmark

## Current protocol: real-data forecasting v1

**The October 19 target audit found that the legacy export copied the same
`OutletT` into `current_temp` and `next_temp`. Its O2 target was also unshifted.
Do not interpret the old `ebm-tuned-f6ozllp7` results as future-state forecasts.**
The old data, checkpoints and configuration are preserved for provenance.

The default is now [`forecast_config.yaml`](config/forecast_config.yaml),
with run name `forecast-real-v1`:

- Real source: `../dataset/cleaned_furnace_data.csv`. No synthetic augmentation.
- Nine measured inputs: fuel flow, draft pressure, current outlet temperature,
  inlet temperature, inlet flow, current O2, damper position, fuel pressure and
  bridgewall temperature. Values retain their source units before scaling.
- Targets: `OutletT` and `ExcessO2` **one recorded timestep ahead**, at each
  timestep of a causal 30-step sequence. The horizon is configured explicitly.
- `Date` is an integer sample ID, not a wall-clock timestamp. Consecutive
  IDs are required; this follows the source's nominal one-minute sampling.
- First 80% of rows train, next 10% validation, last 10% test. Split before
  windowing; no inputs or targets cross splits or gaps. All seeds use the
  same periods. Scalers fit on training windows only.
- Perturbation RNGs reset per evaluation suite and seed, so each model sees
  the same stochastic disturbance realizations rather than training-dependent
  random draws.
- EB-LNN, CfC/LNN and LSTM are compared against **persistence**: predict
  future temperature/O2 as their current measured values. Legacy PID needs
  a derived AFR feature and is intentionally excluded from this protocol.
- No PINN-derived inputs: the pre-existing PINN was prepared before these
  split boundaries. Synthetic LHS samples are operating points, not temporal
  rollouts; shifting their rows would not produce valid forecasting targets.
- EBM settings are starting values transferred from the legacy experiment,
  **not validated forecasting hyperparameters**. Do not select settings using
  test metrics. Old checkpoints are incompatible with the nine-input model.

Source cleaning is inherited from the existing cleaned CSV. Outlier thresholds
were calculated globally, before these splits. The cleaner also supports linear
interpolation, but the saved cleaning report lists no interpolated-cell counts;
interpolation leakage has not been demonstrated. Chronological splitting here
does not retroactively make upstream preprocessing train-only; a stricter
deployment study should also version and audit the cleaning process.

Run a smoke test before starting a long VM job:

```bash
python eblnn-benchmark/scripts/run_benchmark.py --smoke-test
# When ready, all four models and three seeds:
python eblnn-benchmark/scripts/run_benchmark.py
# Or launch in tmux using the existing VM launcher:
bash eblnn-benchmark/scripts/run_vm.sh

python eblnn-benchmark/scripts/compare_results.py
python eblnn-benchmark/scripts/plot_results.py
python -m unittest discover -s eblnn-benchmark/tests -v
```

Smoke mode evaluates validation subsets only, disables W&B and uses a separate
`forecast-real-v1-smoke` run (or `smoke/` beneath `--output_dir`).
Results record source SHA-256, sensor layout, forecast horizon and split ranges
in `protocol_metadata`, plus fitted scaling statistics in `scalers.npz`.
Resume refuses changed configs, protocols or source hashes, and missing/empty
forecast checkpoints, histories or scalers.
Safety metrics below count predicted threshold crossings; they do not establish
closed-loop furnace safety. Composite rankings are descriptive, not proof of
forecasting skill or statistical significance.

## Historical snapshot benchmark

The following describes the legacy real + synthetic experiment, not the new
forecasting protocol. To reproduce it explicitly, pass
`--config eblnn-benchmark/config/benchmark_config.yaml`; use explicit
`--results_dir` when comparing/plotting its saved outputs.

Legacy comparison:

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

## Legacy quick start

```bash
pip install -r eblnn-benchmark/requirements.txt

# Full benchmark (all four models, three seeds)
python eblnn-benchmark/scripts/run_benchmark.py \
    --config eblnn-benchmark/config/benchmark_config.yaml

# Isolated wiring check: all four models, first seed, at most two epochs.
# Evaluation uses validation data, never the held-out test loader.
python eblnn-benchmark/scripts/run_benchmark.py \
    --config eblnn-benchmark/config/benchmark_config.yaml --smoke-test

# Single seed of EB-LNN vs LSTM vs PID for a quick check
python eblnn-benchmark/scripts/run_benchmark.py \
    --config eblnn-benchmark/config/benchmark_config.yaml \
    --models eblnn lstm pid --seeds 42

# Aggregate to comparison_report.md and comparison_summary.json
python eblnn-benchmark/scripts/compare_results.py \
    --results_dir eblnn-benchmark/results/ebm-tuned-f6ozllp7

# Publication figures
python eblnn-benchmark/scripts/plot_results.py \
    --results_dir eblnn-benchmark/results/ebm-tuned-f6ozllp7
```

## Frozen EBM configuration

The historical run is `ebm-tuned-f6ozllp7`, selected from the 30-trial
validation-only campaign by `best_val_physics` (0.01892160, seed 42).
The frozen settings are `alpha=1.0`, `l2_reg=0.1`, `n_steps=20`,
`step_size=0.01`, `noise_scale=0.005`, and `buffer_prob=0.75`.
Compared with the old benchmark config, both `l2_reg` and `buffer_prob`
change; the tuning baseline already used `l2_reg=0.1`.
Do not retune using benchmark test metrics.

Its run name prevents resume logic from reusing the old `results/default`
outputs. Use explicit result paths for this historical run.
`--smoke-test` disables W&B, uses batches of at most eight sequences and at
most two batches for training/validation, and saves under
`results/ebm-tuned-f6ozllp7-smoke`. With `--output_dir`, smoke outputs go in
its `smoke/` subdirectory. Smoke results explicitly record
`smoke_test: true` and `evaluation_split: validation`; their legacy
`test_metrics` field contains validation-only diagnostics, not test scores.
The full data pipeline is still built to check loading, splitting and scaling.

Run the protocol regression checks with:

```bash
python -m unittest discover -s eblnn-benchmark/tests -v
```

## Outputs

For each seed of each model:
- `results/<run>/<model>/seed_<seed>/results.json`
- `results/<run>/<model>/seed_<seed>/history.npz`
- `results/<run>/<model>/seed_<seed>/scalers.npz`
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
# or:  set wandb.mode=offline in forecast_config.yaml and sync later

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
| `RUN_NAME` | config's `run_name` | Run name → results live under `results/$RUN_NAME/`. |
| `CONFIG_PATH` | `eblnn-benchmark/config/forecast_config.yaml` | Override config. |
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
- **Artifacts**: `results.json`, `history.npz` and `scalers.npz` per seed,
  archived in wandb.

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
