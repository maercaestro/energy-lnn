# lnn-test

Focused benchmark for one question:

Can a Liquid Neural Network outperform an LSTM and a simple neural network when trained on the augmented PINN furnace dataset?

This folder avoids the larger ablation framework in `eblnn/` and keeps only the pieces needed to:

- train `LNN`, `LSTM`, and `MLP` on the same augmented dataset,
- report common regression metrics,
- measure disturbance robustness, and
- measure stability through safety-boundary violations under disturbance.

## Dataset

By default the benchmark uses the augmented PINN dataset as:

- real data: `../dataset/real_furnace_eblnn.csv`
- PINN edge cases: `../dataset/edge_cases_v2_eblnn.csv`

The pipeline keeps real and edge-case sequences separate during sequence construction and then combines them into one train/val/test split.

## What "stability" means here

This repository does not contain a closed-loop controller benchmark inside `lnn-test`.
For this simplified benchmark, stability under disturbance is evaluated with two proxies:

- lower prediction degradation under perturbation,
- fewer critical safety-threshold violations under perturbation.

Those disturbance tests include Gaussian noise, sensor dropout, spikes, temporal shuffling, extrapolation shifts, and feature scaling.

## Quick start

Install dependencies:

```bash
pip install -r lnn-test/requirements.txt
```

Run the full benchmark with the default config:

```bash
python lnn-test/scripts/run_benchmark.py
```

Run only LNN and LSTM for two seeds:

```bash
python lnn-test/scripts/run_benchmark.py --models lnn lstm --seeds 42 123
```

Generate the comparison report again from saved outputs:

```bash
python lnn-test/scripts/compare_results.py --results_dir lnn-test/results/default
```

Generate publication-quality plots:

```bash
python lnn-test/scripts/plot_results.py --results_dir lnn-test/results/default
```

## Outputs

Each benchmark run writes:

- `results/<run_name>/<model>/seed_<seed>/results.json`
- `results/<run_name>/<model>/seed_<seed>/history.npz`
- `results/<run_name>/<model>/seed_<seed>/models/best_model.pth`
- `results/<run_name>/comparison_summary.json`
- `results/<run_name>/comparison_report.md`
- `results/<run_name>/plots/fig1_clean_rmse.png`
- `results/<run_name>/plots/fig2_noise_degradation.png`
- `results/<run_name>/plots/fig3_safety_violations.png`
- `results/<run_name>/plots/fig4_composite_ranking.png`

## Comparison report

The report (`comparison_report.md`) contains four thesis-ready tables:

1. **Table 1 — Prediction Accuracy**: RMSE, MAE, R² in physical units with best-in-class bolded.
2. **Table 2 — Disturbance Robustness**: degradation ratio under noise and all perturbation categories.
3. **Table 3 — Safety Stability**: critical and total violation rates (clean vs disturbance).
4. **Table 4 — Composite Ranking**: weighted score (40% accuracy, 35% robustness, 25% safety).

## Composite ranking

The composite score is a min-max normalised weighted sum over accuracy, disturbance robustness, and safety stability metrics. Lower score = better model. The weights are configurable in `compare_results.py::RANKING_WEIGHTS`.

## Main files

- `src/data.py` shared real + PINN edge-case pipeline
- `src/models.py` LNN, LSTM, and simple MLP baselines
- `src/trainer.py` shared supervised training loop
- `src/evaluate.py` common metrics, disturbance tests, and safety metrics
- `scripts/run_benchmark.py` end-to-end benchmark runner
- `scripts/compare_results.py` aggregate results, thesis tables, and composite ranking
- `scripts/plot_results.py` publication-quality figures (4 plots)