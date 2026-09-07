# EB-LNN EBM/CD Tuning

This campaign tunes only parameters introduced by the EB-LNN extension. The
pilot-study CfC backbone, data composition, sequence length, optimizer, and
training protocol remain fixed. Trials use the real furnace data plus all
edge cases and select models using validation physics MSE; no test metrics are
computed here.

## Search space

| Group | Parameters |
| --- | --- |
| Contrastive divergence | `alpha`, `l2_reg` |
| Langevin sampler | `n_steps`, `step_size`, `noise_scale` |
| Replay initialization | `buffer_prob` |

The fixed values and search baseline are in `base_config.yaml`. The Bayesian
search definition is in `sweep.yaml`. The CFC hidden size, 30-step temporal
window, and EBM-head widths are deliberately fixed to the existing selected
configuration.

## Run on a VM

From the repository root on the VM:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r eblnn/requirements.txt
# .env is already ignored by the repository.
printf 'WANDB_API_KEY=%s\n' 'your-wandb-api-key' > .env
tmux new -s eblnn-tuning
COUNT=30 bash eblnn/ebm_tuning/run_vm.sh
```

Detach with `Ctrl-b d`; reattach with `tmux attach -t eblnn-tuning`. W&B
records each trial's per-epoch metrics and uploads a versioned
`ebm-tuning-trial` artifact containing `trial.json` and `history.npz`.
These can be downloaded later from the W&B run's Artifacts tab even if the VM
is deleted. Local copies are written to `eblnn/results/ebm_tuning/<wandb-run-id>/`.
Set `COUNT=20` for an initial campaign or `COUNT=30` for the final selection.
The launcher automatically loads `WANDB_API_KEY` from the repository-root
`.env` file and passes it to each W&B child process. It never writes the key to
configs, logs, or artifacts.

After selecting the configuration by `best_val_physics`, copy those EBM/CD
and Langevin values into the benchmark configuration, freeze them, and run the
multi-seed benchmark. Do not use benchmark test results to choose these values.