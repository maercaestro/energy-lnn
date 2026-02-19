# Generative EB-LNN

**Energy-Based Liquid Neural Network with a Generative Search Space**

This module is the successor to `pilot-study/`. It implements a *proper*
Energy-Based Model head trained via **Contrastive Divergence** and
**Langevin Dynamics** — eliminating the hardcoded energy formula used in the
pilot.

---

## What changed from the pilot study

| Aspect | Pilot study | This module |
|---|---|---|
| EBM head | `nn.Linear(H → 1)` | `MLP(H → … → 1)` |
| EBM training | MSE vs `calculate_true_energy` | Contrastive Divergence |
| Energy formula | Hardcoded (`w_fuel·fuel + w_safety·CO`) | **Learned** |
| Negative samples | None | Langevin-generated fantasy states |
| Replay buffer | None | ✓ (capacity 10 000) |
| CD loss | None | `L_CD = E(real) − E(fantasy)` |

In the pilot the EBM head was essentially a supervised regression head
that memorised a hand-designed cost function. Here the energy landscape
is shaped *entirely* by the data + Langevin MCMC, making it a genuine
generative model.

---

## Architecture

```
(State_t, Action_t)
        ↓
 [CfC Body — Liquid Neural Network backbone]
        ↓  Hidden State h_t
        ├──→ [Physics Head]  →  (next_temp, next_O2)     MSE loss
        └──→ [EBM Head MLP]  →  Scalar Energy E(h_t)     CD  loss

Generative Search Space (Langevin MCMC)
────────────────────────────────────────
  x_0  ~  ReplayBuffer  (or N(0,I) for 5% of batches)
  x_{i+1} = x_i  −  (α/2)·∇_x E_θ(x_i)  +  √α · ε,   ε~N(0,I)
  x_T  →  fantasy (negative) state

Joint Loss
──────────
  L = L_physics  +  α · L_CD
  L_CD = E_θ(x_real) − E_θ(x_fantasy) + λ(E_real² + E_fantasy²)
```

---

## Directory structure

```
eblnn/
├── src/
│   ├── __init__.py         Public API
│   ├── model.py            EBLNN_Generative, EBMHead, PhysicsHead
│   ├── sampler.py          LangevinSampler + ReplayBuffer
│   ├── losses.py           ContrastiveDivergenceLoss, JointLoss
│   ├── data.py             DataPipeline (wraps pilot's FurnaceDataGenerator)
│   └── train.py            GenerativeTrainer (full loop + W&B)
├── config/
│   ├── base_config.yaml    Default hyperparameters
│   └── sweep_config.yaml   W&B sweep over CD / Langevin params
├── experiments/
│   └── run_experiment.py   CLI entry-point
├── results/                (created on first run)
│   ├── models/
│   └── plots/
├── data/                   (created on first run)
└── requirements.txt
```

---

## Quick start

```bash
cd energy-lnn/eblnn

# Install dependencies
pip install -r requirements.txt

# Single run with defaults
python experiments/run_experiment.py

# Override key parameters
python experiments/run_experiment.py \
    --alpha 2.0 \
    --hidden_size 256 \
    --n_steps 30 \
    --no_wandb

# W&B hyperparameter sweep
wandb sweep config/sweep_config.yaml
wandb agent <sweep_id>
```

---

## Key hyperparameters (new vs pilot)

| Parameter | Default | Meaning |
|---|---|---|
| `alpha` | 1.0 | CD loss weight relative to physics loss |
| `l2_reg` | 0.01 | Energy-magnitude regularisation |
| `n_steps` | 20 | Langevin MCMC steps per training batch |
| `step_size` | 0.01 | Langevin gradient step size |
| `noise_scale` | 0.005 | Gaussian noise per Langevin step |
| `buffer_capacity` | 10 000 | Replay buffer size |
| `buffer_prob` | 0.95 | P(init from buffer vs fresh noise) |
| `ebm_hidden_dims` | [128, 64] | MLP widths for EBM head |

---

## W&B metrics to watch

| Metric | Expected behaviour |
|---|---|
| `train/cd_gap` | Should grow positive (negatives higher energy than positives) |
| `train/e_pos` | Should decrease (model assigns low energy to real data) |
| `train/e_neg` | Should increase (model assigns high energy to fantasy states) |
| `val/physics` | Should track training; governs prediction quality |
| `val/loss` | Used for early stopping and model selection |

---

## Design notes

### Why Langevin over random negatives?
Random negatives (Gibbs) are far from the data manifold and provide
weak learning signal. Langevin guides the sampler *along the gradient*
toward the model's low-energy regions, producing hard negatives that
sharpen the energy boundary.

### Why a replay buffer?
Running $n=20$ Langevin steps from scratch every batch iteration is
expensive. Storing past fantasy samples and continuing the chains from
where they left off (95% probability) reduces the required steps to
achieve mixing without sacrificing sample quality.  
Reference: Du & Mordatch, *Implicit Generation and Modeling with EBMs*,
NeurIPS 2019.

### EBM head as MLP not Linear
The pilot's single `nn.Linear` could only learn a hyperplane in hidden-
state space. The multi-layer EBM head learns non-linear energy contours,
necessary to represent the multi-modal, non-convex cost landscape of a
combustion control problem.
