# Causality Analysis - Quick Reference Guide

## 🚀 Quick Start

```bash
# Run all three analyses
cd /Users/abuhuzaifahbidin/Documents/GitHub/energy-lnn/pilot-study
python experiments/analyze_causality.py --model results/models/best_model.pth
```

## 📊 Three Analysis Types

| Analysis | What It Does | Key Output | Interpretation |
|----------|--------------|------------|----------------|
| **1. Neural Saliency** | Which features drive energy cost | Feature importance ranking | Higher = more causal influence |
| **2. Temporal Sensitivity** | How system responds to changes | Response lag time | Reactive (<5 steps) vs Inertial (>5 steps) |
| **3. Internal Gating** | What's happening inside the model | Fast vs slow neural units | Model dynamics and capacity |

## 🎯 Common Use Cases

### "Which input should I control first?"
```bash
python experiments/analyze_causality.py --model MODEL.pth --analysis saliency
```
→ Check `feature_importance` in output JSON

### "How fast does the system respond?"
```bash
python experiments/analyze_causality.py --model MODEL.pth --analysis temporal
```
→ Check `max_difference_timestep` in output JSON

### "Is my model learning complex dynamics?"
```bash
python experiments/analyze_causality.py --model MODEL.pth --analysis gating
```
→ Check `dynamics_type` and `mean_velocity` in output JSON

## 📁 Output Files

```
results/causality_analysis/
├── analysis_1_neural_saliency.png       ← Feature importance heatmap
├── analysis_2_temporal_sensitivity.png  ← Time-lag response plot
├── analysis_3_internal_gating.png       ← Hidden state dynamics
└── analysis_summary.json                ← All numerical results
```

## 🔍 Interpreting Results

### Analysis 1: Feature Importance

```json
"feature_importance": {
  "fuel_flow": 0.000053,
  "air_fuel_ratio": 0.000470,  ← 10x more important!
  "current_temp": 0.000009
}
```

**Meaning**: `air_fuel_ratio` has the strongest causal effect on energy cost.

**Action**: Prioritize optimizing AFR in control strategies.

---

### Analysis 2: System Response

```json
"max_difference_timestep": 6,
"system_behavior": "Inertial (delayed response)"
```

**Meaning**: System takes 6 timesteps to show maximum effect of input change.

**Action**: Design controllers with 6+ step lookahead or use Model Predictive Control (MPC).

---

### Analysis 3: Internal Dynamics

```json
"mean_velocity": 0.495820,
"dynamics_type": "Fast Dynamics (rapid state changes)"
```

**Meaning**: Model learns fast-changing patterns; not stuck in slow modes.

**Action**: Current architecture is appropriate; sufficient capacity for problem.

## 🛠️ Command Options

### Minimal (all defaults)
```bash
python experiments/analyze_causality.py --model results/models/best_model.pth
```

### Custom output location
```bash
python experiments/analyze_causality.py --model MODEL.pth --output my_results
```

### Specific analysis only
```bash
python experiments/analyze_causality.py --model MODEL.pth --analysis saliency
# Options: saliency, temporal, gating, all
```

### Use GPU
```bash
python experiments/analyze_causality.py --model MODEL.pth --device cuda
```

## 📈 Expected Results (Example)

Based on EBLNN furnace model:

| Metric | Typical Value | What It Means |
|--------|---------------|---------------|
| Most influential feature | `air_fuel_ratio` | AFR optimization is critical |
| Response lag | 4-8 timesteps | Moderate inertia |
| Mean velocity | 0.3-0.6 | Balanced dynamics |
| Fast units | 5-15 units | Sufficient fast dynamics |

## ⚠️ Common Issues

| Error | Solution |
|-------|----------|
| "Model file not found" | Train model first: `python experiments/run_single_experiment.py` |
| "Data file not found" | Check path: `data/synthetic_temperature_data.csv` exists? |
| "CUDA out of memory" | Use `--device cpu` |

## 🔗 Workflow Integration

```
1. Train Model         → python experiments/run_single_experiment.py
2. Analyze Causality   → python experiments/analyze_causality.py --model results/models/best_model.pth
3. Review Results      → Open results/causality_analysis/*.png
4. Extract Insights    → Read analysis_summary.json
5. Apply to Control    → Use feature_importance to prioritize controls
```

## 💡 Pro Tips

1. **Run after hyperparameter sweeps**: Compare causality across different model configurations
2. **Check consistency**: Run analysis multiple times with different random test sequences
3. **Compare perturbations**: Test different input features in temporal sensitivity
4. **Validate findings**: Cross-reference with domain knowledge (e.g., combustion theory)
5. **Save analysis versions**: Use `--output` with timestamp for experiment tracking

## 📚 Related Files

- **Main script**: `experiments/analyze_causality.py`
- **Full documentation**: `experiments/CAUSALITY_ANALYSIS_README.md`
- **Model definition**: `src/model.py`
- **Data generation**: `src/data_generation.py`

## 🎓 Key Concepts

**Saliency**: Gradient magnitude showing how much output changes with small input changes

**Time-lag**: Delay between cause (input change) and effect (energy change)

**Gating**: Internal mechanism controlling information flow in neural network

**CfC Layer**: Closed-form Continuous layer (liquid neural network component)

**Fast/Slow Units**: Hidden units with high/low variance (different time scales)

---

**Need more details?** See `CAUSALITY_ANALYSIS_README.md`

**Questions?** Open an issue on GitHub
