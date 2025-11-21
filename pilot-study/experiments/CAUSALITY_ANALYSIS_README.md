# Causality Analysis for EBLNN

## Overview

The `analyze_causality.py` script provides comprehensive causality analysis for Energy-Based Liquid Neural Network (EBLNN) models. It implements three complementary methods to understand how the model makes predictions and what factors drive energy costs.

## Three Types of Analysis

### 1. Neural Saliency (Gradient-Based Causality)

**Purpose**: Identify which input features most strongly influence energy predictions.

**Method**: 
- Computes gradients of energy predictions with respect to input features
- Uses backpropagation to measure sensitivity
- Generates saliency maps showing temporal and feature-wise importance

**Outputs**:
- Heatmap: Gradient magnitude for each input feature over 30 timesteps
- Bar chart: Overall feature importance (averaged over time)
- Most influential feature identification

**Interpretation**:
- Higher gradient magnitude = stronger causal influence on energy cost
- Temporal patterns show when features matter most
- Identifies which controls (fuel_flow, air_fuel_ratio, etc.) drive costs

### 2. Temporal Sensitivity (Time-Lag Analysis)

**Purpose**: Understand system dynamics and response behavior.

**Method**:
- Takes a baseline sequence
- Applies +10% perturbation to fuel_flow at t=0
- Tracks energy prediction differences over time

**Outputs**:
- Line plot: Baseline vs perturbed energy trajectories
- Bar chart: Energy difference at each timestep
- System behavior classification (Reactive vs Inertial)

**Interpretation**:
- **Reactive**: Max difference occurs within first 5 timesteps (fast response)
- **Inertial**: Max difference occurs after 5 timesteps (delayed/gradual response)
- Reveals system memory and propagation delays

### 3. Internal Gating Analysis (CfC Interpretability)

**Purpose**: Understand internal neural dynamics and hidden state behavior.

**Method**:
- Extracts hidden states from CfC (Closed-form Continuous) layer
- Analyzes activation patterns, magnitudes, and rates of change
- Identifies fast vs slow neural units

**Outputs**:
- Heatmap: Hidden state activations over time (first 50 units)
- Line plots: Activation magnitude and velocity over time
- Histogram: Distribution of activation values
- Fast/slow unit identification

**Interpretation**:
- **Fast units**: High variance, rapid changes (handle transient dynamics)
- **Slow units**: Low variance, stable (maintain long-term memory)
- Mean velocity indicates overall dynamics type
- Reveals internal gating mechanisms of liquid neural network

## Usage

### Basic Usage (All Analyses)

```bash
python experiments/analyze_causality.py \
    --model results/models/best_model.pth \
    --data data/synthetic_temperature_data.csv
```

### Run Specific Analysis

```bash
# Only neural saliency
python experiments/analyze_causality.py \
    --model results/models/best_model.pth \
    --analysis saliency

# Only temporal sensitivity
python experiments/analyze_causality.py \
    --model results/models/best_model.pth \
    --analysis temporal

# Only internal gating
python experiments/analyze_causality.py \
    --model results/models/best_model.pth \
    --analysis gating
```

### Custom Output Directory

```bash
python experiments/analyze_causality.py \
    --model results/models/best_model.pth \
    --output my_analysis_results
```

### GPU Acceleration

```bash
python experiments/analyze_causality.py \
    --model results/models/best_model.pth \
    --device cuda
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path to trained model checkpoint (.pth) | **Required** |
| `--data` | Path to data CSV file | `data/synthetic_temperature_data.csv` |
| `--output` | Output directory for results | `results/causality_analysis` |
| `--device` | Device to run on (cpu/cuda) | `cpu` |
| `--analysis` | Which analysis to run (all/saliency/temporal/gating) | `all` |

## Output Files

After running the analysis, you'll find the following files in the output directory:

```
results/causality_analysis/
├── analysis_1_neural_saliency.png       # Saliency heatmap and feature importance
├── analysis_2_temporal_sensitivity.png  # Time-lag response plots
├── analysis_3_internal_gating.png       # Hidden state dynamics visualization
└── analysis_summary.json                # Numerical results (JSON format)
```

### JSON Summary Structure

```json
{
  "analysis_1_neural_saliency": {
    "feature_importance": {
      "fuel_flow": 0.000053,
      "air_fuel_ratio": 0.000470,
      "current_temp": 0.000009,
      "inflow_temp": 0.000007,
      "inflow_rate": 0.000008
    },
    "most_influential_feature": "air_fuel_ratio",
    "saliency_map": [...]
  },
  "analysis_2_temporal_sensitivity": {
    "max_energy_difference": 0.028629,
    "max_difference_timestep": 6,
    "system_behavior": "Inertial (delayed response)",
    "mean_absolute_difference": 0.006545
  },
  "analysis_3_internal_gating": {
    "hidden_size": 128,
    "mean_activation_magnitude": 4.929429,
    "mean_velocity": 0.495820,
    "dynamics_type": "Fast Dynamics (rapid state changes)",
    "fast_units": [4, 29, 10, 11, 82],
    "slow_units": [119, 31, 71, 73, 111]
  }
}
```

## Example Results Interpretation

### Example 1: Energy Optimization

**Finding**: Neural saliency shows `air_fuel_ratio` has 10x higher importance than `fuel_flow`.

**Insight**: To reduce energy costs, focus on optimizing air-fuel ratio first, then adjust fuel flow.

**Action**: Implement control strategy prioritizing AFR tuning within optimal range (14.7).

### Example 2: Control Response Design

**Finding**: Temporal sensitivity shows "Inertial" behavior with max difference at t=6.

**Insight**: System has ~6 timestep lag between input change and energy impact.

**Action**: Design controllers with predictive capabilities or lookahead of 6+ timesteps.

### Example 3: Model Capacity Assessment

**Finding**: Internal gating shows 10 fast units and mean velocity of 0.49.

**Insight**: Model has sufficient capacity for rapid dynamics; not "stuck" in slow modes.

**Action**: Current architecture appropriate; no need to increase hidden size.

## Technical Details

### Model Requirements

- **Input features** (5): `fuel_flow`, `air_fuel_ratio`, `current_temp`, `inflow_temp`, `inflow_rate`
- **Output features** (2): `next_temp`, `next_excess_o2`
- **Architecture**: EBLNN with CfC body + prediction head + energy head
- **Checkpoint format**: PyTorch state dict (`.pth` file)

### Data Requirements

- **Format**: CSV file with columns matching input/output features
- **Structure**: Sequential data that can be reshaped into sequences
- **Scaling**: Script automatically fits StandardScaler on first 80% of data

### Dependencies

```python
torch>=1.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
sklearn>=0.24.0
ncps>=1.0.0  # For CfC layer
```

## Troubleshooting

### Issue: "Model file not found"
**Solution**: Ensure you've trained a model first using `run_single_experiment.py`

### Issue: "Could not infer hidden_size"
**Solution**: Script will use default (128). If your model uses different size, it will still load correctly.

### Issue: "CUDA out of memory"
**Solution**: Use `--device cpu` instead of `cuda`, or reduce batch size in code

### Issue: "Data file not found"
**Solution**: Run data generation first or provide correct path with `--data`

## Integration with Experiment Workflow

```bash
# 1. Generate data
python experiments/run_single_experiment.py --config config/base_config.yaml

# 2. Train model
# (Model saved to results/models/best_model.pth)

# 3. Run causality analysis
python experiments/analyze_causality.py --model results/models/best_model.pth

# 4. Review results
ls results/causality_analysis/
cat results/causality_analysis/analysis_summary.json
```

## Advanced Usage: Custom Perturbations

To test different perturbation scenarios, modify the `analysis_2_temporal_sensitivity` method:

```python
# In analyze_causality.py, line ~360
# Change perturbation_magnitude parameter
results = analyzer.analysis_2_temporal_sensitivity(perturbation_magnitude=0.20)  # +20%
```

Or test different features:

```python
# In analyze_causality.py, line ~378
# Change perturbed_feature
perturbed_feature = 'current_temp'  # Instead of 'fuel_flow'
```

## Citation

If you use this analysis in your research, please cite:

```bibtex
@software{eblnn_causality_2025,
  title={Causality Analysis for Energy-Based Liquid Neural Networks},
  author={EBLNN Project},
  year={2025},
  url={https://github.com/maercaestro/energy-lnn}
}
```

## Contact & Support

For questions or issues:
1. Check this README
2. Review example outputs in `results/causality_analysis/`
3. Open an issue on GitHub

---

**Last Updated**: 2025-11-21  
**Script Version**: 1.0  
**Compatible with**: EBLNN models using CfC architecture from ncps library
