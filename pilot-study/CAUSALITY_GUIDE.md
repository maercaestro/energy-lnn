# 🔍 Causality Analysis Guide for EBLNN

Complete guide for interpreting your trained Energy-Based Liquid Neural Network using three complementary causality analyses.

---

## 📋 Quick Start

### **Command Line (Recommended)**

```bash
cd /Users/abuhuzaifahbidin/Documents/GitHub/energy-lnn/pilot-study

# Run all three analyses
python experiments/analyze_causality.py \
    --model results/models/eblnn_best_model.pth \
    --data data/synthetic_temperature_data.csv \
    --output results/causality_analysis \
    --device cpu
```

### **Python Script**

```python
from experiments.analyze_causality import CausalityAnalyzer

analyzer = CausalityAnalyzer(
    model_path='results/models/eblnn_best_model.pth',
    data_path='data/synthetic_temperature_data.csv'
)

results = analyzer.run_all_analyses()
```

---

## 🎯 Three Analyses Explained

### **1️⃣ Neural Saliency (Gradient-Based Causality)**

**Question:** Which input features drive energy cost predictions?

**How it works:**
1. Forward pass through model with test batch
2. Backpropagate from energy prediction to inputs
3. Compute absolute gradient magnitudes

**Output:**
- 📊 **Heatmap**: Shows which features matter at each timestep
- 📈 **Feature Importance**: Overall ranking of input features

**Interpretation:**
- **High gradient** = Feature strongly influences energy cost
- **Temporal patterns** = When features matter most
- **Zero gradients** = Feature ignored by model

**Command:**
```bash
python experiments/analyze_causality.py \
    --model results/models/eblnn_best_model.pth \
    --data data/synthetic_temperature_data.csv \
    --analysis saliency
```

**Expected Results:**
- `fuel_flow` and `air_fuel_ratio` should have **highest** importance (control combustion)
- `current_temp` should have **lower** importance (state variable)
- Gradients should be **concentrated early** in sequence (initial conditions matter)

---

### **2️⃣ Temporal Sensitivity (Time-Lag Analysis)**

**Question:** How do perturbations propagate through time?

**How it works:**
1. Create baseline sequence
2. Perturb one feature at one timestep (+10%)
3. Compare energy predictions over time

**Output:**
- 📈 **Line Plot**: Energy difference over timesteps
- 🔄 **System Behavior**: REACTIVE (immediate) or INERTIAL (gradual)

**Interpretation:**
- **Immediate spike** = System reacts quickly (no memory)
- **Long tail** = System has inertia (smooth dynamics)
- **Oscillation** = Potential instability

**Command:**
```bash
python experiments/analyze_causality.py \
    --model results/models/eblnn_best_model.pth \
    --data data/synthetic_temperature_data.csv \
    --analysis temporal
```

**Python (Compare Multiple Features):**
```python
analyzer = CausalityAnalyzer(model_path='...', data_path='...')

for feature in ['fuel_flow', 'air_fuel_ratio', 'inflow_rate']:
    result = analyzer.analysis_2_temporal_sensitivity(
        perturbation_feature=feature,
        perturbation_magnitude=0.1,
        save_path=f'results/temporal_{feature}.png'
    )
    print(f"{feature}: max_impact = {abs(result['energy_difference']).max():.6f}")
```

**Expected Results:**
- **Fuel flow** perturbation → Immediate spike (direct control)
- **Temperature** perturbation → Gradual change (thermal inertia)
- **Air-fuel ratio** → Moderate response (combustion dynamics)

---

### **3️⃣ Internal Gating (CfC Interpretability)**

**Question:** What are the internal dynamics of the liquid neural network?

**How it works:**
1. Extract hidden states from CfC layer
2. Analyze activation magnitudes and velocities
3. Visualize temporal evolution

**Output:**
- 🗺️ **Hidden State Heatmap**: Evolution of first 50 neurons
- 📊 **Activation Plot**: Magnitude over time
- 🏃 **Velocity Plot**: Rate of change
- 📉 **Histogram**: Distribution of activations

**Interpretation:**
- **High velocity** = Fast dynamics (quickly adapting)
- **Low velocity** = Slow dynamics (smooth evolution)
- **Sparse activations** = Efficient representation
- **Dense activations** = Complex computation

**Command:**
```bash
python experiments/analyze_causality.py \
    --model results/models/eblnn_best_model.pth \
    --data data/synthetic_temperature_data.csv \
    --analysis gating
```

**Expected Results:**
- **Smooth activation patterns** (not chaotic)
- **Moderate velocity** (stable but responsive)
- **Gaussian-ish distribution** (well-trained)

---

## 📁 Output Structure

After running `run_all_analyses()`, you'll get:

```
results/causality_analysis/
├── saliency_heatmap.png          # Which features drive energy cost
├── temporal_sensitivity.png      # How perturbations propagate
├── internal_gating.png           # Internal CfC dynamics (4 subplots)
└── analysis_summary.json         # Numerical results
```

**analysis_summary.json** contains:
```json
{
  "saliency": {
    "feature_importance": {
      "fuel_flow": 0.0234,
      "air_fuel_ratio": 0.0189,
      "current_temp": 0.0123,
      "inflow_temp": 0.0156,
      "inflow_rate": 0.0145
    },
    "most_important_feature": "fuel_flow"
  },
  "temporal": {
    "perturbation_feature": "fuel_flow",
    "perturbation_magnitude": 0.1,
    "max_impact": 0.0456,
    "cumulative_impact": 0.234,
    "system_behavior": "REACTIVE"
  },
  "gating": {
    "avg_activation_magnitude": 0.567,
    "avg_velocity": 0.0234,
    "dynamics_type": "SLOW"
  }
}
```

---

## 🚀 Complete Examples

### **Example 1: Analyze Best Model from Sweep**

```bash
# After your 20-run sweep completes on Azure VM
cd ~/energy-lnn/pilot-study

# Run full analysis on best model
python experiments/analyze_causality.py \
    --model results/models/eblnn_best_model.pth \
    --data data/synthetic_temperature_data.csv \
    --output results/causality_best_model \
    --device cpu

# View results
ls -lh results/causality_best_model/
```

### **Example 2: Compare Multiple Models**

```python
from experiments.analyze_causality import CausalityAnalyzer
import json

models = [
    'results/models/eblnn_alpha_0.5.pth',
    'results/models/eblnn_alpha_1.0.pth',
    'results/models/eblnn_alpha_2.0.pth'
]

all_results = {}

for model_path in models:
    model_name = model_path.split('/')[-1].replace('.pth', '')
    
    analyzer = CausalityAnalyzer(
        model_path=model_path,
        data_path='data/synthetic_temperature_data.csv',
        output_dir=f'results/causality_{model_name}'
    )
    
    results = analyzer.run_all_analyses()
    all_results[model_name] = results['saliency']['feature_importance']

# Print comparison
print("\n" + "="*80)
print("FEATURE IMPORTANCE COMPARISON ACROSS MODELS")
print("="*80)

for model_name, importances in all_results.items():
    print(f"\n{model_name}:")
    for feature, importance in importances.items():
        print(f"  {feature:20s}: {importance:.6f}")
```

### **Example 3: Test Different Perturbation Magnitudes**

```python
analyzer = CausalityAnalyzer(
    model_path='results/models/eblnn_best_model.pth',
    data_path='data/synthetic_temperature_data.csv'
)

# Test 5%, 10%, 15%, 20% perturbations
magnitudes = [0.05, 0.10, 0.15, 0.20]

print("Testing perturbation sensitivity:")
for mag in magnitudes:
    result = analyzer.analysis_2_temporal_sensitivity(
        perturbation_feature='fuel_flow',
        perturbation_magnitude=mag,
        save_path=f'results/temporal_{int(mag*100)}pct.png'
    )
    
    max_impact = abs(result['energy_difference']).max()
    print(f"  {mag*100:>5.1f}% → Max impact: {max_impact:.6f}")
```

### **Example 4: Programmatic Usage**

```python
from experiments.example_causality import (
    example_basic_all_analyses,
    example_compare_features,
    example_analyze_best_from_sweep
)

# Run all analyses
results = example_basic_all_analyses()

# Compare feature perturbations
feature_comparison = example_compare_features()

# Analyze best model with insights
best_model_insights = example_analyze_best_from_sweep()
```

---

## 🔧 Command-Line Options

```bash
python experiments/analyze_causality.py --help
```

**Available options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path to model checkpoint (required) | - |
| `--data` | Path to data CSV | `data/synthetic_temperature_data.csv` |
| `--output` | Output directory | `results/causality_analysis` |
| `--device` | Device (cpu/cuda) | `cpu` |
| `--analysis` | Which analysis (all/saliency/temporal/gating) | `all` |

**Examples:**

```bash
# Run only saliency analysis
python experiments/analyze_causality.py \
    --model results/models/eblnn_best_model.pth \
    --analysis saliency

# Use GPU if available
python experiments/analyze_causality.py \
    --model results/models/eblnn_best_model.pth \
    --device cuda

# Custom output directory
python experiments/analyze_causality.py \
    --model results/models/eblnn_best_model.pth \
    --output my_custom_analysis/
```

---

## 📊 Interpreting Results

### **What to Look For:**

#### ✅ **Good Signs:**
- Fuel flow and air-fuel ratio have **highest saliency** (they control combustion)
- Temporal response is **smooth** (not chaotic)
- Hidden state dynamics are **stable** (not exploding/vanishing)
- System behavior matches **physical expectations** (thermal inertia)

#### ⚠️ **Warning Signs:**
- Current temp has **higher saliency** than control inputs (model not learning control)
- Temporal response **oscillates wildly** (instability)
- Hidden states **barely change** (model not learning dynamics)
- All features have **similar importance** (model not discriminating)

### **Physical Validation:**

1. **Saliency should match domain knowledge:**
   - Fuel flow ≈ Air-fuel ratio > Inflow rate > Temperatures
   
2. **Temporal response should match physics:**
   - Fuel changes → immediate effect (combustion)
   - Temperature changes → gradual effect (thermal mass)
   
3. **Internal dynamics should be stable:**
   - No sudden jumps in hidden states
   - Activation magnitudes bounded (typically -3 to +3)

---

## 🐛 Troubleshooting

### **Error: Model file not found**
```bash
# Check model exists
ls results/models/*.pth

# If not found, check WandB artifacts or training output
wandb artifact get your-entity/energy-based-lnn/model:latest
```

### **Error: Data file not found**
```bash
# Generate data if missing
cd pilot-study
python -c "from src.data_generation import load_or_generate_data; load_or_generate_data()"
```

### **Error: CUDA out of memory**
```bash
# Use CPU instead
python experiments/analyze_causality.py \
    --model results/models/eblnn_best_model.pth \
    --device cpu
```

### **Error: Import errors**
```bash
# Make sure you're in the right directory
cd /Users/abuhuzaifahbidin/Documents/GitHub/energy-lnn/pilot-study

# Activate environment if using conda/venv
source venv/bin/activate  # or conda activate your-env
```

### **Plots not showing**
If running on Azure VM without display:
- Plots are saved as PNG files automatically
- Use `scp` to copy files to local machine:
```bash
# On your Mac
scp azureuser@VM_IP:~/energy-lnn/pilot-study/results/causality_analysis/*.png .
```

---

## 📖 Further Reading

### **Neural Saliency:**
- Simonyan et al. (2014): "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps"
- Helps understand which inputs matter most for predictions

### **Temporal Sensitivity:**
- Time-lag analysis common in dynamical systems
- Tests model's understanding of temporal dependencies

### **CfC Interpretability:**
- Hasani et al. (2021): "Liquid Time-constant Networks"
- CfC uses ODE-inspired dynamics with interpretable gates

---

## ✅ Checklist

After running causality analysis, you should have:

- [ ] **Saliency heatmap** showing feature importance over time
- [ ] **Temporal sensitivity plot** showing perturbation propagation
- [ ] **Internal gating visualization** with 4 subplots
- [ ] **JSON summary** with numerical results
- [ ] **Understanding** of which features drive energy cost
- [ ] **Validation** that model behavior matches physics
- [ ] **Insights** for improving model or control strategy

---

## 🎓 Use Cases

### **Research:**
- Understand model decision-making
- Validate physical consistency
- Identify important features for control

### **Production:**
- Verify model before deployment
- Debug unexpected predictions
- Build trust with stakeholders

### **Optimization:**
- Identify which inputs to optimize
- Understand temporal trade-offs
- Design better control strategies

---

## 🚀 Next Steps

After causality analysis:

1. **If results are good:** Deploy model to production
2. **If fuel_flow dominates:** Focus control strategy on fuel optimization
3. **If dynamics are fast:** Consider model-based control
4. **If dynamics are slow:** Consider PID-style control
5. **If results are unexpected:** Re-examine training data or hyperparameters

---

**Questions?** Check `experiments/example_causality.py` for more examples!

**Happy analyzing! 🎉**
