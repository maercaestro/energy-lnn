# 🚀 Best Config + Causality Analysis - Quick Guide

## Overview

This script trains your EBLNN with the **best configuration from your 20-run sweep** and automatically runs all three causality analyses, logging everything to WandB.

**Best Configuration Discovered:**
- **Alpha (α):** 0.5
- **Hidden Size:** 256  
- **W_Safety:** 100
- **Learning Rate:** 0.0005

---

## 🎯 What This Does

1. **Trains model** with best hyperparameters (200 epochs with early stopping)
2. **Evaluates** on test set and logs metrics
3. **Runs 3 causality analyses:**
   - Neural Saliency (feature importance)
   - Temporal Sensitivity (perturbation propagation)
   - Internal Gating (CfC dynamics)
4. **Logs everything to WandB:**
   - Training curves
   - Test metrics
   - All causality plots
   - Feature importance charts
   - Complete analysis summary

---

## 🚀 Usage

### **On Your Local Mac (Test First):**

```bash
cd /Users/abuhuzaifahbidin/Documents/GitHub/energy-lnn/pilot-study

# Run complete workflow
python experiments/run_best_with_analysis.py
```

### **On Azure VM (Final Production Run):**

```bash
# SSH to Azure VM
ssh azureuser@YOUR_VM_IP

# Navigate to project
cd ~/energy-lnn/pilot-study

# Activate environment
source venv/bin/activate

# Run in screen session
screen -S best-training
python experiments/run_best_with_analysis.py

# Detach: Ctrl+A then D
```

---

## ⏱️ Expected Duration

| Phase | Duration | Description |
|-------|----------|-------------|
| **Data Loading** | ~1-2 min | Load/generate 10,000 sequences (cached) |
| **Training** | ~1-2 hours | 200 epochs with early stopping |
| **Testing** | ~2 min | Evaluate on test set |
| **Analysis 1** | ~30 sec | Neural Saliency |
| **Analysis 2** | ~1 min | Temporal Sensitivity (3 features) |
| **Analysis 3** | ~30 sec | Internal Gating |
| **Total** | **~1-2 hours** | End-to-end |

---

## 📊 What Gets Logged to WandB

### **Training Metrics:**
- `train_loss`, `val_loss` (per epoch)
- `train_rmse_temperature`, `train_rmse_excess_o2`, `train_rmse_energy`
- `val_rmse_temperature`, `val_rmse_excess_o2`, `val_rmse_energy`
- Learning curves (plot)
- Parity plots for predictions (plot)

### **Test Metrics:**
- `final_test_rmse_energy` ⭐
- `final_test_rmse_temperature`
- `final_test_rmse_excess_o2`
- `final_test_loss`

### **Causality Analysis:**

#### **1. Neural Saliency:**
- `causality/saliency_heatmap` (image)
- `causality/feature_importance` (bar chart)
- `causality/feature_importance/fuel_flow` (scalar)
- `causality/feature_importance/air_fuel_ratio` (scalar)
- `causality/feature_importance/current_temp` (scalar)
- `causality/feature_importance/inflow_temp` (scalar)
- `causality/feature_importance/inflow_rate` (scalar)

#### **2. Temporal Sensitivity:**
- `causality/temporal_fuel_flow` (image)
- `causality/temporal_air_fuel_ratio` (image)
- `causality/temporal_inflow_rate` (image)
- `causality/temporal/fuel_flow/max_impact` (scalar)
- `causality/temporal/fuel_flow/cumulative_impact` (scalar)
- `causality/temporal/fuel_flow/system_behavior` (string: REACTIVE/INERTIAL)
- *(same for air_fuel_ratio and inflow_rate)*

#### **3. Internal Gating:**
- `causality/internal_gating` (image with 4 subplots)
- `causality/gating/avg_activation_magnitude` (scalar)
- `causality/gating/avg_velocity` (scalar)
- `causality/gating/dynamics_type` (string: FAST/SLOW)

### **Artifacts:**
- **Model Checkpoint:** `eblnn_best_model.pth`
- **Analysis Summary:** `complete_analysis_summary.json`
- **All Plots:** Uploaded as artifact `best-model-analysis`

---

## 📁 Local Output Structure

```
pilot-study/
├── results/
│   ├── models/
│   │   ├── eblnn_best_model.pth        # Best model during training
│   │   └── eblnn_last_model.pth        # Final model
│   ├── plots/
│   │   └── (training visualizations)
│   └── causality_analysis/
│       ├── saliency_heatmap.png        # Feature importance heatmap
│       ├── temporal_fuel_flow.png      # Fuel flow perturbation
│       ├── temporal_air_fuel_ratio.png # AFR perturbation
│       ├── temporal_inflow_rate.png    # Inflow rate perturbation
│       ├── internal_gating.png         # CfC dynamics (4 subplots)
│       └── complete_analysis_summary.json  # All results
```

---

## 📈 Viewing Results

### **WandB Dashboard:**

1. Go to: `https://wandb.ai/your-entity/energy-based-lnn`
2. Find run: `best-config-with-analysis`
3. Navigate to tabs:
   - **Charts:** See training curves and causality metrics
   - **Images:** View all plots (training + causality)
   - **Artifacts:** Download model and analysis files
   - **Overview:** See run summary and configuration

### **Local Files:**

```bash
# View summary
cat results/causality_analysis/complete_analysis_summary.json

# View plots
open results/causality_analysis/*.png
```

---

## 🎯 Expected Results

Based on your best config (alpha=0.5, hidden_size=256):

### **Training:**
- Should achieve **lowest test RMSE** from your sweep
- Early stopping likely triggers around epoch 50-100
- Smooth training curves (no wild oscillations)

### **Neural Saliency:**
- **fuel_flow** should have highest importance (~0.02-0.03)
- **air_fuel_ratio** should be second (~0.015-0.025)
- **current_temp** should be lower (~0.01-0.015)

### **Temporal Sensitivity:**
- **fuel_flow** → REACTIVE (immediate spike)
- **air_fuel_ratio** → REACTIVE (moderate spike)
- **inflow_rate** → INERTIAL (gradual change)

### **Internal Gating:**
- Dynamics: **SLOW** (smooth, stable evolution)
- Avg activation: ~0.5-1.0
- Avg velocity: ~0.01-0.05

---

## 🔧 Customization Options

### **Change Training Duration:**

Edit line in script:
```python
config['training']['epochs'] = 200  # Change to 100 or 300
config['training']['patience'] = 15  # Change early stopping patience
```

### **Test Different Features in Temporal Analysis:**

Edit line in script:
```python
features_to_test = ['fuel_flow', 'air_fuel_ratio', 'inflow_rate']
# Add or remove features: ['fuel_flow', 'current_temp', 'inflow_temp', ...]
```

### **Change Perturbation Magnitude:**

Edit line in script:
```python
perturbation_magnitude=0.1,  # Change to 0.05, 0.15, 0.20, etc.
```

---

## 🐛 Troubleshooting

### **Issue: CUDA out of memory**

Add to script before training:
```python
config['training']['batch_size'] = 32  # Reduce from 64
```

### **Issue: Training too slow**

```python
config['training']['epochs'] = 100  # Reduce from 200
```

### **Issue: WandB not logging**

```bash
# Check login
wandb login

# Check .env file
cat .env
# Should contain: WANDB_API_KEY=your-key
```

### **Issue: Model not found for analysis**

The script automatically falls back to `eblnn_last_model.pth` if best model not found.

---

## 📊 Comparison with Sweep Results

Your sweep found this config had the best performance. This script:
- ✅ **Confirms** the result by retraining
- ✅ **Extends** training (200 epochs vs previous)
- ✅ **Adds** comprehensive causality analysis
- ✅ **Documents** everything in WandB

---

## 🎓 After This Completes

You'll have:
1. ✅ **Verified best model** retrained with more epochs
2. ✅ **Complete interpretability** through 3 causality analyses
3. ✅ **Publication-ready plots** all logged to WandB
4. ✅ **Comprehensive summary** in JSON format
5. ✅ **Model checkpoint** ready for deployment

---

## 🚀 Next Steps

1. **Review WandB dashboard** to validate results
2. **Compare** with sweep results (should be similar or better)
3. **Analyze causality insights** for control strategy
4. **Download model** for production deployment
5. **Write paper** using plots and insights from analysis

---

## 💡 Pro Tips

- Run on **Azure VM with GPU** for faster training
- Use **screen session** so you can disconnect
- **Monitor WandB** dashboard for live progress
- **Save WandB run URL** for future reference

---

## ✅ Success Criteria

After completion, check:
- [ ] Training converged (val_loss decreased)
- [ ] Test RMSE similar to or better than sweep best
- [ ] Saliency shows fuel_flow as most important
- [ ] Temporal response matches physical expectations
- [ ] Internal dynamics are stable (not chaotic)
- [ ] All plots visible in WandB
- [ ] Model checkpoint saved locally

---

**Ready to run! 🎉**

```bash
python experiments/run_best_with_analysis.py
```
