# 🚀 Deployment Guide: 20-Run Random Search

## Overview
This guide explains how to deploy the optimized 20-run random search sweep on your Azure VM.

**Previous setup:** 144 grid search experiments (~288 hours = 12 days)  
**New setup:** 20 random search experiments with early termination (~40 hours = 1.5-2 days)

---

## 📋 What Changed

### 1. New Configuration File
- **File:** `config/sweep_config_random20.yaml`
- **Method:** Random search (instead of grid)
- **Run Cap:** 20 experiments
- **Early Termination:** Hyperband (stops poorly performing runs after 15 epochs)
- **Parameters:** Same ranges as original (alpha, hidden_size, w_safety, learning_rate)

### 2. Updated Scripts
- **`experiments/run_sweep.py`**: Now supports `--config` argument
- **`experiments/run_single_experiment.py`**: Already compatible with sweep agent

---

## 🛠️ Deployment Steps

### Step 1: Kill Existing Processes on Azure VM
Your current sweep is running two processes:
```bash
# Kill the running processes
kill 556527 556563

# Verify they're stopped
ps aux | grep python
```

### Step 2: Pull Latest Changes from GitHub
```bash
# Navigate to project directory
cd ~/Documents/GitHub/energy-lnn/pilot-study

# Pull latest changes
git pull origin main
```

### Step 3: Verify New Configuration
```bash
# Check that the new config file exists
ls -lh config/sweep_config_random20.yaml

# Preview the configuration
cat config/sweep_config_random20.yaml
```

### Step 4: Test Configuration (Optional but Recommended)
```bash
# Quick test without WandB
python experiments/run_single_experiment.py --no-wandb

# This should complete in ~2 minutes and verify:
# ✓ Data generation works
# ✓ Model initialization works
# ✓ Training loop works
# ✓ Early stopping works
```

### Step 5: Start Sweep in Screen Session
```bash
# Create a new screen session
screen -S sweep20

# Activate your environment (if using conda/venv)
# conda activate your-env-name

# Start the sweep with new configuration
python experiments/run_sweep.py --config config/sweep_config_random20.yaml

# Detach from screen: Press Ctrl+A then D
```

### Step 6: Monitor Progress
```bash
# Reattach to screen session
screen -r sweep20

# Check WandB dashboard
# URL will be printed in terminal output
# Format: https://wandb.ai/your-entity/energy-based-lnn/sweeps/SWEEP_ID

# Monitor system resources
htop
```

---

## 📊 Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| **Data Generation** | ~2 min | Generate 10,000 sequences (cached after first run) |
| **Per Experiment** | ~2 hours | Train for up to 110 epochs (early stopping may reduce) |
| **Early Termination** | Variable | Hyperband stops poor runs after 15 epochs (~15 min) |
| **Total Sweep** | **~30-40 hours** | 20 runs with early termination |

### Time Savings
- Original: 144 runs × 2 hours = **288 hours** (12 days)
- Optimized: 20 runs × 2 hours ≈ **40 hours** (1.5 days)
- **Savings: 248 hours (10.5 days)** 🎉

---

## 🔍 Configuration Details

### Random Search Parameters
```yaml
method: random
run_cap: 20

parameters:
  alpha:
    values: [0.5, 1.0, 2.0, 5.0]
  hidden_size:
    values: [64, 128, 256]
  w_safety:
    values: [50.0, 100.0, 200.0]
  learning_rate:
    values: [0.0001, 0.0005, 0.001, 0.005]
```

### Early Termination (Hyperband)
```yaml
early_terminate:
  type: hyperband
  min_iter: 15      # Evaluate after 15 epochs
  eta: 3            # Keep top 1/3 of runs
  s: 2              # Number of brackets
```

**How it works:**
1. All runs train for at least 15 epochs
2. After 15 epochs, Hyperband evaluates performance
3. Bottom 2/3 of runs are terminated
4. Top 1/3 continue to 45 epochs (15 × 3)
5. Process repeats until best runs reach 110 epochs

---

## 🎯 Success Criteria

### After 1 Hour (During First Run)
- ✅ WandB sweep created successfully
- ✅ First experiment started
- ✅ Metrics logging to WandB
- ✅ No errors in screen session

### After 20 Runs (~40 Hours)
- ✅ All 20 experiments completed
- ✅ WandB dashboard shows results
- ✅ Best model identified
- ✅ Plots and metrics logged

---

## 🐛 Troubleshooting

### Issue: "Import 'dotenv' could not be resolved"
This is a Pylance lint warning, not a runtime error. The code will still run.
```bash
# If it bothers you, install in your environment:
pip install python-dotenv
```

### Issue: "Import 'torch' could not be resolved"
Same as above - Pylance warning. Verify torch is installed:
```bash
pip list | grep torch
```

### Issue: "WANDB_API_KEY not found"
Make sure `.env` file exists with your API key:
```bash
# Create .env file
echo "WANDB_API_KEY=your-key-here" > .env
```

### Issue: Sweep Not Starting
Check WandB login:
```bash
wandb login
```

### Issue: Out of Memory
Monitor with `htop`. If needed, reduce batch size in `config/base_config.yaml`:
```yaml
training:
  batch_size: 32  # Reduced from 64
```

---

## 📈 Monitoring Tips

### View Live Progress
```bash
# Reattach to screen
screen -r sweep20

# Watch logs in real-time
tail -f wandb/latest-run/logs/debug-internal.log
```

### Check GPU Usage (if using GPU VM)
```bash
# Install nvidia-smi
nvidia-smi

# Watch GPU usage
watch -n 1 nvidia-smi
```

### Estimate Completion Time
```python
# After a few runs complete, check average time per run
# On WandB dashboard: Runs > Table > Sort by duration
# Multiply by remaining runs
```

---

## ✅ Post-Sweep Analysis

### 1. View Results on WandB
- Navigate to sweep dashboard
- Sort by `test_rmse_energy` (lower is better)
- Check parallel coordinates plot
- Review parameter importance

### 2. Identify Best Configuration
```python
# Best run shows optimal hyperparameters:
# - alpha: X.X
# - hidden_size: XXX
# - w_safety: XX.X
# - learning_rate: X.XXXX
```

### 3. Run Final Training with Best Config
```bash
# Update config/base_config.yaml with best parameters
# Then run single experiment
python experiments/run_single_experiment.py
```

---

## 🔄 Alternative: Run from Different Config

### Use Original 144-Run Grid Search
```bash
python experiments/run_sweep.py --config config/sweep_config.yaml
```

### Create Custom Sweep
1. Copy `config/sweep_config_random20.yaml`
2. Modify parameters as needed
3. Run with `--config` pointing to your file

---

## 📞 Support

If you encounter issues:
1. Check screen session logs: `screen -r sweep20`
2. Check WandB dashboard for error messages
3. Verify environment variables: `cat .env`
4. Review this guide's troubleshooting section

---

## 🎉 Summary

You've successfully configured an optimized hyperparameter sweep that will:
- ✅ Explore same parameter space as before
- ✅ Complete 7x faster (40 hours vs 288 hours)
- ✅ Use intelligent early termination
- ✅ Provide comprehensive WandB visualizations

**Next command to run on Azure VM:**
```bash
screen -S sweep20
python experiments/run_sweep.py --config config/sweep_config_random20.yaml
```

Good luck! 🚀
