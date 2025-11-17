# ⚡ Quick Reference: Azure VM Deployment

## 🎯 One-Command Deployment

```bash
# Stop existing sweep
kill 556527 556563

# Pull updates
cd ~/Documents/GitHub/energy-lnn/pilot-study && git pull

# Start new sweep in screen
screen -S sweep20
python experiments/run_sweep.py --config config/sweep_config_random20.yaml
```

Press `Ctrl+A` then `D` to detach from screen.

---

## 📊 Key Stats

| Metric | Original | Optimized |
|--------|----------|-----------|
| **Method** | Grid Search | Random Search |
| **Runs** | 144 | 20 |
| **Duration** | ~288 hours (12 days) | ~40 hours (1.5 days) |
| **Early Stop** | No | Yes (Hyperband) |
| **Time Saved** | - | **248 hours** ⚡ |

---

## 🔧 Essential Commands

### Start Sweep
```bash
screen -S sweep20
python experiments/run_sweep.py --config config/sweep_config_random20.yaml
```

### Monitor Sweep
```bash
# Reattach to screen
screen -r sweep20

# Check processes
ps aux | grep python

# Monitor resources
htop
```

### Stop Sweep
```bash
# Find process IDs
ps aux | grep python

# Kill processes
kill <PID1> <PID2>
```

### Test Before Running
```bash
# Quick local test (no WandB)
python experiments/run_single_experiment.py --no-wandb
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `config/sweep_config_random20.yaml` | 20-run random search config |
| `config/sweep_config.yaml` | Original 144-run grid config |
| `config/base_config.yaml` | Default hyperparameters |
| `experiments/run_sweep.py` | Sweep orchestrator |
| `.env` | WandB API key storage |

---

## 🎨 WandB Dashboard

Access at: `https://wandb.ai/your-entity/energy-based-lnn`

**Key Metrics:**
- `test_rmse_energy` ← **Minimize this**
- `test_rmse_temperature`
- `test_rmse_excess_o2`
- `train_loss`, `val_loss`

**Views:**
- Table: Compare all runs
- Parallel Coordinates: Visualize parameter effects
- Parameter Importance: See what matters most

---

## 🚨 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Can't find screen | `screen -ls` to list sessions |
| Sweep won't start | Check `wandb login` |
| Out of memory | Reduce `batch_size` in base_config.yaml |
| Process killed | Check logs in screen session |
| Import errors | Run `pip install -r requirements.txt` |

---

## 📈 Progress Tracking

### After 2 Hours (First Run)
- ✅ Check WandB: Should see 1 completed run
- ✅ Check screen: Should show second run starting

### After 24 Hours
- ✅ ~12 runs completed
- ✅ Early termination in effect
- ✅ Can see best config emerging

### After 40 Hours
- ✅ All 20 runs complete
- ✅ Best config identified
- ✅ Ready for final training

---

## 🎯 Next Steps After Sweep

1. **Identify Best Config**
   - Open WandB dashboard
   - Sort by `test_rmse_energy`
   - Note best hyperparameters

2. **Update Base Config**
   ```yaml
   # config/base_config.yaml
   training:
     alpha: X.X        # From best run
     learning_rate: X.XXXX
     w_safety: XX.X
   
   model:
     hidden_size: XXX
   ```

3. **Final Training**
   ```bash
   python experiments/run_single_experiment.py
   ```

---

## 💡 Pro Tips

- **Screen Management**: Use `screen -list` to see all sessions
- **Multiple Agents**: Can run multiple sweep agents in parallel
- **Partial Sweeps**: Use `--count N` to run only N experiments
- **Resume Sweep**: If interrupted, same command resumes where it left off
- **Compare Configs**: Run both configs to see grid vs random performance

---

## 🔗 Useful Links

- **WandB Sweeps Docs**: https://docs.wandb.ai/guides/sweeps
- **Hyperband Paper**: https://arxiv.org/abs/1603.06560
- **Screen Manual**: `man screen`

---

## 🆘 Emergency Stops

### Kill Everything Python
```bash
pkill -9 python
```

### Force Kill Screen
```bash
screen -X -S sweep20 quit
```

### Check What's Running
```bash
ps aux | grep -E "(python|screen)"
```

---

**Last Updated**: Current deployment
**Status**: Ready to deploy ✅
