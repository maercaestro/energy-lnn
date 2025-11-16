# Energy-Based Liquid Neural Network (EBLNN)

## Pilot Study: Multi-Experiment Framework

A hybrid architecture combining **Liquid Neural Networks (LNN)** with **Energy-Based Models (EBM)** for multi-objective optimization in furnace thermodynamic systems.

### 🎯 Research Objectives

1. **Optimize excess O₂** (1.5-2.5% range)
2. **Minimize energy consumption** (fuel cost)
3. **Maintain safety** (minimize CO emissions)

### 🏗️ Architecture

```
Input (5 features)
    ↓
[CfC Body] - Liquid Neural Network (Closed-form Continuous)
    ↓
    ├─→ [Prediction Head] → (next_temp, next_excess_o2)  [Physics]
    └─→ [Energy Head]      → (energy_cost)               [Multi-objective EBM]
```

**Joint Loss Function:**
```
L_total = L_LNN + α × L_EBM
```

---

## 📁 Project Structure

```
pilot-study/
├── config/
│   ├── base_config.yaml          # Base configuration
│   └── sweep_config.yaml         # Hyperparameter sweep settings
├── data/
│   └── synthetic_temperature_data.csv  # Generated data (created on first run)
├── experiments/
│   ├── run_single_experiment.py  # Single experiment runner
│   └── run_sweep.py              # Hyperparameter sweep runner
├── notebook/
│   └── energy_lnn_pilot.ipynb    # Original pilot notebook
├── src/
│   ├── __init__.py
│   ├── data_generation.py        # Synthetic data generation
│   ├── model.py                  # EBLNN model architecture
│   ├── train.py                  # Training loop with WandB
│   └── utils.py                  # Visualization utilities
├── results/
│   ├── models/                   # Saved model checkpoints
│   └── plots/                    # Generated plots
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up WandB (Optional but Recommended)

```bash
# Login to Weights & Biases
wandb login

# Or set your API key
export WANDB_API_KEY="your-api-key-here"
```

### 3. Run a Single Experiment

```bash
# Run with default configuration
python experiments/run_single_experiment.py

# Run with custom hyperparameters
python experiments/run_single_experiment.py \
    --alpha 2.0 \
    --hidden-size 256 \
    --w-safety 150.0 \
    --learning-rate 0.0005

# Run without WandB (local only)
python experiments/run_single_experiment.py --no-wandb
```

### 4. Run Hyperparameter Sweep

```bash
# Run full grid search sweep
python experiments/run_sweep.py

# The sweep will test all combinations of:
# - Alpha: [0.5, 1.0, 2.0, 5.0]
# - Hidden Size: [64, 128, 256]
# - W_Safety: [50.0, 100.0, 200.0]
# - Learning Rate: [0.0001, 0.0005, 0.001, 0.005]
# 
# Total: 4 × 3 × 3 × 4 = 144 experiments
```

---

## ⚙️ Configuration

### Base Configuration (`config/base_config.yaml`)

Key parameters you can adjust:

- **Model Architecture:**
  - `hidden_size`: Size of CfC hidden layer (default: 128)
  
- **Training:**
  - `epochs`: Number of training epochs (default: 110)
  - `batch_size`: Batch size (default: 64)
  - `learning_rate`: Learning rate (default: 0.001)
  - `alpha`: Balance between LNN and EBM loss (default: 1.0)
  - `w_safety`: Safety weight in energy calculation (default: 100.0)

- **Data:**
  - `num_sequences`: Number of sequences to generate (default: 10,000)
  - `sequence_length`: Length of each sequence (default: 30)

### Sweep Configuration (`config/sweep_config.yaml`)

Defines the hyperparameter grid for sweeps. Modify `values` lists to change search space.

---

## 📊 Outputs

### WandB Dashboard

All experiments are logged to Weights & Biases with:
- **Metrics:** Loss curves, RMSE for temperature, O₂, and energy
- **Visualizations:** 
  - Loss curves (total, LNN, EBM)
  - Parity plots (true vs. predicted)
  - Energy landscape (learned vs. physics-based)
- **Hyperparameters:** All configuration values
- **Model artifacts:** Best model checkpoints

### Local Files

- **Models:** Saved to `results/models/best_model.pth`
- **Plots:** Generated plots in `results/plots/`
- **Data:** Synthetic data in `data/synthetic_temperature_data.csv`

---

## 📈 Key Metrics

### Physics Predictions (LNN Head)
- **Temperature RMSE:** Accuracy of next temperature prediction
- **Excess O₂ RMSE:** Accuracy of next O₂ prediction

### Energy Predictions (EBM Head)
- **Energy RMSE:** Accuracy of multi-objective cost prediction
- **Energy NRMSE:** Normalized RMSE (% of mean)

### Training Metrics
- **Total Loss:** Combined LNN + α×EBM loss
- **LNN Loss:** Physics prediction loss (MSE)
- **EBM Loss:** Energy prediction loss (MSE)

---

## 🔬 Hyperparameter Tuning Guide

### Alpha (α)
- **Low (0.5):** Prioritize physics accuracy
- **Medium (1.0-2.0):** Balanced
- **High (5.0):** Prioritize energy/cost optimization

### Hidden Size
- **Small (64):** Faster training, may underfit
- **Medium (128):** Good balance (default)
- **Large (256):** More capacity, slower training

### W_Safety
- **Low (50):** Less emphasis on CO/safety
- **Medium (100):** Balanced (default)
- **High (200):** Strong safety constraint

### Learning Rate
- **Low (0.0001):** Slow but stable
- **Medium (0.001):** Good default
- **High (0.005):** Faster but may be unstable

---

## 🛠️ Development

### Project Layout

- **`src/data_generation.py`**: Physics-based synthetic data generation
- **`src/model.py`**: EBLNN architecture and energy functions
- **`src/train.py`**: Training loop and evaluation
- **`src/utils.py`**: Visualization and plotting utilities

### Adding New Features

1. **New hyperparameters:** Update `config/base_config.yaml` and `sweep_config.yaml`
2. **New metrics:** Add to `EBLNNTrainer.evaluate()` in `src/train.py`
3. **New visualizations:** Add functions to `src/utils.py`

---

## 📝 Citation

If you use this code for research, please cite:

```bibtex
@software{eblnn2025,
  title={Energy-Based Liquid Neural Network for Multi-Objective Optimization},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/energy-lnn}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

---

## 📜 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **NCPs Library:** For the CfC implementation
- **Weights & Biases:** For experiment tracking
- **PyTorch Team:** For the deep learning framework
