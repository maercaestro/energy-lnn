# Furnace Physics-Informed Neural Network (PINN) Training

Production-ready PyTorch training script for modeling an industrial furnace using Physics-Informed Neural Networks.

## 🎯 Overview

This PINN learns to predict:
- **Outlet Temperature** (T_out)
- **Excess O2** (O2)

Based on 6 operational inputs while simultaneously learning two critical physical parameters:
- **Thermal Efficiency** (η)
- **Leakage Coefficient** (k)

## 🔧 Installation

### On Local Machine
```bash
pip install -r requirements_pinn.txt
```

### On Azure VM
```bash
# Update system
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev

# Install PyTorch (CPU version for Azure VMs without GPU)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip3 install -r requirements_pinn.txt

# Login to WandB
wandb login
```

## 🚀 Quick Start

### Basic Training Run
```bash
python train_pinn.py \
  --data_path furnace_data_cleaned.csv \
  --epochs 500 \
  --batch_size 64 \
  --lr 0.001 \
  --hidden_dim 64 \
  --layers 4 \
  --w_energy 0.1 \
  --w_mass 0.1
```

### With GPU (if available)
```bash
python train_pinn.py \
  --data_path furnace_data_cleaned.csv \
  --use_cuda \
  --batch_size 128 \
  --epochs 1000
```

### Custom Checkpoint Directory
```bash
python train_pinn.py \
  --data_path furnace_data_cleaned.csv \
  --checkpoint_dir ./my_experiments/run_001 \
  --log_dir ./my_experiments/logs
```

## 📊 WandB Sweeps (Hyperparameter Optimization)

### Initialize Sweep
```bash
# Create sweep
wandb sweep sweep_config.yaml

# Run sweep agent
wandb agent YOUR_SWEEP_ID
```

### Run Multiple Agents in Parallel
```bash
# Terminal 1
wandb agent YOUR_SWEEP_ID

# Terminal 2
wandb agent YOUR_SWEEP_ID

# Terminal 3
wandb agent YOUR_SWEEP_ID
```

## 🏗️ Model Architecture

```
Input (6 features)
    ↓
[Linear → Tanh] × N layers (hidden_dim neurons each)
    ↓
Linear → Output (2 predictions)
    ↓
[Outlet Temperature, Excess O2]

Learnable Parameters: η (efficiency), k (leakage)
```

## 📈 Loss Function

```
Total Loss = MSE(Data) + λ₁·MSE(Energy Balance) + λ₂·MSE(Mass Balance)

Where:
  Energy Balance: Q_in(η) = Q_out(T_out)
  Mass Balance:   O2 = f(k, Draft, Fuel)
```

## 🔍 Monitoring Training

### Real-time Metrics (WandB)
- Training loss (data + physics)
- Validation MSE, RMSE, MAE, R²
- Learned parameters (η, k) evolution
- Learning rate schedule

### Local Logs
```bash
tail -f logs/training_YYYYMMDD_HHMMSS.log
```

## 📁 Output Structure

```
checkpoints/
  ├── best_pinn_model.pth          # Best model based on validation loss
  ├── checkpoint_epoch_100.pth     # Periodic checkpoints
  ├── checkpoint_epoch_200.pth
  ├── scaler.pkl                   # StandardScaler for inference
  └── training_summary.json        # Final metrics & config

logs/
  └── training_YYYYMMDD_HHMMSS.log # Detailed training log
```

## 🎛️ Key Hyperparameters

| Parameter | Default | Description | Sweep Range |
|-----------|---------|-------------|-------------|
| `--lr` | 0.001 | Learning rate | 1e-4 to 1e-2 |
| `--hidden_dim` | 64 | Hidden layer size | 32, 64, 128, 256 |
| `--layers` | 4 | Number of layers | 3 to 6 |
| `--w_energy` | 0.1 | Energy physics weight | 0.01 to 1.0 |
| `--w_mass` | 0.1 | Mass physics weight | 0.01 to 1.0 |
| `--batch_size` | 64 | Training batch size | 32, 64, 128 |
| `--patience` | 50 | Early stopping patience | - |

## 🧪 Physics Constraints

The model enforces:
1. **Energy Balance**: Heat input = Heat absorbed by process
2. **Mass Balance**: Air leakage affects oxygen concentration
3. **Clamping**: Prevents division by zero during low-flow transients
4. **Smooth Activations**: Tanh for continuous derivatives

## 📊 Expected Results

After training, typical performance:
- **Temperature RMSE**: 5-15 °C
- **O2 RMSE**: 0.5-2.0 %
- **R² Score**: > 0.90
- **Learned η**: 0.60-0.75 (typical furnace efficiency)
- **Learned k**: 1.0-2.5 (leakage coefficient)

## 🐛 Troubleshooting

### Loss is NaN
- Reduce learning rate: `--lr 0.0001`
- Increase gradient clipping: `--grad_clip 0.5`
- Reduce physics weights: `--w_energy 0.01 --w_mass 0.01`

### Poor Convergence
- Increase model capacity: `--hidden_dim 128 --layers 5`
- Adjust physics weights: `--w_energy 0.5 --w_mass 0.5`
- Enable learning rate scheduling (automatic)

### Out of Memory (OOM)
- Reduce batch size: `--batch_size 32`
- Reduce model size: `--hidden_dim 32 --layers 3`

## 📝 Command Line Arguments

```bash
python train_pinn.py --help
```

## 🔬 For Thesis/Research

### Ablation Studies
```bash
# Data-only (no physics)
python train_pinn.py --w_energy 0.0 --w_mass 0.0

# Energy-only physics
python train_pinn.py --w_energy 0.5 --w_mass 0.0

# Mass-only physics
python train_pinn.py --w_energy 0.0 --w_mass 0.5

# Full PINN
python train_pinn.py --w_energy 0.5 --w_mass 0.5
```

### Reproducibility
All experiments are logged to WandB with full config for reproducibility.

## 📧 Support

For issues or questions, please check:
1. Training logs in `logs/` directory
2. WandB dashboard for metrics visualization
3. Model checkpoints in `checkpoints/` directory

## 📚 Citation

If you use this code for research, please cite:
```
[Your thesis/paper details]
```
