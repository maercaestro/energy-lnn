import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import argparse
import wandb
from sklearn.preprocessing import StandardScaler
import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
import logging

# ==========================================
# 1. LOGGING SETUP
# ==========================================
def setup_logging(log_dir):
    """Setup logging to both file and console"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# ==========================================
# 2. PHYSICS CONSTANTS & CONFIG
# ==========================================
# Thermodynamic Constants for Standard Methane & Crude Oil
CONSTANTS = {
    "LHV_METHANE": 50000.0,  # kJ/kg
    "AFR_STOICH": 17.2,      # kg Air / kg Fuel
    "RHO_FUEL": 0.657,       # kg/m3 (STP)
    "RHO_PROC": 850.0,       # kg/m3 (Approx avg density)
    "CP_PROC": 2.5           # kJ/kg.K
}

# ==========================================
# 3. THE PINN MODEL
# ==========================================
class FurnacePINN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, layers=4):
        super(FurnacePINN, self).__init__()
        
        # Build hidden layers dynamically
        layer_list = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers - 1):
            layer_list.append(nn.Linear(hidden_dim, hidden_dim))
            layer_list.append(nn.Tanh())
        
        # Output layer (OutletT, ExcessO2)
        layer_list.append(nn.Linear(hidden_dim, 2))
        
        self.net = nn.Sequential(*layer_list)
        
        # LEARNABLE PHYSICS PARAMETERS
        # Initialize with reasonable engineering guesses
        self.leakage_k = nn.Parameter(torch.tensor(1.5, dtype=torch.float32)) 
        self.efficiency_eta = nn.Parameter(torch.tensor(0.65, dtype=torch.float32))

    def forward(self, x):
        pred = self.net(x)
        return pred, self.leakage_k, self.efficiency_eta

# ==========================================
# 4. ROBUST PHYSICS LOSS
# ==========================================
def physics_loss_function(preds, batch_y, inputs_phys, params, weights):
    """
    Calculates Loss = MSE_Data + lambda1 * Energy_Residual + lambda2 * Mass_Residual
    Includes CLAMPING to prevent exploding gradients during low-flow.
    """
    # Unpack
    pred_T, pred_O2 = preds[:, 0], preds[:, 1]
    # batch_y is NOT used for physics, only for data loss if mixed. 
    # But here we separate them.
    
    leak_k, eff_eta = params
    w_energy, w_mass = weights
    
    # Inputs: [Tin, Vol_Proc, Vol_Fuel, Tamb, Draft, Density]
    Tin      = inputs_phys[:, 0]
    Vol_Proc = inputs_phys[:, 1]
    Vol_Fuel = inputs_phys[:, 2]
    Draft    = inputs_phys[:, 4]
    
    # --- CLAMPING (CRITICAL FOR STABILITY) ---
    # Prevent division by zero or negative mass flow
    Vol_Proc = torch.clamp(Vol_Proc, min=10.0) 
    Vol_Fuel = torch.clamp(Vol_Fuel, min=1.0)
    
    # --- Energy Balance ---
    # Convert m3/h -> kg/s
    m_fuel = (Vol_Fuel * CONSTANTS["RHO_FUEL"]) / 3600.0
    m_proc = (Vol_Proc * CONSTANTS["RHO_PROC"]) / 3600.0
    
    Q_in  = m_fuel * CONSTANTS["LHV_METHANE"] * eff_eta
    Q_out = m_proc * CONSTANTS["CP_PROC"] * (pred_T - Tin)
    
    res_energy = torch.mean((Q_in - Q_out)**2)
    
    # --- Mass Balance ---
    # Air Flow = k * sqrt(Draft)
    # Clamp Draft to be positive just in case
    m_air = leak_k * torch.sqrt(torch.abs(Draft) + 1e-6)
    
    # Lambda = Air / (Fuel * Stoich)
    lam = m_air / (m_fuel * CONSTANTS["AFR_STOICH"] + 1e-6)
    
    # O2 approx = 21 * (lam - 1) / lam
    # Clamp lambda to avoid negative O2
    lam = torch.clamp(lam, min=1.01) 
    O2_calc = 21.0 * (lam - 1.0) / lam
    
    res_mass = torch.mean((pred_O2 - O2_calc)**2)
    
    return w_energy * res_energy + w_mass * res_mass, res_energy.item(), res_mass.item()

# ==========================================
# 5. METRICS CALCULATION
# ==========================================
def calculate_metrics(model, X_scaled, X_phys, y_true, device='cpu'):
    """Calculate comprehensive metrics for evaluation"""
    model.eval()
    with torch.no_grad():
        X_s = X_scaled.to(device)
        y_t = y_true.to(device)
        
        preds, k, eta = model(X_s)
        
        # MSE for each output
        mse_temp = nn.MSELoss()(preds[:, 0], y_t[:, 0])
        mse_o2 = nn.MSELoss()(preds[:, 1], y_t[:, 1])
        mse_total = nn.MSELoss()(preds, y_t)
        
        # RMSE
        rmse_temp = torch.sqrt(mse_temp)
        rmse_o2 = torch.sqrt(mse_o2)
        rmse_total = torch.sqrt(mse_total)
        
        # MAE
        mae_temp = torch.mean(torch.abs(preds[:, 0] - y_t[:, 0]))
        mae_o2 = torch.mean(torch.abs(preds[:, 1] - y_t[:, 1]))
        
        # R² Score
        ss_tot_temp = torch.sum((y_t[:, 0] - torch.mean(y_t[:, 0]))**2)
        ss_res_temp = torch.sum((y_t[:, 0] - preds[:, 0])**2)
        r2_temp = 1 - (ss_res_temp / (ss_tot_temp + 1e-8))
        
        ss_tot_o2 = torch.sum((y_t[:, 1] - torch.mean(y_t[:, 1]))**2)
        ss_res_o2 = torch.sum((y_t[:, 1] - preds[:, 1])**2)
        r2_o2 = 1 - (ss_res_o2 / (ss_tot_o2 + 1e-8))
        
    return {
        'mse_total': mse_total.item(),
        'mse_temp': mse_temp.item(),
        'mse_o2': mse_o2.item(),
        'rmse_total': rmse_total.item(),
        'rmse_temp': rmse_temp.item(),
        'rmse_o2': rmse_o2.item(),
        'mae_temp': mae_temp.item(),
        'mae_o2': mae_o2.item(),
        'r2_temp': r2_temp.item(),
        'r2_o2': r2_o2.item(),
        'learned_efficiency': eta.item(),
        'learned_leakage': k.item()
    }

# ==========================================
# 6. TRAINING LOOP
# ==========================================
def train(args):
    # Setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("=" * 80)
    logger.info("FURNACE PINN TRAINING")
    logger.info("=" * 80)
    
    # Initialize WandB
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name=f"pinn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    config = wandb.config
    
    # Create unique checkpoint directory for this run to avoid architecture conflicts
    run_checkpoint_dir = os.path.join(config.checkpoint_dir, f"run_{run.id}")
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    logger.info(f"WandB Run: {run.url}")
    logger.info(f"Checkpoints will be saved to: {run_checkpoint_dir}")

    # --- Data Loading ---
    logger.info(f"Loading data from: {config.data_path}")
    if not os.path.exists(config.data_path):
        logger.error(f"Data file {config.data_path} not found.")
        raise FileNotFoundError(f"Data file {config.data_path} not found.")
        
    df = pd.read_csv(config.data_path)
    logger.info(f"Total samples: {len(df)}")
    
    # Chronological Split (70% train, 15% val, 15% test)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Val samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Features & Scaling
    feature_cols = config.feature_cols.split(',')
    target_cols = config.target_cols.split(',')
    
    logger.info(f"Features: {feature_cols}")
    logger.info(f"Targets: {target_cols}")
    
    # Verify columns exist
    missing_features = set(feature_cols) - set(df.columns)
    missing_targets = set(target_cols) - set(df.columns)
    if missing_features or missing_targets:
        logger.error(f"Missing columns - Features: {missing_features}, Targets: {missing_targets}")
        raise ValueError(f"Missing columns in dataset")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[feature_cols])
    X_val_scaled = scaler.transform(val_df[feature_cols])
    X_test_scaled = scaler.transform(test_df[feature_cols])
    
    # Save scaler for later use
    scaler_path = os.path.join(config.checkpoint_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to: {scaler_path}")
    
    # Convert to Tensors
    device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Training data (need both scaled for NN and physical for loss)
    X_train_s = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_train_p = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
    y_train = torch.tensor(train_df[target_cols].values, dtype=torch.float32)
    
    # Validation data
    X_val_s = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_val_p = torch.tensor(val_df[feature_cols].values, dtype=torch.float32)
    y_val = torch.tensor(val_df[target_cols].values, dtype=torch.float32)
    
    # Test data
    X_test_s = torch.tensor(X_test_scaled, dtype=torch.float32)
    X_test_p = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
    y_test = torch.tensor(test_df[target_cols].values, dtype=torch.float32)

    # Create DataLoader
    dataset = TensorDataset(X_train_s, X_train_p, y_train)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    # Initialize Model
    model = FurnacePINN(
        input_dim=len(feature_cols),
        hidden_dim=config.hidden_dim,
        layers=config.layers
    ).to(device)
    
    logger.info(f"Model architecture: {config.layers} layers x {config.hidden_dim} hidden units")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config.scheduler_patience
    )
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info("=" * 80)
    logger.info("Starting Training...")
    logger.info("=" * 80)

    for epoch in range(config.epochs):
        # ========== TRAINING ==========
        model.train()
        total_loss_epoch = 0
        total_data_loss = 0
        total_physics_loss = 0
        num_batches = 0
        
        for bx_s, bx_p, by in loader:
            bx_s = bx_s.to(device)
            bx_p = bx_p.to(device)
            by = by.to(device)
            
            optimizer.zero_grad()
            
            preds, k, eta = model(bx_s)
            
            # Data Loss
            loss_data = nn.MSELoss()(preds, by)
            
            # Physics Loss
            loss_phy, e_res, m_res = physics_loss_function(
                preds, by, bx_p, (k, eta), (config.w_energy, config.w_mass)
            )
            
            loss = loss_data + loss_phy
            loss.backward()
            
            # Gradient Clipping (Prevents explosions)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            
            optimizer.step()
            
            total_loss_epoch += loss.item()
            total_data_loss += loss_data.item()
            total_physics_loss += loss_phy.item()
            num_batches += 1

        avg_train_loss = total_loss_epoch / num_batches
        avg_data_loss = total_data_loss / num_batches
        avg_physics_loss = total_physics_loss / num_batches
        
        # ========== VALIDATION ==========
        val_metrics = calculate_metrics(model, X_val_s, X_val_p, y_val, device)
        val_loss = val_metrics['mse_total']
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to WandB
        wandb.log({
            "epoch": epoch,
            "train/total_loss": avg_train_loss,
            "train/data_loss": avg_data_loss,
            "train/physics_loss": avg_physics_loss,
            "val/mse_total": val_metrics['mse_total'],
            "val/mse_temp": val_metrics['mse_temp'],
            "val/mse_o2": val_metrics['mse_o2'],
            "val/rmse_temp": val_metrics['rmse_temp'],
            "val/rmse_o2": val_metrics['rmse_o2'],
            "val/mae_temp": val_metrics['mae_temp'],
            "val/mae_o2": val_metrics['mae_o2'],
            "val/r2_temp": val_metrics['r2_temp'],
            "val/r2_o2": val_metrics['r2_o2'],
            "params/efficiency_eta": val_metrics['learned_efficiency'],
            "params/leakage_k": val_metrics['learned_leakage'],
            "learning_rate": current_lr
        })
        
        # Logging
        if epoch % config.log_interval == 0:
            logger.info(
                f"Epoch {epoch:4d}/{config.epochs} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val MSE: {val_loss:.6f} | "
                f"Val RMSE (T/O2): {val_metrics['rmse_temp']:.4f}/{val_metrics['rmse_o2']:.4f} | "
                f"η={val_metrics['learned_efficiency']:.4f} k={val_metrics['learned_leakage']:.4f} | "
                f"LR={current_lr:.6f}"
            )
        
        # ========== CHECKPOINTING ==========
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model (unique per run)
            best_model_path = os.path.join(run_checkpoint_dir, 'best_pinn_model.pth')
            
            # Create config dict without WandB objects
            config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config_dict
            }, best_model_path)
            
            logger.info(f"✅ Best model saved! Val MSE: {val_loss:.6f}")
            wandb.run.summary['best_val_mse'] = val_loss
            wandb.run.summary['best_epoch'] = epoch
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config.patience:
            logger.info(f"⚠️  Early stopping triggered after {epoch + 1} epochs")
            logger.info(f"   No improvement for {config.patience} consecutive epochs")
            break
        
        # Periodic checkpoint
        if (epoch + 1) % config.checkpoint_interval == 0:
            checkpoint_path = os.path.join(run_checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, checkpoint_path)
    
    # ========== FINAL EVALUATION ON TEST SET ==========
    logger.info("=" * 80)
    logger.info("Final Evaluation on Test Set")
    logger.info("=" * 80)
    
    # Load best model
    best_checkpoint = torch.load(best_model_path, weights_only=False)
    try:
        model.load_state_dict(best_checkpoint['model_state_dict'], strict=True)
    except RuntimeError as e:
        logger.warning(f"Failed to load checkpoint due to architecture mismatch: {e}")
        logger.warning("Using current model state for evaluation (this may happen with different sweep hyperparameters)")
        # Model already has the latest trained weights, so we can proceed
    
    test_metrics = calculate_metrics(model, X_test_s, X_test_p, y_test, device)
    
    logger.info(f"Test MSE Total: {test_metrics['mse_total']:.6f}")
    logger.info(f"Test RMSE Temperature: {test_metrics['rmse_temp']:.4f} °C")
    logger.info(f"Test RMSE Excess O2: {test_metrics['rmse_o2']:.4f} %")
    logger.info(f"Test MAE Temperature: {test_metrics['mae_temp']:.4f} °C")
    logger.info(f"Test MAE Excess O2: {test_metrics['mae_o2']:.4f} %")
    logger.info(f"Test R² Temperature: {test_metrics['r2_temp']:.4f}")
    logger.info(f"Test R² Excess O2: {test_metrics['r2_o2']:.4f}")
    logger.info(f"Learned Efficiency η: {test_metrics['learned_efficiency']:.4f}")
    logger.info(f"Learned Leakage k: {test_metrics['learned_leakage']:.4f}")
    
    # Log test metrics to WandB
    wandb.log({
        "test/mse_total": test_metrics['mse_total'],
        "test/rmse_temp": test_metrics['rmse_temp'],
        "test/rmse_o2": test_metrics['rmse_o2'],
        "test/mae_temp": test_metrics['mae_temp'],
        "test/mae_o2": test_metrics['mae_o2'],
        "test/r2_temp": test_metrics['r2_temp'],
        "test/r2_o2": test_metrics['r2_o2']
    })
    
    wandb.run.summary.update({
        'test_' + k: v for k, v in test_metrics.items()
    })
    
    # Save final summary (exclude WandB objects)
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    
    summary = {
        'best_epoch': best_checkpoint['epoch'],
        'best_val_metrics': best_checkpoint['val_metrics'],
        'test_metrics': test_metrics,
        'config': config_dict
    }
    
    summary_path = os.path.join(config.checkpoint_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 80)
    logger.info("✅ Training Complete!")
    logger.info("=" * 80)
    
    wandb.finish()

# ==========================================
# 7. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Physics-Informed Neural Network for Industrial Furnace'
    )
    
    # Data & Paths
    parser.add_argument('--data_path', type=str, default='furnace_data_cleaned.csv',
                        help='Path to CSV data file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save training logs')
    
    # Feature Configuration
    parser.add_argument('--feature_cols', type=str,
                        default='InletT-Avg,InletFlow,FGFlow,AmbientT,DraftP,Density',
                        help='Comma-separated list of feature column names')
    parser.add_argument('--target_cols', type=str,
                        default='OutletT,ExcessO2',
                        help='Comma-separated list of target column names')
    
    # Model Architecture
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden layer dimension')
    parser.add_argument('--layers', type=int, default=4,
                        help='Number of hidden layers')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=500,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='L2 regularization weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm')
    
    # Physics Loss Weights
    parser.add_argument('--w_energy', type=float, default=0.1,
                        help='Weight for energy balance physics loss')
    parser.add_argument('--w_mass', type=float, default=0.1,
                        help='Weight for mass balance physics loss')
    
    # Early Stopping & Scheduling
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--scheduler_patience', type=int, default=10,
                        help='LR scheduler patience (epochs)')
    
    # Logging & Checkpointing
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log training stats every N epochs')
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                        help='Save checkpoint every N epochs')
    
    # WandB Configuration
    parser.add_argument('--wandb_project', type=str, default='furnace-pinn-thesis',
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='WandB entity (username or team)')
    
    # Device
    parser.add_argument('--use_cuda', type=lambda x: str(x).lower() == 'true', default=False,
                        help='Use CUDA if available')
    
    args = parser.parse_args()
    
    try:
        train(args)
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}", exc_info=True)
        wandb.finish(exit_code=1)
        sys.exit(1)