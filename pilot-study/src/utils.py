"""
Visualization and Evaluation Utilities
Plotting functions for loss curves, parity plots, and energy landscapes
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI threading issues
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb
from typing import Dict, Optional


def plot_loss_curves(history: Dict, save_path: Optional[str] = None):
    """
    Plots training and validation loss curves.
    
    Args:
        history: Dictionary with loss history
        save_path: Path to save figure (optional)
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Total Loss
    axes[0].plot(epochs, history['train_loss'], label='Training Total Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Validation Total Loss', linewidth=2)
    axes[0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: LNN Prediction Loss
    axes[1].plot(epochs, history['val_lnn_loss'], label='Validation LNN Loss', 
                 color='orange', linewidth=2)
    axes[1].set_title('Prediction (LNN) Loss', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: EBM Energy Loss
    axes[2].plot(epochs, history['val_ebm_loss'], label='Validation EBM Loss', 
                 color='green', linewidth=2)
    axes[2].set_title('Energy (EBM) Loss', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved loss curves to {save_path}")
    
    return fig


def plot_parity_plots(
    true_temp: np.ndarray,
    pred_temp: np.ndarray,
    true_o2: np.ndarray,
    pred_o2: np.ndarray,
    true_energy: np.ndarray,
    pred_energy: np.ndarray,
    save_path: Optional[str] = None,
    sample_size: int = 5000
):
    """
    Plots parity plots (true vs. predicted) for all targets.
    
    Args:
        true_temp: True temperature values
        pred_temp: Predicted temperature values
        true_o2: True O2 values
        pred_o2: Predicted O2 values
        true_energy: True energy values
        pred_energy: Predicted energy values
        save_path: Path to save figure (optional)
        sample_size: Number of points to plot (for performance)
    """
    # Sample points to avoid overcrowding
    n_points = min(sample_size, len(true_temp))
    indices = np.random.choice(len(true_temp), n_points, replace=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Temperature
    min_temp = min(np.min(true_temp), np.min(pred_temp))
    max_temp = max(np.max(true_temp), np.max(pred_temp))
    axes[0].scatter(true_temp[indices], pred_temp[indices], alpha=0.3, s=10)
    axes[0].plot([min_temp, max_temp], [min_temp, max_temp], 'r--', linewidth=2, label='y=x')
    axes[0].set_title('LNN Head: Temperature (°C)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('True Next Temperature')
    axes[0].set_ylabel('Predicted Next Temperature')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Excess O₂
    min_o2 = min(np.min(true_o2), np.min(pred_o2))
    max_o2 = max(np.max(true_o2), np.max(pred_o2))
    axes[1].scatter(true_o2[indices], pred_o2[indices], alpha=0.3, s=10, color='orange')
    axes[1].plot([min_o2, max_o2], [min_o2, max_o2], 'r--', linewidth=2, label='y=x')
    axes[1].set_title('LNN Head: Excess O₂ (%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('True Next O₂')
    axes[1].set_ylabel('Predicted Next O₂')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Energy (Cost)
    min_e = min(np.min(true_energy), np.min(pred_energy))
    max_e = max(np.max(true_energy), np.max(pred_energy))
    axes[2].scatter(true_energy[indices], pred_energy[indices], alpha=0.3, s=10, color='green')
    axes[2].plot([min_e, max_e], [min_e, max_e], 'r--', linewidth=2, label='y=x')
    axes[2].set_title('EBM Head: Energy (Cost)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('True Energy')
    axes[2].set_ylabel('Predicted Energy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved parity plots to {save_path}")
    
    return fig


def plot_energy_landscape(
    model,
    input_scaler,
    target_scaler,
    device: str = 'cpu',
    save_path: Optional[str] = None,
    T_curr_fixed: float = 450.0,
    T_in_fixed: float = 150.0,
    F_in_fixed: float = 125.0
):
    """
    Plots the learned energy landscape from the EBM head.
    
    Args:
        model: Trained EBLNN model
        input_scaler: StandardScaler for inputs
        target_scaler: StandardScaler for targets
        device: Device model is on
        save_path: Path to save figure (optional)
        T_curr_fixed: Fixed current temperature
        T_in_fixed: Fixed inflow temperature
        F_in_fixed: Fixed inflow rate
    """
    from .model import calculate_true_energy
    
    model.eval()
    
    # Create a grid of actions
    fuel_flow_range = np.linspace(1.0, 20.0, 50)
    afr_range = np.linspace(10.0, 25.0, 50)
    xx, yy = np.meshgrid(fuel_flow_range, afr_range)
    
    # Create the input batch
    # Input cols: ['fuel_flow', 'air_fuel_ratio', 'current_temp', 'inflow_temp', 'inflow_rate']
    grid_inputs = np.zeros((xx.ravel().shape[0], 5))
    grid_inputs[:, 0] = xx.ravel()
    grid_inputs[:, 1] = yy.ravel()
    grid_inputs[:, 2] = T_curr_fixed
    grid_inputs[:, 3] = T_in_fixed
    grid_inputs[:, 4] = F_in_fixed
    
    # Scale the inputs
    grid_scaled = input_scaler.transform(grid_inputs)
    
    # Reshape for LNN (batch_size, seq_len=1, features)
    grid_tensor = torch.FloatTensor(grid_scaled).unsqueeze(1).to(device)
    
    # Run the model
    with torch.no_grad():
        _, e_pred, _ = model(grid_tensor)
    
    # De-scale the energy prediction
    e_pred_cpu = e_pred.squeeze().cpu().numpy()
    energy_mean = target_scaler.mean_[2]
    energy_std = target_scaler.scale_[2]
    e_pred_denorm = (e_pred_cpu * energy_std) + energy_mean
    
    # Reshape back to 2D grid
    Z_pred = e_pred_denorm.reshape(xx.shape)
    
    # Calculate true energy landscape for comparison
    O2_STOICH = 20.9
    true_o2_grid = []
    for afr in yy.ravel():
        if afr > 1.0:
            true_o2_grid.append(O2_STOICH * (1 - 1/afr) * 100)
        else:
            true_o2_grid.append(0.0)
    
    true_o2_grid = np.array(true_o2_grid)
    true_energy_grid = calculate_true_energy(xx.ravel(), true_o2_grid)
    Z_true = true_energy_grid.reshape(xx.shape)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot 1: Learned Energy Landscape
    contour1 = axes[0].contourf(xx, yy, Z_pred, levels=50, cmap='viridis_r')
    plt.colorbar(contour1, ax=axes[0], label='Predicted Energy (Cost)')
    axes[0].set_title(f'Learned Energy Landscape (EBM Head)\nFixed State: T_curr={T_curr_fixed}°C',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Fuel Flow (units/hr)')
    axes[0].set_ylabel('Air-Fuel Ratio')
    
    # Plot 2: True Physics Energy Landscape
    contour2 = axes[1].contourf(xx, yy, Z_true, levels=50, cmap='viridis_r')
    plt.colorbar(contour2, ax=axes[1], label='True Energy (Cost)')
    axes[1].set_title(f'True Physics Energy Landscape\nFixed State: T_curr={T_curr_fixed}°C',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Fuel Flow (units/hr)')
    axes[1].set_ylabel('Air-Fuel Ratio')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved energy landscape to {save_path}")
    
    return fig


def log_plots_to_wandb(
    history: Dict,
    predictions: Dict,
    model,
    input_scaler,
    target_scaler,
    device: str = 'cpu'
):
    """
    Log all plots to Weights & Biases.
    
    Args:
        history: Training history dictionary
        predictions: Test predictions dictionary
        model: Trained model
        input_scaler: Input scaler
        target_scaler: Target scaler
        device: Device model is on
    """
    # Loss curves
    fig_loss = plot_loss_curves(history)
    wandb.log({"loss_curves": wandb.Image(fig_loss)})
    plt.close(fig_loss)
    
    # Parity plots
    fig_parity = plot_parity_plots(
        predictions['true_temp'],
        predictions['pred_temp'],
        predictions['true_o2'],
        predictions['pred_o2'],
        predictions['true_energy'],
        predictions['pred_energy']
    )
    wandb.log({"parity_plots": wandb.Image(fig_parity)})
    plt.close(fig_parity)
    
    # Energy landscape
    fig_landscape = plot_energy_landscape(
        model, input_scaler, target_scaler, device
    )
    wandb.log({"energy_landscape": wandb.Image(fig_landscape)})
    plt.close(fig_landscape)
    
    print("All plots logged to WandB")
