"""
Training Module with WandB Integration
Handles data preparation, training loop, and logging
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Optional
import wandb

from .model import EBLNN, calculate_true_energy


class EBLNNTrainer:
    """
    Trainer class for Energy-Based Liquid Neural Network.
    Handles data preparation, training loop, validation, and WandB logging.
    """
    
    def __init__(
        self,
        model: EBLNN,
        config: Dict,
        device: str = 'cpu',
        use_wandb: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: EBLNN model instance
            config: Configuration dictionary with hyperparameters
            device: Device to train on ('cpu' or 'cuda')
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Extract hyperparameters
        self.alpha = config.get('alpha', 1.0)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epochs = config.get('epochs', 110)
        self.batch_size = config.get('batch_size', 64)
        self.w_safety = config.get('w_safety', 100.0)
        
        # Early stopping parameters
        self.early_stopping = config.get('early_stopping', True)
        self.patience = config.get('patience', 15)
        self.min_delta = config.get('min_delta', 1e-4)
        
        # Loss functions
        self.lnn_criterion = nn.MSELoss()
        self.ebm_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_lnn_loss': [],
            'train_ebm_loss': [],
            'val_lnn_loss': [],
            'val_ebm_loss': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        sequence_length: int = 30,
        test_size: float = 0.2,
        val_size: float = 0.1,
        seed: int = 42
    ) -> Tuple:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with synthetic data
            sequence_length: Length of each sequence
            test_size: Fraction for test set
            val_size: Fraction for validation set
            seed: Random seed
        
        Returns:
            train_loader, val_loader, test_loader, input_scaler, target_scaler
        """
        print("Preparing data...")
        
        # 1. Compute the "true energy" target for the EBM head
        df['energy_true'] = calculate_true_energy(
            df['fuel_flow'].values,
            df['next_excess_o2'].values,
            w_fuel=1.0,
            w_safety=self.w_safety
        )
        
        # 2. Define Input Features and Output Targets
        input_cols = [
            'fuel_flow', 'air_fuel_ratio', 'current_temp',
            'inflow_temp', 'inflow_rate'
        ]
        
        output_cols = [
            'next_temp', 'next_excess_o2', 'energy_true'
        ]
        
        INPUT_FEATURES = len(input_cols)
        OUTPUT_FEATURES = len(output_cols)
        
        # 3. Reshape data into sequences
        n_sequences = len(df) // sequence_length
        
        x_data = df[input_cols].values.reshape(n_sequences, sequence_length, INPUT_FEATURES)
        y_data = df[output_cols].values.reshape(n_sequences, sequence_length, OUTPUT_FEATURES)
        
        # 4. Train/Val/Test Split
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=test_size, random_state=seed
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_test, y_test, test_size=0.5, random_state=seed
        )
        
        print(f"Data split: {len(x_train)} train, {len(x_val)} val, {len(x_test)} test sequences")
        
        # 5. Scaling (CRITICAL for LNNs)
        x_train_2d = x_train.reshape(-1, INPUT_FEATURES)
        y_train_2d = y_train.reshape(-1, OUTPUT_FEATURES)
        
        self.input_scaler = StandardScaler().fit(x_train_2d)
        self.target_scaler = StandardScaler().fit(y_train_2d)
        
        # Apply scaling
        x_train = self.input_scaler.transform(x_train_2d).reshape(x_train.shape)
        y_train = self.target_scaler.transform(y_train_2d).reshape(y_train.shape)
        
        x_val = self.input_scaler.transform(x_val.reshape(-1, INPUT_FEATURES)).reshape(x_val.shape)
        y_val = self.target_scaler.transform(y_val.reshape(-1, OUTPUT_FEATURES)).reshape(y_val.shape)
        
        x_test = self.input_scaler.transform(x_test.reshape(-1, INPUT_FEATURES)).reshape(x_test.shape)
        y_test = self.target_scaler.transform(y_test.reshape(-1, OUTPUT_FEATURES)).reshape(y_test.shape)
        
        print("Data scaling complete")
        
        # 6. Create PyTorch Tensors and DataLoaders
        train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print("DataLoaders created")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Dictionary with average losses
        """
        self.model.train()
        total_train_loss = 0
        total_train_lnn_loss = 0
        total_train_ebm_loss = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            
            # Forward pass
            y_pred, e_pred, _ = self.model(x_batch)
            
            # Separate the true targets
            y_true_pred = y_batch[:, :, 0:2]  # [next_temp, next_excess_o2]
            y_true_energy = y_batch[:, :, 2].unsqueeze(-1)  # [energy_true]
            
            # Calculate joint loss
            loss_lnn = self.lnn_criterion(y_pred, y_true_pred)
            loss_ebm = self.ebm_criterion(e_pred, y_true_energy)
            total_loss = loss_lnn + self.alpha * loss_ebm
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            total_train_loss += total_loss.item()
            total_train_lnn_loss += loss_lnn.item()
            total_train_ebm_loss += loss_ebm.item()
        
        return {
            'train_loss': total_train_loss / len(train_loader),
            'train_lnn_loss': total_train_lnn_loss / len(train_loader),
            'train_ebm_loss': total_train_ebm_loss / len(train_loader)
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary with average losses
        """
        self.model.eval()
        total_val_loss = 0
        total_val_lnn_loss = 0
        total_val_ebm_loss = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                y_pred, e_pred, _ = self.model(x_batch)
                
                y_true_pred = y_batch[:, :, 0:2]
                y_true_energy = y_batch[:, :, 2].unsqueeze(-1)
                
                loss_lnn = self.lnn_criterion(y_pred, y_true_pred)
                loss_ebm = self.ebm_criterion(e_pred, y_true_energy)
                total_loss = loss_lnn + self.alpha * loss_ebm
                
                total_val_loss += total_loss.item()
                total_val_lnn_loss += loss_lnn.item()
                total_val_ebm_loss += loss_ebm.item()
        
        return {
            'val_loss': total_val_loss / len(val_loader),
            'val_lnn_loss': total_val_lnn_loss / len(val_loader),
            'val_ebm_loss': total_val_ebm_loss / len(val_loader)
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: str = 'results/models'
    ):
        """
        Full training loop with WandB logging and early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_path: Path to save best model
        """
        print(f"\nStarting training for up to {self.epochs} epochs...")
        print(f"  Alpha: {self.alpha}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  W_safety: {self.w_safety}")
        print(f"  Batch size: {self.batch_size}")
        if self.early_stopping:
            print(f"  Early stopping: enabled (patience={self.patience}, min_delta={self.min_delta})")
        
        os.makedirs(save_path, exist_ok=True)
        
        for epoch in range(self.epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Store in history
            for key, value in metrics.items():
                self.history[key].append(value)
            
            # Log to WandB
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    **metrics
                })
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | "
                      f"Train Loss: {metrics['train_loss']:.4f} | "
                      f"Val Loss: {metrics['val_loss']:.4f} | "
                      f"LNN: {metrics['val_lnn_loss']:.4f} | "
                      f"EBM: {metrics['val_ebm_loss']:.4f}")
            
            # Check for improvement
            if metrics['val_loss'] < self.best_val_loss - self.min_delta:
                self.best_val_loss = metrics['val_loss']
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0
                
                # Save best model
                model_path = os.path.join(save_path, 'best_model.pth')
                torch.save(self.model.state_dict(), model_path)
                
                print(f"  -> New best model saved! Val loss: {self.best_val_loss:.4f}")
                
                if self.use_wandb:
                    wandb.run.summary['best_val_loss'] = self.best_val_loss
                    wandb.run.summary['best_epoch'] = self.best_epoch
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping check
            if self.early_stopping and self.epochs_without_improvement >= self.patience:
                print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
                print(f"   No improvement for {self.patience} consecutive epochs")
                print(f"   Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
                
                if self.use_wandb:
                    wandb.run.summary['early_stopped'] = True
                    wandb.run.summary['stopped_at_epoch'] = epoch + 1
                break
        
        print(f"\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        if self.early_stopping and self.epochs_without_improvement < self.patience:
            print(f"Completed all {self.epochs} epochs without early stopping")
    
    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\nEvaluating on test set...")
        
        self.model.eval()
        
        all_y_true = []
        all_y_pred = []
        all_e_true = []
        all_e_pred = []
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                y_pred, e_pred, _ = self.model(x_batch)
                
                y_true_pred = y_batch[:, :, 0:2]
                y_true_energy = y_batch[:, :, 2].unsqueeze(-1)
                
                # Store all predictions and true values
                all_y_true.append(y_true_pred.cpu().numpy())
                all_y_pred.append(y_pred.cpu().numpy())
                all_e_true.append(y_true_energy.cpu().numpy())
                all_e_pred.append(e_pred.cpu().numpy())
        
        # Concatenate all batches and reshape to 2D
        all_y_true = np.concatenate(all_y_true).reshape(-1, 2)
        all_y_pred = np.concatenate(all_y_pred).reshape(-1, 2)
        all_e_true = np.concatenate(all_e_true).reshape(-1, 1)
        all_e_pred = np.concatenate(all_e_pred).reshape(-1, 1)
        
        # De-normalize the results
        y_true_full = np.hstack([all_y_true, all_e_true])
        y_pred_full = np.hstack([all_y_pred, all_e_pred])
        
        y_true_denorm = self.target_scaler.inverse_transform(y_true_full)
        y_pred_denorm = self.target_scaler.inverse_transform(y_pred_full)
        
        # Separate the de-normalized columns
        true_temp = y_true_denorm[:, 0]
        pred_temp = y_pred_denorm[:, 0]
        
        true_o2 = y_true_denorm[:, 1]
        pred_o2 = y_pred_denorm[:, 1]
        
        true_energy = y_true_denorm[:, 2]
        pred_energy = y_pred_denorm[:, 2]
        
        # Calculate metrics (RMSE)
        rmse_temp = np.sqrt(np.mean((pred_temp - true_temp)**2))
        rmse_o2 = np.sqrt(np.mean((pred_o2 - true_o2)**2))
        rmse_energy = np.sqrt(np.mean((pred_energy - true_energy)**2))
        
        # Calculate Normalized RMSE for Energy
        mean_true_energy = np.mean(true_energy)
        nrmse_energy_percent = (rmse_energy / mean_true_energy) * 100 if mean_true_energy > 1e-6 else 0.0
        
        metrics = {
            'test_rmse_temp': rmse_temp,
            'test_rmse_o2': rmse_o2,
            'test_rmse_energy': rmse_energy,
            'test_nrmse_energy_percent': nrmse_energy_percent
        }
        
        print("\n--- Test Set Evaluation ---")
        print(f"Prediction Head (LNN):")
        print(f"  - Next Temperature RMSE: {rmse_temp:.4f} °C")
        print(f"  - Next Excess O₂ RMSE: {rmse_o2:.4f} %")
        print(f"\nEnergy Head (EBM):")
        print(f"  - Energy (Cost) RMSE: {rmse_energy:.4f}")
        print(f"  - Energy (Cost) NRMSE: {nrmse_energy_percent:.2f} % (of mean)")
        
        # Log to WandB
        if self.use_wandb:
            wandb.log(metrics)
            wandb.run.summary.update(metrics)
        
        # Store predictions for visualization
        self.test_predictions = {
            'true_temp': true_temp,
            'pred_temp': pred_temp,
            'true_o2': true_o2,
            'pred_o2': pred_o2,
            'true_energy': true_energy,
            'pred_energy': pred_energy
        }
        
        return metrics
