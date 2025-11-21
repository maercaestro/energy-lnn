"""
Causality Analysis for Energy-Based Liquid Neural Network (EBLNN)

This script implements three complementary causality analyses:
1. Neural Saliency: Gradient-based feature importance for energy prediction
2. Temporal Sensitivity: Time-lag analysis of input perturbations
3. Internal Gating Analysis: CfC layer interpretability and dynamics

Author: Generated for EBLNN Project
Date: 2025-11-21
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import EBLNN
from src.data_generation import load_or_generate_data

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


class CausalityAnalyzer:
    """
    Comprehensive causality analysis system for EBLNN models.
    
    Implements three types of analyses:
    - Neural Saliency: Gradient-based causality
    - Temporal Sensitivity: Time-lag perturbation analysis
    - Internal Gating: CfC interpretability
    """
    
    def __init__(
        self,
        model_path: str,
        data_path: str,
        device: str = 'cpu',
        output_dir: str = 'results/causality_analysis'
    ):
        """
        Initialize the CausalityAnalyzer.
        
        Args:
            model_path: Path to trained model checkpoint (.pth file)
            data_path: Path to data CSV file
            device: Device to run analysis on ('cpu' or 'cuda')
            output_dir: Directory to save analysis results
        """
        self.model_path = model_path
        self.data_path = data_path
        self.device = torch.device(device)
        self.output_dir = output_dir
        
        # Feature names
        self.input_features = [
            'fuel_flow', 'air_fuel_ratio', 'current_temp',
            'inflow_temp', 'inflow_rate'
        ]
        self.output_features = ['next_temp', 'next_excess_o2']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Initializing CausalityAnalyzer...")
        print(f"  Model: {model_path}")
        print(f"  Data: {data_path}")
        print(f"  Device: {device}")
        print(f"  Output: {output_dir}")
        
        # Load model and data
        self.model = self._load_model()
        self.df = self._load_data()
        self.input_scaler, self.output_scaler = self._create_scalers()
        
        print("✅ Initialization complete!\n")
    
    def _load_model(self) -> EBLNN:
        """
        Load trained EBLNN model from checkpoint.
        
        Returns:
            Loaded EBLNN model in evaluation mode
        """
        print("Loading model...")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Infer hidden size from state dict
        # The CfC body has a weight matrix we can use to infer hidden_size
        hidden_size = None
        for key, tensor in checkpoint.items():
            if 'cfc_body' in key and 'weight' in key:
                # Find the hidden dimension
                if len(tensor.shape) >= 2:
                    hidden_size = tensor.shape[0]
                    break
        
        if hidden_size is None:
            # Default fallback
            hidden_size = 128
            print(f"  Warning: Could not infer hidden_size, using default {hidden_size}")
        else:
            print(f"  Inferred hidden_size: {hidden_size}")
        
        # Create model
        model = EBLNN(
            input_size=len(self.input_features),
            hidden_size=hidden_size,
            output_size_prediction=2,
            output_size_energy=1
        )
        
        # Load state dict
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        model.eval()
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Model loaded: {num_params:,} parameters")
        
        return model
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            DataFrame with furnace data
        """
        print("Loading data...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        print(f"  Loaded {len(df)} rows")
        
        return df
    
    def _create_scalers(self) -> Tuple[StandardScaler, StandardScaler]:
        """
        Create scalers fitted on training data (first 80%).
        
        Returns:
            Tuple of (input_scaler, output_scaler)
        """
        print("Creating scalers...")
        
        # Use first 80% as training data for scaler fitting
        train_size = int(0.8 * len(self.df))
        train_df = self.df.iloc[:train_size]
        
        # Fit input scaler
        input_scaler = StandardScaler()
        input_data = train_df[self.input_features].values
        input_scaler.fit(input_data)
        
        # Fit output scaler
        output_scaler = StandardScaler()
        output_data = train_df[self.output_features].values
        output_scaler.fit(output_data)
        
        print("  Scalers fitted on training data")
        
        return input_scaler, output_scaler
    
    def get_test_batch(
        self,
        sequence_length: int = 30,
        num_sequences: int = 10,
        start_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Extract test sequences from data.
        
        Args:
            sequence_length: Length of each sequence
            num_sequences: Number of sequences to extract
            start_idx: Starting index (if None, use test set start)
        
        Returns:
            Tuple of (scaled_tensor, raw_array)
        """
        # Use last 20% as test data
        test_start = int(0.8 * len(self.df))
        
        if start_idx is None:
            start_idx = test_start
        
        # Extract sequences
        sequences = []
        for i in range(num_sequences):
            seq_start = start_idx + i * sequence_length
            seq_end = seq_start + sequence_length
            
            if seq_end > len(self.df):
                break
            
            seq_data = self.df.iloc[seq_start:seq_end][self.input_features].values
            sequences.append(seq_data)
        
        sequences = np.array(sequences)  # Shape: (num_sequences, seq_len, input_size)
        
        # Scale sequences
        original_shape = sequences.shape
        sequences_2d = sequences.reshape(-1, len(self.input_features))
        sequences_scaled = self.input_scaler.transform(sequences_2d)
        sequences_scaled = sequences_scaled.reshape(original_shape)
        
        # Convert to tensor
        tensor = torch.FloatTensor(sequences_scaled).to(self.device)
        
        return tensor, sequences
    
    def analysis_1_neural_saliency(self) -> Dict:
        """
        Analysis 1: Neural Saliency (Gradient-Based Causality)
        
        Computes the gradient of energy predictions with respect to input features
        to identify which features drive energy cost.
        
        Returns:
            Dictionary with saliency results
        """
        print("\n" + "="*70)
        print("ANALYSIS 1: NEURAL SALIENCY (Gradient-Based Causality)")
        print("="*70)
        
        # Get test batch
        inputs, raw_inputs = self.get_test_batch(sequence_length=30, num_sequences=50)
        
        # Enable gradients on inputs
        inputs.requires_grad = True
        
        # Forward pass
        y_pred, energy_pred, _ = self.model(inputs)
        
        # Compute mean energy (scalar for backprop)
        energy_mean = energy_pred.mean()
        
        # Backward pass
        self.model.zero_grad()
        energy_mean.backward()
        
        # Get gradients (saliency map)
        saliency = inputs.grad.abs().cpu().numpy()  # Shape: (batch, seq_len, input_size)
        
        # Average over batch and compute per-feature, per-timestep saliency
        saliency_avg = saliency.mean(axis=0)  # Shape: (seq_len, input_size)
        
        # Overall feature importance (average over time)
        feature_importance = saliency_avg.mean(axis=0)
        
        # Print results
        print("\nFeature Importance (Impact on Energy Cost):")
        print("-" * 50)
        for i, feature in enumerate(self.input_features):
            print(f"  {feature:20s}: {feature_importance[i]:.6f}")
        
        # Identify most influential feature
        most_influential_idx = np.argmax(feature_importance)
        most_influential = self.input_features[most_influential_idx]
        print(f"\n➤ Most influential feature: {most_influential}")
        
        # Visualize saliency heatmap
        self._plot_saliency_heatmap(saliency_avg, feature_importance)
        
        results = {
            'saliency_map': saliency_avg.tolist(),
            'feature_importance': {
                feature: float(importance)
                for feature, importance in zip(self.input_features, feature_importance)
            },
            'most_influential_feature': most_influential
        }
        
        print("✅ Analysis 1 complete!")
        
        return results
    
    def _plot_saliency_heatmap(self, saliency_avg: np.ndarray, feature_importance: np.ndarray):
        """Plot saliency heatmap showing gradient magnitude over time."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        
        # Heatmap: Feature saliency over time
        ax1 = axes[0]
        im = ax1.imshow(
            saliency_avg.T,
            aspect='auto',
            cmap='YlOrRd',
            interpolation='nearest'
        )
        ax1.set_xlabel('Timestep', fontsize=11)
        ax1.set_ylabel('Input Feature', fontsize=11)
        ax1.set_yticks(range(len(self.input_features)))
        ax1.set_yticklabels(self.input_features)
        ax1.set_title('Neural Saliency: Gradient Magnitude (Energy ← Inputs)', 
                      fontsize=13, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('|Gradient|', fontsize=10)
        
        # Bar plot: Overall feature importance
        ax2 = axes[1]
        colors = plt.cm.YlOrRd(feature_importance / feature_importance.max())
        bars = ax2.barh(self.input_features, feature_importance, color=colors)
        ax2.set_xlabel('Average Gradient Magnitude', fontsize=11)
        ax2.set_title('Feature Importance (Averaged Over Time)', fontsize=11)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add values on bars
        for i, (bar, val) in enumerate(zip(bars, feature_importance)):
            ax2.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'analysis_1_neural_saliency.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Saved: {save_path}")
        plt.close()
    
    def analysis_2_temporal_sensitivity(self, perturbation_magnitude: float = 0.10) -> Dict:
        """
        Analysis 2: Temporal Sensitivity (Time-Lag Analysis)
        
        Applies a perturbation to one input feature at t=0 and tracks how
        the energy prediction changes over time.
        
        Args:
            perturbation_magnitude: Magnitude of perturbation (default 10%)
        
        Returns:
            Dictionary with temporal sensitivity results
        """
        print("\n" + "="*70)
        print("ANALYSIS 2: TEMPORAL SENSITIVITY (Time-Lag Analysis)")
        print("="*70)
        
        # Get single baseline sequence
        baseline, raw_baseline = self.get_test_batch(sequence_length=30, num_sequences=1)
        
        # Test perturbation on fuel_flow (index 0)
        perturbed_feature = 'fuel_flow'
        perturb_idx = self.input_features.index(perturbed_feature)
        
        print(f"\nPerturbation: {perturbed_feature} +{perturbation_magnitude*100:.0f}% at t=0")
        
        # Create perturbed sequence
        perturbed = baseline.clone()
        perturbed[0, 0, perturb_idx] *= (1 + perturbation_magnitude)
        
        # Forward pass for both
        self.model.eval()
        with torch.no_grad():
            _, energy_baseline, _ = self.model(baseline)
            _, energy_perturbed, _ = self.model(perturbed)
        
        # Calculate difference
        energy_diff = (energy_perturbed - energy_baseline).squeeze().cpu().numpy()
        
        # Denormalize for interpretability
        baseline_np = energy_baseline.squeeze().cpu().numpy()
        perturbed_np = energy_perturbed.squeeze().cpu().numpy()
        
        # Compute metrics
        max_diff = np.max(np.abs(energy_diff))
        max_diff_time = np.argmax(np.abs(energy_diff))
        mean_diff = np.mean(np.abs(energy_diff))
        
        # Determine system behavior
        if max_diff_time <= 5:
            behavior = "Reactive (responds quickly)"
        else:
            behavior = "Inertial (delayed response)"
        
        print("\nResults:")
        print(f"  Max energy difference: {max_diff:.6f}")
        print(f"  Time of max difference: t={max_diff_time}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")
        print(f"  System behavior: {behavior}")
        
        # Visualize
        self._plot_temporal_sensitivity(
            energy_baseline.squeeze().cpu().numpy(),
            energy_perturbed.squeeze().cpu().numpy(),
            energy_diff,
            perturbed_feature,
            perturbation_magnitude
        )
        
        results = {
            'perturbed_feature': perturbed_feature,
            'perturbation_magnitude': perturbation_magnitude,
            'max_energy_difference': float(max_diff),
            'max_difference_timestep': int(max_diff_time),
            'mean_absolute_difference': float(mean_diff),
            'system_behavior': behavior,
            'energy_baseline': baseline_np.tolist(),
            'energy_perturbed': perturbed_np.tolist(),
            'energy_difference': energy_diff.tolist()
        }
        
        print("✅ Analysis 2 complete!")
        
        return results
    
    def _plot_temporal_sensitivity(
        self,
        baseline: np.ndarray,
        perturbed: np.ndarray,
        difference: np.ndarray,
        feature_name: str,
        perturbation_mag: float
    ):
        """Plot temporal sensitivity analysis results."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        timesteps = np.arange(len(baseline))
        
        # Top: Energy predictions
        ax1 = axes[0]
        ax1.plot(timesteps, baseline, 'b-', linewidth=2, label='Baseline', marker='o', markersize=4)
        ax1.plot(timesteps, perturbed, 'r--', linewidth=2, label='Perturbed', marker='s', markersize=4)
        ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='Perturbation at t=0')
        ax1.set_ylabel('Energy Prediction (scaled)', fontsize=11)
        ax1.set_title(f'Temporal Sensitivity: {feature_name} +{perturbation_mag*100:.0f}%', 
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(alpha=0.3)
        
        # Bottom: Difference
        ax2 = axes[1]
        colors = ['green' if d < 0 else 'red' for d in difference]
        ax2.bar(timesteps, difference, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Timestep', fontsize=11)
        ax2.set_ylabel('Energy Difference\n(Perturbed - Baseline)', fontsize=11)
        ax2.set_title('Response to Perturbation Over Time', fontsize=11)
        ax2.grid(alpha=0.3, axis='y')
        
        # Annotate max difference
        max_idx = np.argmax(np.abs(difference))
        ax2.annotate(
            f'Max: {difference[max_idx]:.4f}\nat t={max_idx}',
            xy=(max_idx, difference[max_idx]),
            xytext=(max_idx + 3, difference[max_idx] * 1.3),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7)
        )
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'analysis_2_temporal_sensitivity.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Saved: {save_path}")
        plt.close()
    
    def analysis_3_internal_gating(self) -> Dict:
        """
        Analysis 3: Internal Gating Analysis (CfC Interpretability)
        
        Extracts and analyzes hidden states from the CfC layer to understand
        internal dynamics and gating behavior.
        
        Returns:
            Dictionary with internal gating analysis results
        """
        print("\n" + "="*70)
        print("ANALYSIS 3: INTERNAL GATING ANALYSIS (CfC Interpretability)")
        print("="*70)
        
        # Get test batch
        inputs, _ = self.get_test_batch(sequence_length=30, num_sequences=1)
        
        # Forward pass and extract hidden states
        self.model.eval()
        with torch.no_grad():
            # Get hidden state sequence from CfC body
            hidden_seq, last_hidden = self.model.cfc_body(inputs)
        
        # Extract hidden states: shape (1, seq_len, hidden_size)
        hidden_states = hidden_seq.squeeze(0).cpu().numpy()  # (seq_len, hidden_size)
        
        print(f"\nHidden state shape: {hidden_states.shape}")
        print(f"  Sequence length: {hidden_states.shape[0]}")
        print(f"  Hidden size: {hidden_states.shape[1]}")
        
        # Compute statistics
        activation_magnitudes = np.linalg.norm(hidden_states, axis=1)  # L2 norm per timestep
        velocities = np.diff(hidden_states, axis=0)  # Rate of change
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        # Identify fast vs slow units
        unit_variances = np.var(hidden_states, axis=0)
        fast_units = np.argsort(unit_variances)[-10:]  # Top 10 most dynamic
        slow_units = np.argsort(unit_variances)[:10]   # Top 10 most stable
        
        print("\nDynamics Summary:")
        print(f"  Mean activation magnitude: {activation_magnitudes.mean():.6f}")
        print(f"  Max activation magnitude: {activation_magnitudes.max():.6f}")
        print(f"  Mean velocity (rate of change): {velocity_magnitudes.mean():.6f}")
        print(f"  Max velocity: {velocity_magnitudes.max():.6f}")
        print(f"\n  Fast units (high variance): {list(fast_units[:5])}")
        print(f"  Slow units (low variance): {list(slow_units[:5])}")
        
        # Determine dynamics type
        if velocity_magnitudes.mean() > 0.1:
            dynamics_type = "Fast Dynamics (rapid state changes)"
        else:
            dynamics_type = "Slow Dynamics (gradual state changes)"
        
        print(f"\n  Overall dynamics: {dynamics_type}")
        
        # Visualize
        self._plot_internal_gating(
            hidden_states,
            activation_magnitudes,
            velocity_magnitudes,
            fast_units,
            slow_units
        )
        
        results = {
            'hidden_size': int(hidden_states.shape[1]),
            'sequence_length': int(hidden_states.shape[0]),
            'mean_activation_magnitude': float(activation_magnitudes.mean()),
            'max_activation_magnitude': float(activation_magnitudes.max()),
            'mean_velocity': float(velocity_magnitudes.mean()),
            'max_velocity': float(velocity_magnitudes.max()),
            'fast_units': fast_units.tolist(),
            'slow_units': slow_units.tolist(),
            'dynamics_type': dynamics_type
        }
        
        print("✅ Analysis 3 complete!")
        
        return results
    
    def _plot_internal_gating(
        self,
        hidden_states: np.ndarray,
        activation_mags: np.ndarray,
        velocity_mags: np.ndarray,
        fast_units: np.ndarray,
        slow_units: np.ndarray
    ):
        """Plot internal gating analysis results."""
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Hidden state heatmap (first 50 units)
        ax1 = fig.add_subplot(gs[0, :])
        num_units_to_plot = min(50, hidden_states.shape[1])
        im = ax1.imshow(
            hidden_states[:, :num_units_to_plot].T,
            aspect='auto',
            cmap='RdBu_r',
            interpolation='nearest',
            vmin=-2,
            vmax=2
        )
        ax1.set_xlabel('Timestep', fontsize=10)
        ax1.set_ylabel('Hidden Unit', fontsize=10)
        ax1.set_title(f'Hidden State Heatmap (First {num_units_to_plot} Units)', 
                     fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='Activation')
        
        # 2. Activation magnitude over time
        ax2 = fig.add_subplot(gs[1, 0])
        timesteps = np.arange(len(activation_mags))
        ax2.plot(timesteps, activation_mags, 'b-', linewidth=2, marker='o', markersize=3)
        ax2.fill_between(timesteps, 0, activation_mags, alpha=0.3)
        ax2.set_xlabel('Timestep', fontsize=10)
        ax2.set_ylabel('L2 Norm', fontsize=10)
        ax2.set_title('Activation Magnitude Over Time', fontsize=11)
        ax2.grid(alpha=0.3)
        
        # 3. Velocity (rate of change) over time
        ax3 = fig.add_subplot(gs[1, 1])
        velocity_timesteps = np.arange(len(velocity_mags))
        ax3.plot(velocity_timesteps, velocity_mags, 'r-', linewidth=2, marker='s', markersize=3)
        ax3.fill_between(velocity_timesteps, 0, velocity_mags, alpha=0.3, color='red')
        ax3.set_xlabel('Timestep', fontsize=10)
        ax3.set_ylabel('Velocity (L2 Norm)', fontsize=10)
        ax3.set_title('Rate of Change Over Time', fontsize=11)
        ax3.grid(alpha=0.3)
        
        # 4. Distribution of activations
        ax4 = fig.add_subplot(gs[2, :])
        all_activations = hidden_states.flatten()
        ax4.hist(all_activations, bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax4.set_xlabel('Activation Value', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.set_title('Distribution of Hidden State Activations', fontsize=11)
        ax4.legend()
        ax4.grid(alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = (
            f"Mean: {all_activations.mean():.4f}\n"
            f"Std: {all_activations.std():.4f}\n"
            f"Min: {all_activations.min():.4f}\n"
            f"Max: {all_activations.max():.4f}"
        )
        ax4.text(
            0.98, 0.97, stats_text,
            transform=ax4.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.suptitle(
            'Internal Gating Analysis: CfC Hidden State Dynamics',
            fontsize=14,
            fontweight='bold',
            y=0.995
        )
        
        save_path = os.path.join(self.output_dir, 'analysis_3_internal_gating.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Saved: {save_path}")
        plt.close()
    
    def run_all_analyses(self) -> Dict:
        """
        Run all three causality analyses.
        
        Returns:
            Dictionary containing results from all analyses
        """
        print("\n" + "="*70)
        print(" COMPREHENSIVE CAUSALITY ANALYSIS FOR EBLNN")
        print("="*70)
        print(f"\nModel: {self.model_path}")
        print(f"Data: {self.data_path}")
        print(f"Output: {self.output_dir}")
        
        # Run analyses
        results = {}
        
        try:
            results['analysis_1_neural_saliency'] = self.analysis_1_neural_saliency()
        except Exception as e:
            print(f"❌ Analysis 1 failed: {e}")
            results['analysis_1_neural_saliency'] = {'error': str(e)}
        
        try:
            results['analysis_2_temporal_sensitivity'] = self.analysis_2_temporal_sensitivity()
        except Exception as e:
            print(f"❌ Analysis 2 failed: {e}")
            results['analysis_2_temporal_sensitivity'] = {'error': str(e)}
        
        try:
            results['analysis_3_internal_gating'] = self.analysis_3_internal_gating()
        except Exception as e:
            print(f"❌ Analysis 3 failed: {e}")
            results['analysis_3_internal_gating'] = {'error': str(e)}
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'analysis_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*70)
        print(" ANALYSIS COMPLETE")
        print("="*70)
        print(f"\n📄 Summary saved: {summary_path}")
        print(f"📊 Visualizations saved in: {self.output_dir}")
        
        return results


def main():
    """Main entry point for causality analysis."""
    parser = argparse.ArgumentParser(
        description='Causality Analysis for EBLNN Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all analyses
  python analyze_causality.py --model results/models/best_model.pth
  
  # Run specific analysis
  python analyze_causality.py --model results/models/best_model.pth --analysis saliency
  
  # Specify output directory
  python analyze_causality.py --model results/models/best_model.pth --output my_analysis
  
  # Use GPU
  python analyze_causality.py --model results/models/best_model.pth --device cuda
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pth file)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/synthetic_temperature_data.csv',
        help='Path to data CSV file (default: data/synthetic_temperature_data.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/causality_analysis',
        help='Output directory for analysis results (default: results/causality_analysis)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run analysis on (default: cpu)'
    )
    
    parser.add_argument(
        '--analysis',
        type=str,
        default='all',
        choices=['all', 'saliency', 'temporal', 'gating'],
        help='Which analysis to run (default: all)'
    )
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return 1
    
    # Validate data path
    if not os.path.exists(args.data):
        print(f"❌ Error: Data file not found: {args.data}")
        return 1
    
    # Create analyzer
    try:
        analyzer = CausalityAnalyzer(
            model_path=args.model,
            data_path=args.data,
            device=args.device,
            output_dir=args.output
        )
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        return 1
    
    # Run analyses
    try:
        if args.analysis == 'all':
            analyzer.run_all_analyses()
        elif args.analysis == 'saliency':
            results = analyzer.analysis_1_neural_saliency()
            summary_path = os.path.join(args.output, 'analysis_summary.json')
            with open(summary_path, 'w') as f:
                json.dump({'analysis_1_neural_saliency': results}, f, indent=2)
            print(f"\n📄 Summary saved: {summary_path}")
        elif args.analysis == 'temporal':
            results = analyzer.analysis_2_temporal_sensitivity()
            summary_path = os.path.join(args.output, 'analysis_summary.json')
            with open(summary_path, 'w') as f:
                json.dump({'analysis_2_temporal_sensitivity': results}, f, indent=2)
            print(f"\n📄 Summary saved: {summary_path}")
        elif args.analysis == 'gating':
            results = analyzer.analysis_3_internal_gating()
            summary_path = os.path.join(args.output, 'analysis_summary.json')
            with open(summary_path, 'w') as f:
                json.dump({'analysis_3_internal_gating': results}, f, indent=2)
            print(f"\n📄 Summary saved: {summary_path}")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n✅ All operations completed successfully!")
    return 0


if __name__ == '__main__':
    exit(main())
