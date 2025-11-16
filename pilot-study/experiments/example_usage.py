"""
Example: How to use the EBLNN framework programmatically
This shows how to run experiments from Python code
"""

import sys
import torch
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_generation import FurnaceDataGenerator, load_or_generate_data
from src.model import create_model
from src.train import EBLNNTrainer
from src.utils import plot_loss_curves, plot_parity_plots, plot_energy_landscape


def run_custom_experiment():
    """
    Example of running a custom EBLNN experiment.
    """
    print("="*70)
    print("EBLNN Custom Experiment Example")
    print("="*70)
    
    # Configuration
    config = {
        'training': {
            'epochs': 50,  # Shorter for demo
            'batch_size': 64,
            'learning_rate': 0.001,
            'alpha': 1.0,
            'w_safety': 100.0
        },
        'model': {
            'input_size': 5,
            'hidden_size': 128
        },
        'data': {
            'num_sequences': 1000,  # Smaller dataset for demo
            'sequence_length': 30,
            'seed': 42
        }
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Step 1: Generate data
    print("\n1. Generating synthetic data...")
    generator = FurnaceDataGenerator(seed=42)
    df = generator.generate_data(
        num_sequences=config['data']['num_sequences'],
        sequence_length=config['data']['sequence_length']
    )
    print(f"   Generated {len(df)} data points")
    
    # Step 2: Create model
    print("\n2. Creating EBLNN model...")
    model = create_model(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        device=device
    )
    
    # Step 3: Create trainer
    print("\n3. Setting up trainer...")
    trainer = EBLNNTrainer(
        model=model,
        config=config['training'],
        device=device,
        use_wandb=False  # Disable WandB for this example
    )
    
    # Step 4: Prepare data
    print("\n4. Preparing data loaders...")
    train_loader, val_loader, test_loader = trainer.prepare_data(
        df,
        sequence_length=config['data']['sequence_length']
    )
    
    # Step 5: Train model
    print("\n5. Training model...")
    print("   (This will take a few minutes)")
    trainer.train(train_loader, val_loader, save_path='results/models')
    
    # Step 6: Evaluate
    print("\n6. Evaluating on test set...")
    metrics = trainer.evaluate(test_loader)
    
    # Step 7: Generate visualizations
    print("\n7. Generating visualizations...")
    
    # Loss curves
    fig1 = plot_loss_curves(trainer.history, save_path='results/plots/loss_curves_example.png')
    
    # Parity plots
    fig2 = plot_parity_plots(
        trainer.test_predictions['true_temp'],
        trainer.test_predictions['pred_temp'],
        trainer.test_predictions['true_o2'],
        trainer.test_predictions['pred_o2'],
        trainer.test_predictions['true_energy'],
        trainer.test_predictions['pred_energy'],
        save_path='results/plots/parity_plots_example.png'
    )
    
    # Energy landscape
    fig3 = plot_energy_landscape(
        model,
        trainer.input_scaler,
        trainer.target_scaler,
        device=device,
        save_path='results/plots/energy_landscape_example.png'
    )
    
    print("\n" + "="*70)
    print("✅ Example experiment complete!")
    print("="*70)
    print("\nResults:")
    print(f"  - Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"  - Test RMSE (Temperature): {metrics['test_rmse_temp']:.4f} °C")
    print(f"  - Test RMSE (O₂): {metrics['test_rmse_o2']:.4f} %")
    print(f"  - Test RMSE (Energy): {metrics['test_rmse_energy']:.4f}")
    print(f"\nPlots saved to: results/plots/")
    print(f"Model saved to: results/models/best_model.pth")
    print()


def run_quick_data_exploration():
    """
    Quick example showing data generation and statistics.
    """
    print("\n" + "="*70)
    print("Data Generation Example")
    print("="*70)
    
    # Create generator
    generator = FurnaceDataGenerator(seed=42)
    
    # Generate small dataset
    df = generator.generate_data(num_sequences=100, sequence_length=30, verbose=True)
    
    # Get statistics
    stats = generator.get_data_statistics(df)
    print("\nData Statistics:")
    print(stats.to_string(index=False))
    
    print("\n✅ Data exploration complete!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='EBLNN Examples')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['experiment', 'data'],
        default='experiment',
        help='Which example to run'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'experiment':
        run_custom_experiment()
    elif args.mode == 'data':
        run_quick_data_exploration()
