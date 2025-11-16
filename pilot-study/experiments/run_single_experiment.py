"""
Single Experiment Runner
Runs a single EBLNN training experiment with specified configuration
"""

import os
import sys
import yaml
import torch
import wandb
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_generation import load_or_generate_data
from src.model import create_model
from src.train import EBLNNTrainer
from src.utils import log_plots_to_wandb


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: dict, sweep_config: dict) -> dict:
    """Merge base config with sweep parameters."""
    merged = base_config.copy()
    
    # Update training parameters with sweep values
    if 'alpha' in sweep_config:
        merged['training']['alpha'] = sweep_config['alpha']
    if 'hidden_size' in sweep_config:
        merged['model']['hidden_size'] = sweep_config['hidden_size']
    if 'w_safety' in sweep_config:
        merged['training']['w_safety'] = sweep_config['w_safety']
    if 'learning_rate' in sweep_config:
        merged['training']['learning_rate'] = sweep_config['learning_rate']
    
    return merged


def run_experiment(config: dict, use_wandb: bool = True):
    """
    Run a single EBLNN experiment.
    
    Args:
        config: Configuration dictionary
        use_wandb: Whether to use WandB logging
    """
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config['paths']['results'], exist_ok=True)
    os.makedirs(config['paths']['models'], exist_ok=True)
    os.makedirs(config['paths']['plots'], exist_ok=True)
    
    # Initialize WandB
    if use_wandb:
        run = wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb'].get('entity'),
            config={
                'alpha': config['training']['alpha'],
                'hidden_size': config['model']['hidden_size'],
                'w_safety': config['training']['w_safety'],
                'learning_rate': config['training']['learning_rate'],
                'batch_size': config['training']['batch_size'],
                'epochs': config['training']['epochs']
            },
            tags=config['wandb'].get('tags', []),
            notes=config['wandb'].get('notes', '')
        )
        print(f"WandB run: {run.name}")
    
    # Load or generate data
    data_path = os.path.join(project_root, config['data']['data_path'])
    df = load_or_generate_data(
        data_path,
        num_sequences=config['data']['num_sequences'],
        sequence_length=config['data']['sequence_length'],
        seed=config['data']['seed'],
        force_regenerate=config['data']['force_regenerate']
    )
    
    # Create model
    model = create_model(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        device=device
    )
    
    # Create trainer
    trainer = EBLNNTrainer(
        model=model,
        config=config['training'],
        device=device,
        use_wandb=use_wandb
    )
    
    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(
        df,
        sequence_length=config['data']['sequence_length'],
        test_size=config['split']['test_size'],
        val_size=config['split']['val_size'],
        seed=config['split']['seed']
    )
    
    # Train model
    trainer.train(
        train_loader,
        val_loader,
        save_path=config['paths']['models']
    )
    
    # Evaluate on test set
    metrics = trainer.evaluate(test_loader)
    
    # Generate and log visualizations
    if use_wandb:
        log_plots_to_wandb(
            history=trainer.history,
            predictions=trainer.test_predictions,
            model=model,
            input_scaler=trainer.input_scaler,
            target_scaler=trainer.target_scaler,
            device=device
        )
    
    # Finish WandB run
    if use_wandb:
        wandb.finish()
    
    print("\n✅ Experiment complete!")
    return metrics


def main():
    """Main entry point for single experiment."""
    parser = argparse.ArgumentParser(description='Run single EBLNN experiment')
    parser.add_argument(
        '--config',
        type=str,
        default='config/base_config.yaml',
        help='Path to base configuration file'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable WandB logging'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=None,
        help='Override alpha value'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=None,
        help='Override hidden size'
    )
    parser.add_argument(
        '--w-safety',
        type=float,
        default=None,
        help='Override w_safety value'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Override learning rate'
    )
    
    args = parser.parse_args()
    
    # Load base configuration
    config_path = os.path.join(project_root, args.config)
    config = load_config(config_path)
    
    # Override with command-line arguments
    if args.alpha is not None:
        config['training']['alpha'] = args.alpha
    if args.hidden_size is not None:
        config['model']['hidden_size'] = args.hidden_size
    if args.w_safety is not None:
        config['training']['w_safety'] = args.w_safety
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    
    # Run experiment
    run_experiment(config, use_wandb=not args.no_wandb)


if __name__ == '__main__':
    main()
