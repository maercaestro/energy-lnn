"""
WandB Sweep Runner
Orchestrates multiple experiments with different hyperparameter combinations
"""

import os
import sys
import yaml
import wandb
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from run_single_experiment import load_config, merge_configs, run_experiment


def run_sweep_agent():
    """Run a single experiment as part of a WandB sweep."""
    # Initialize WandB run (will get config from sweep)
    run = wandb.init()
    
    # Load base configuration
    config_path = os.path.join(project_root, 'config/base_config.yaml')
    base_config = load_config(config_path)
    
    # Merge with sweep parameters
    sweep_params = dict(wandb.config)
    config = merge_configs(base_config, sweep_params)
    
    # Update WandB config with full configuration
    wandb.config.update(config, allow_val_change=True)
    
    # Run experiment (don't call wandb.init again, it's already initialized)
    print(f"\n{'='*70}")
    print(f"Running experiment with configuration:")
    print(f"  Alpha: {config['training']['alpha']}")
    print(f"  Hidden Size: {config['model']['hidden_size']}")
    print(f"  W_Safety: {config['training']['w_safety']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"{'='*70}\n")
    
    # Note: We pass use_wandb=False because wandb.init() is already called
    # But we manually set use_wandb=True in the trainer
    from src.data_generation import load_or_generate_data
    from src.model import create_model
    from src.train import EBLNNTrainer
    from src.utils import log_plots_to_wandb
    import torch
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config['paths']['results'], exist_ok=True)
    os.makedirs(config['paths']['models'], exist_ok=True)
    os.makedirs(config['paths']['plots'], exist_ok=True)
    
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
        use_wandb=True
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
    log_plots_to_wandb(
        history=trainer.history,
        predictions=trainer.test_predictions,
        model=model,
        input_scaler=trainer.input_scaler,
        target_scaler=trainer.target_scaler,
        device=device
    )
    
    print("\n✅ Sweep experiment complete!")


def create_and_run_sweep():
    """Create a WandB sweep and run agents."""
    # Load sweep configuration
    sweep_config_path = os.path.join(project_root, 'config/sweep_config.yaml')
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Load base config to get project name
    base_config_path = os.path.join(project_root, 'config/base_config.yaml')
    base_config = load_config(base_config_path)
    
    project_name = base_config['wandb']['project']
    entity = base_config['wandb'].get('entity')
    
    # Create sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project=project_name,
        entity=entity
    )
    
    print(f"\n{'='*70}")
    print(f"Created WandB sweep: {sweep_id}")
    print(f"Project: {project_name}")
    print(f"{'='*70}\n")
    print(f"Sweep configuration:")
    print(f"  Method: {sweep_config['method']}")
    print(f"  Parameters to sweep:")
    for param, values in sweep_config['parameters'].items():
        print(f"    - {param}: {values.get('values', values)}")
    print(f"{'='*70}\n")
    
    # Calculate total experiments
    alpha_count = len(sweep_config['parameters']['alpha']['values'])
    hidden_count = len(sweep_config['parameters']['hidden_size']['values'])
    safety_count = len(sweep_config['parameters']['w_safety']['values'])
    lr_count = len(sweep_config['parameters']['learning_rate']['values'])
    total_experiments = alpha_count * hidden_count * safety_count * lr_count
    
    print(f"Total experiments in grid sweep: {total_experiments}")
    print(f"  (Alpha: {alpha_count} × Hidden: {hidden_count} × Safety: {safety_count} × LR: {lr_count})")
    print(f"\nStarting sweep agents...\n")
    
    # Run sweep agent
    # count=None means run all experiments in the sweep
    wandb.agent(sweep_id, function=run_sweep_agent, count=None, project=project_name, entity=entity)
    
    print(f"\n{'='*70}")
    print(f"✅ All sweep experiments complete!")
    print(f"View results at: https://wandb.ai/{entity or 'your-entity'}/{project_name}/sweeps/{sweep_id}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run WandB hyperparameter sweep')
    parser.add_argument(
        '--count',
        type=int,
        default=None,
        help='Number of experiments to run (default: all)'
    )
    
    args = parser.parse_args()
    
    if args.count:
        print(f"Running {args.count} experiments from sweep...")
        # If count is specified, we need to modify the agent call
        # For now, just run the full sweep creation
        create_and_run_sweep()
    else:
        print("Running full hyperparameter sweep...")
        create_and_run_sweep()
