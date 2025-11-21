"""
Run Best Configuration with Causality Analysis

This script:
1. Trains EBLNN with best hyperparameters from sweep
2. Runs three causality analyses on trained model
3. Logs everything to WandB (metrics, plots, analysis results)

Best Configuration from Sweep:
- alpha: 0.5
- hidden_size: 256
- w_safety: 100
- learning_rate: 0.0005

Author: Energy-LNN Research Team
Date: 2025-11-21
"""

import os
import sys
import json
import yaml
import torch
import wandb
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_generation import load_or_generate_data
from src.model import EBLNN
from src.train import EBLNNTrainer
from src.utils import log_plots_to_wandb
from experiments.analyze_causality import CausalityAnalyzer


def run_best_config_with_analysis():
    """
    Train model with best config and run causality analysis.
    """
    print("=" * 80)
    print("🚀 Training EBLNN with Best Configuration + Causality Analysis")
    print("=" * 80)
    
    # Load base config
    config_path = os.path.join(project_root, 'config/base_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update with best hyperparameters from sweep
    config['training']['alpha'] = 0.5
    config['model']['hidden_size'] = 256
    config['training']['w_safety'] = 100.0
    config['training']['learning_rate'] = 0.0005
    
    # Use more epochs since this is final training
    config['training']['epochs'] = 200
    config['training']['early_stopping'] = True
    config['training']['patience'] = 15
    
    print("\n📋 Best Configuration:")
    print(f"   Alpha (α): {config['training']['alpha']}")
    print(f"   Hidden Size: {config['model']['hidden_size']}")
    print(f"   W_Safety: {config['training']['w_safety']}")
    print(f"   Learning Rate: {config['training']['learning_rate']}")
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   Early Stopping: {config['training']['early_stopping']}")
    
    # Initialize WandB
    run = wandb.init(
        project=config['wandb']['project'],
        name='best-config-with-analysis',
        tags=['best-model', 'causality-analysis', 'final-training'],
        notes='Training with best config from sweep + comprehensive causality analysis',
        config=config
    )
    
    print(f"\n✅ WandB initialized: {run.url}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Create output directories
    os.makedirs(config['paths']['results'], exist_ok=True)
    os.makedirs(config['paths']['models'], exist_ok=True)
    os.makedirs(config['paths']['plots'], exist_ok=True)
    
    # ============================================================================
    # STEP 1: TRAIN MODEL
    # ============================================================================
    print("\n" + "=" * 80)
    print("📚 STEP 1: Training Model with Best Configuration")
    print("=" * 80)
    
    # Load or generate data
    print("\n📊 Loading data...")
    data = load_or_generate_data(
        data_path=config['data']['data_path'],
        num_sequences=config['data']['num_sequences'],
        sequence_length=config['data']['sequence_length'],
        seed=config['data']['seed'],
        force_regenerate=config['data']['force_regenerate']
    )
    print(f"✅ Data loaded: {len(data)} samples")
    
    # Create model
    print(f"\n🏗️  Building model...")
    model = EBLNN(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        output_size_prediction=2,
        output_size_energy=1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model created:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = EBLNNTrainer(
        model=model,
        config=config,
        device=device,
        use_wandb=True
    )
    
    # Prepare data loaders
    print("\n🔄 Preparing data loaders...")
    train_loader, val_loader, test_loader = trainer.prepare_data(data)
    print(f"✅ Data loaders ready:")
    print(f"   Train: {len(train_loader)} batches")
    print(f"   Validation: {len(val_loader)} batches")
    print(f"   Test: {len(test_loader)} batches")
    
    # Train model
    print("\n🎓 Starting training...")
    print("=" * 80)
    
    try:
        trainer.train(
            train_loader,
            val_loader,
            save_path=config['paths']['models']
        )
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish(exit_code=1)
        raise
    
    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print("=" * 80)
    
    # Verify models were saved
    best_model_path = os.path.join(config['paths']['models'], 'eblnn_best_model.pth')
    last_model_path = os.path.join(config['paths']['models'], 'eblnn_last_model.pth')
    
    print("\n🔍 Checking saved models:")
    if os.path.exists(best_model_path):
        print(f"   ✅ Best model: {best_model_path}")
    else:
        print(f"   ❌ Best model not found: {best_model_path}")
    
    if os.path.exists(last_model_path):
        print(f"   ✅ Last model: {last_model_path}")
    else:
        print(f"   ❌ Last model not found: {last_model_path}")
    
    if not os.path.exists(best_model_path) and not os.path.exists(last_model_path):
        print("\n❌ ERROR: No models were saved during training!")
        print("   This likely means training failed silently.")
        print("   Check the trainer code for issues.")
        wandb.finish(exit_code=1)
        return None
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    print("\n📈 Final Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"   {metric}: {value:.6f}")
        wandb.log({f'final_{metric}': value})
    
    # Generate and log training visualizations
    print("\n📸 Generating training visualizations...")
    log_plots_to_wandb(
        history=trainer.history,
        predictions=trainer.test_predictions,
        model=model,
        input_scaler=trainer.input_scaler,
        target_scaler=trainer.target_scaler,
        device=device
    )
    print("✅ Training plots logged to WandB")
    
    # ============================================================================
    # STEP 2: CAUSALITY ANALYSIS
    # ============================================================================
    print("\n" + "=" * 80)
    print("🔍 STEP 2: Causality Analysis")
    print("=" * 80)
    
    # Path to best model
    best_model_path = os.path.join(config['paths']['models'], 'eblnn_best_model.pth')
    last_model_path = os.path.join(config['paths']['models'], 'eblnn_last_model.pth')
    
    # Verify model exists
    if not os.path.exists(best_model_path) and not os.path.exists(last_model_path):
        print(f"\n❌ Error: No model found!")
        print(f"   Expected at: {best_model_path}")
        print(f"   Or at: {last_model_path}")
        print("\n⚠️  Training may have failed to save model. Check logs above.")
        wandb.finish(exit_code=1)
        return None
    
    # Prefer best model, fallback to last
    if os.path.exists(best_model_path):
        model_to_analyze = best_model_path
        print(f"\n✅ Found best model: {best_model_path}")
    else:
        model_to_analyze = last_model_path
        print(f"\n⚠️  Best model not found, using last model: {last_model_path}")
    
    print(f"📂 Analyzing model: {model_to_analyze}")
    
    # Create causality analyzer
    print("\n🔧 Initializing CausalityAnalyzer...")
    analyzer = CausalityAnalyzer(
        model_path=model_to_analyze,
        data_path=config['data']['data_path'],
        device=str(device),
        output_dir=os.path.join(config['paths']['results'], 'causality_analysis')
    )
    
    # --- Analysis 1: Neural Saliency ---
    print("\n" + "-" * 80)
    print("🔍 Analysis 1: Neural Saliency (Gradient-Based Causality)")
    print("-" * 80)
    
    saliency_results = analyzer.analysis_1_neural_saliency(
        num_samples=32,
        save_path=os.path.join(analyzer.output_dir, 'saliency_heatmap.png')
    )
    
    # Log to WandB
    wandb.log({
        'causality/saliency_heatmap': wandb.Image(
            os.path.join(analyzer.output_dir, 'saliency_heatmap.png'),
            caption='Neural Saliency: Which features drive energy cost'
        )
    })
    
    # Log feature importance as bar chart
    feature_importance_dict = dict(zip(
        analyzer.input_features,
        saliency_results['feature_importance'].tolist()
    ))
    wandb.log({'causality/feature_importance': wandb.plot.bar(
        wandb.Table(
            data=[[k, v] for k, v in feature_importance_dict.items()],
            columns=['Feature', 'Importance']
        ),
        'Feature',
        'Importance',
        title='Feature Importance from Neural Saliency'
    )})
    
    # Log individual importance values
    for feature, importance in feature_importance_dict.items():
        wandb.log({f'causality/feature_importance/{feature}': importance})
    
    print(f"✅ Neural Saliency complete and logged to WandB")
    
    # --- Analysis 2: Temporal Sensitivity ---
    print("\n" + "-" * 80)
    print("🔍 Analysis 2: Temporal Sensitivity (Time-Lag Analysis)")
    print("-" * 80)
    
    # Test multiple features
    features_to_test = ['fuel_flow', 'air_fuel_ratio', 'inflow_rate']
    temporal_results = {}
    
    for feature in features_to_test:
        print(f"\n   Testing: {feature}")
        
        result = analyzer.analysis_2_temporal_sensitivity(
            perturbation_feature=feature,
            perturbation_magnitude=0.1,
            perturbation_timestep=0,
            save_path=os.path.join(analyzer.output_dir, f'temporal_{feature}.png')
        )
        
        temporal_results[feature] = result
        
        # Log to WandB
        wandb.log({
            f'causality/temporal_{feature}': wandb.Image(
                os.path.join(analyzer.output_dir, f'temporal_{feature}.png'),
                caption=f'Temporal Sensitivity: {feature} perturbation'
            )
        })
        
        # Log metrics
        max_impact = abs(result['energy_difference']).max()
        cumulative_impact = abs(result['energy_difference']).sum()
        
        wandb.log({
            f'causality/temporal/{feature}/max_impact': max_impact,
            f'causality/temporal/{feature}/cumulative_impact': cumulative_impact,
            f'causality/temporal/{feature}/system_behavior': result['system_behavior']
        })
        
        print(f"      Max impact: {max_impact:.6f}")
        print(f"      Behavior: {result['system_behavior']}")
    
    print(f"\n✅ Temporal Sensitivity complete and logged to WandB")
    
    # --- Analysis 3: Internal Gating ---
    print("\n" + "-" * 80)
    print("🔍 Analysis 3: Internal Gating Analysis (CfC Interpretability)")
    print("-" * 80)
    
    gating_results = analyzer.analysis_3_internal_gating(
        num_samples=8,
        save_path=os.path.join(analyzer.output_dir, 'internal_gating.png')
    )
    
    # Log to WandB
    wandb.log({
        'causality/internal_gating': wandb.Image(
            os.path.join(analyzer.output_dir, 'internal_gating.png'),
            caption='Internal Gating: CfC hidden state dynamics'
        )
    })
    
    # Log metrics
    wandb.log({
        'causality/gating/avg_activation_magnitude': gating_results['avg_activation_magnitude'],
        'causality/gating/avg_velocity': gating_results['avg_velocity'],
        'causality/gating/dynamics_type': gating_results['dynamics_type']
    })
    
    print(f"✅ Internal Gating complete and logged to WandB")
    
    # ============================================================================
    # STEP 3: SAVE COMPREHENSIVE SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("💾 STEP 3: Saving Analysis Summary")
    print("=" * 80)
    
    # Create comprehensive summary
    summary = {
        'training': {
            'config': {
                'alpha': config['training']['alpha'],
                'hidden_size': config['model']['hidden_size'],
                'w_safety': config['training']['w_safety'],
                'learning_rate': config['training']['learning_rate'],
                'epochs_trained': trainer.history['epoch'][-1]
            },
            'final_metrics': test_metrics
        },
        'causality': {
            'saliency': {
                'feature_importance': feature_importance_dict,
                'most_important_feature': max(feature_importance_dict.items(), key=lambda x: x[1])[0]
            },
            'temporal': {
                feature: {
                    'max_impact': float(abs(result['energy_difference']).max()),
                    'cumulative_impact': float(abs(result['energy_difference']).sum()),
                    'system_behavior': result['system_behavior']
                }
                for feature, result in temporal_results.items()
            },
            'gating': {
                'avg_activation_magnitude': float(gating_results['avg_activation_magnitude']),
                'avg_velocity': float(gating_results['avg_velocity']),
                'dynamics_type': gating_results['dynamics_type']
            }
        },
        'model_path': best_model_path,
        'wandb_run_url': run.url
    }
    
    # Save summary
    summary_path = os.path.join(analyzer.output_dir, 'complete_analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Summary saved to: {summary_path}")
    
    # Log summary as artifact
    artifact = wandb.Artifact(
        name='best-model-analysis',
        type='analysis',
        description='Complete causality analysis of best EBLNN model'
    )
    artifact.add_file(summary_path)
    artifact.add_dir(analyzer.output_dir)
    wandb.log_artifact(artifact)
    
    print("✅ Analysis artifact uploaded to WandB")
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("🎉 COMPLETE!")
    print("=" * 80)
    
    print("\n📊 Key Results:")
    print(f"\n   Test RMSE (Energy): {test_metrics['test_rmse_energy']:.6f}")
    print(f"   Test RMSE (Temperature): {test_metrics['test_rmse_temperature']:.6f}")
    print(f"   Test RMSE (O2): {test_metrics['test_rmse_excess_o2']:.6f}")
    
    print(f"\n🔍 Causality Insights:")
    print(f"   Most important feature: {summary['causality']['saliency']['most_important_feature']}")
    print(f"   System dynamics: {gating_results['dynamics_type']}")
    
    print(f"\n📁 All results saved to:")
    print(f"   Model: {best_model_path}")
    print(f"   Analysis: {analyzer.output_dir}")
    print(f"   WandB: {run.url}")
    
    print("\n" + "=" * 80)
    
    # Finish WandB run
    wandb.finish()
    
    return summary


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train EBLNN with best config and run causality analysis'
    )
    parser.add_argument(
        '--no-train',
        action='store_true',
        help='Skip training, only run analysis on existing model'
    )
    
    args = parser.parse_args()
    
    try:
        if args.no_train:
            print("⚠️  Training skipped, running analysis only")
            # TODO: Add analysis-only mode
        else:
            summary = run_best_config_with_analysis()
            
            print("\n✅ Script completed successfully!")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        wandb.finish(exit_code=1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish(exit_code=1)
        raise
