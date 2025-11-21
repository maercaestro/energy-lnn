"""
Example Usage: Causality Analysis for EBLNN

This script demonstrates how to use the CausalityAnalyzer for model interpretation.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.analyze_causality import CausalityAnalyzer


def example_basic_all_analyses():
    """Example 1: Run all analyses with default settings."""
    print("=" * 80)
    print("EXAMPLE 1: Run All Causality Analyses")
    print("=" * 80 + "\n")
    
    analyzer = CausalityAnalyzer(
        model_path='results/models/eblnn_best_model.pth',
        data_path='data/synthetic_temperature_data.csv',
        device='cpu',
        output_dir='results/causality_analysis'
    )
    
    # Run all three analyses
    results = analyzer.run_all_analyses()
    
    print("\n✅ Complete analysis finished!")
    print(f"📁 Results saved to: results/causality_analysis/")
    
    return results


def example_individual_analyses():
    """Example 2: Run individual analyses."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Run Individual Analyses")
    print("=" * 80 + "\n")
    
    analyzer = CausalityAnalyzer(
        model_path='results/models/eblnn_best_model.pth',
        data_path='data/synthetic_temperature_data.csv'
    )
    
    # Analysis 1: Neural Saliency
    print("🔍 Analysis 1: Neural Saliency")
    saliency = analyzer.analysis_1_neural_saliency(
        num_samples=32,
        save_path='results/saliency.png'
    )
    print(f"   Feature importance computed for {len(analyzer.input_features)} features\n")
    
    # Analysis 2: Temporal Sensitivity
    print("🔍 Analysis 2: Temporal Sensitivity")
    temporal = analyzer.analysis_2_temporal_sensitivity(
        perturbation_feature='fuel_flow',
        perturbation_magnitude=0.1,
        perturbation_timestep=0,
        save_path='results/temporal.png'
    )
    print(f"   Perturbation impact analyzed over time\n")
    
    # Analysis 3: Internal Gating
    print("🔍 Analysis 3: Internal Gating")
    gating = analyzer.analysis_3_internal_gating(
        num_samples=8,
        save_path='results/gating.png'
    )
    print(f"   Hidden state dynamics analyzed\n")
    
    return saliency, temporal, gating


def example_compare_features():
    """Example 3: Compare temporal sensitivity across features."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Compare Feature Perturbations")
    print("=" * 80 + "\n")
    
    analyzer = CausalityAnalyzer(
        model_path='results/models/eblnn_best_model.pth',
        data_path='data/synthetic_temperature_data.csv'
    )
    
    features_to_test = ['fuel_flow', 'air_fuel_ratio', 'inflow_rate']
    results = {}
    
    for feature in features_to_test:
        print(f"🔬 Testing: {feature}")
        result = analyzer.analysis_2_temporal_sensitivity(
            perturbation_feature=feature,
            perturbation_magnitude=0.1,
            perturbation_timestep=0,
            save_path=f'results/temporal_{feature}.png'
        )
        
        max_impact = abs(result['energy_difference']).max()
        cumulative_impact = abs(result['energy_difference']).sum()
        
        results[feature] = {
            'max_impact': max_impact,
            'cumulative_impact': cumulative_impact
        }
        
        print(f"   Max impact: {max_impact:.6f}")
        print(f"   Cumulative impact: {cumulative_impact:.6f}\n")
    
    # Find most influential feature
    most_influential = max(results.items(), key=lambda x: x[1]['max_impact'])
    print(f"🏆 Most influential feature: {most_influential[0]}")
    print(f"   Impact: {most_influential[1]['max_impact']:.6f}\n")
    
    return results


def example_custom_perturbation():
    """Example 4: Custom perturbation analysis."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Custom Perturbation Parameters")
    print("=" * 80 + "\n")
    
    analyzer = CausalityAnalyzer(
        model_path='results/models/eblnn_best_model.pth',
        data_path='data/synthetic_temperature_data.csv'
    )
    
    # Test different perturbation magnitudes
    magnitudes = [0.05, 0.10, 0.20]  # 5%, 10%, 20%
    
    print("Testing different perturbation magnitudes:")
    for mag in magnitudes:
        result = analyzer.analysis_2_temporal_sensitivity(
            perturbation_feature='fuel_flow',
            perturbation_magnitude=mag,
            perturbation_timestep=5,
            save_path=f'results/temporal_mag_{int(mag*100)}.png'
        )
        
        max_impact = abs(result['energy_difference']).max()
        print(f"  {mag*100:>5.1f}% perturbation → Max impact: {max_impact:.6f}")
    
    print()


def example_analyze_best_from_sweep():
    """Example 5: Analyze best model from sweep results."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Analyze Best Model from Sweep")
    print("=" * 80 + "\n")
    
    # After your 20-run sweep completes, analyze the best model
    analyzer = CausalityAnalyzer(
        model_path='results/models/eblnn_best_model.pth',
        data_path='data/synthetic_temperature_data.csv',
        output_dir='results/causality_best_model'
    )
    
    # Full analysis
    results = analyzer.run_all_analyses()
    
    # Print key insights
    print("\n" + "=" * 80)
    print("📊 KEY INSIGHTS")
    print("=" * 80)
    
    print("\n🔍 Feature Importance (from Neural Saliency):")
    saliency = results['saliency']
    for i, feature in enumerate(analyzer.input_features):
        importance = saliency['feature_importance'][i]
        bar = "█" * int(importance / saliency['feature_importance'].max() * 50)
        print(f"  {feature:20s} {importance:.6f} {bar}")
    
    print("\n⏱️  Temporal Dynamics (from Temporal Sensitivity):")
    temporal = results['temporal']
    print(f"  Max energy impact: {abs(temporal['energy_difference']).max():.6f}")
    print(f"  System behavior: {temporal['system_behavior']}")
    
    print("\n🧠 Internal Dynamics (from Gating Analysis):")
    gating = results['gating']
    print(f"  Avg activation magnitude: {gating['avg_activation_magnitude']:.6f}")
    print(f"  Avg velocity: {gating['avg_velocity']:.6f}")
    print(f"  Dynamics type: {gating['dynamics_type']}")
    
    print("\n" + "=" * 80)
    print(f"✅ Complete analysis saved to: {analyzer.output_dir}/")
    print("=" * 80 + "\n")
    
    return results


if __name__ == '__main__':
    import os
    
    # Change to project root
    os.chdir(project_root)
    
    # Run examples (uncomment the ones you want to try)
    
    # Example 1: Run all analyses
    results = example_basic_all_analyses()
    
    # Example 2: Run individual analyses
    # saliency, temporal, gating = example_individual_analyses()
    
    # Example 3: Compare feature perturbations
    # feature_comparison = example_compare_features()
    
    # Example 4: Custom perturbation parameters
    # example_custom_perturbation()
    
    # Example 5: Analyze best model from sweep
    # best_model_results = example_analyze_best_from_sweep()
