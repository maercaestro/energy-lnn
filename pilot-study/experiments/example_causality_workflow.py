"""
Example Usage: Causality Analysis Workflow

This script demonstrates a complete workflow for training and analyzing
an EBLNN model, including all three causality analyses.

Run this after training your model to understand causal relationships.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.analyze_causality import CausalityAnalyzer


def main():
    """
    Complete workflow example for causality analysis.
    """
    print("="*70)
    print(" EBLNN CAUSALITY ANALYSIS - EXAMPLE WORKFLOW")
    print("="*70)
    
    # Configuration
    model_path = "results/models/best_model.pth"
    data_path = "data/synthetic_temperature_data.csv"
    output_dir = "results/causality_analysis"
    
    # Check if files exist
    print("\n1. Checking prerequisites...")
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Please train a model first:")
        print("   python experiments/run_single_experiment.py")
        return 1
    
    if not os.path.exists(data_path):
        print(f"❌ Data not found: {data_path}")
        print("   Data will be generated on first run.")
        return 1
    
    print("✅ All prerequisites found")
    
    # Create analyzer
    print("\n2. Initializing analyzer...")
    analyzer = CausalityAnalyzer(
        model_path=model_path,
        data_path=data_path,
        device='cpu',  # Change to 'cuda' if GPU available
        output_dir=output_dir
    )
    
    # Run individual analyses with custom parameters
    print("\n3. Running analyses...")
    
    # Analysis 1: Neural Saliency
    print("\n   → Running Analysis 1: Neural Saliency...")
    saliency_results = analyzer.analysis_1_neural_saliency()
    
    # Extract key insight
    most_influential = saliency_results['most_influential_feature']
    feature_scores = saliency_results['feature_importance']
    
    print(f"\n   Key Insight from Analysis 1:")
    print(f"   - Most influential feature: {most_influential}")
    print(f"   - Feature scores:")
    for feature, score in feature_scores.items():
        stars = '█' * int(score * 20000)  # Scale for visualization
        print(f"     {feature:20s}: {score:.6f} {stars}")
    
    # Analysis 2: Temporal Sensitivity
    print("\n   → Running Analysis 2: Temporal Sensitivity...")
    temporal_results = analyzer.analysis_2_temporal_sensitivity(
        perturbation_magnitude=0.10  # 10% perturbation
    )
    
    print(f"\n   Key Insight from Analysis 2:")
    print(f"   - System behavior: {temporal_results['system_behavior']}")
    print(f"   - Max impact at timestep: {temporal_results['max_difference_timestep']}")
    print(f"   - Max energy difference: {temporal_results['max_energy_difference']:.6f}")
    
    # Analysis 3: Internal Gating
    print("\n   → Running Analysis 3: Internal Gating...")
    gating_results = analyzer.analysis_3_internal_gating()
    
    print(f"\n   Key Insight from Analysis 3:")
    print(f"   - Dynamics type: {gating_results['dynamics_type']}")
    print(f"   - Mean velocity: {gating_results['mean_velocity']:.6f}")
    print(f"   - Fast units: {len(gating_results['fast_units'])} units")
    print(f"   - Slow units: {len(gating_results['slow_units'])} units")
    
    # Synthesize insights
    print("\n" + "="*70)
    print(" SYNTHESIZED INSIGHTS")
    print("="*70)
    
    print("\n1. CONTROL STRATEGY:")
    print(f"   → Focus on optimizing: {most_influential}")
    print(f"   → This feature has the strongest causal effect on energy cost")
    
    print("\n2. SYSTEM DYNAMICS:")
    behavior = temporal_results['system_behavior']
    lag = temporal_results['max_difference_timestep']
    if 'Reactive' in behavior:
        print(f"   → System responds quickly (lag={lag} steps)")
        print(f"   → Use reactive control strategies")
    else:
        print(f"   → System has inertia (lag={lag} steps)")
        print(f"   → Consider Model Predictive Control (MPC) with {lag}+ step horizon")
    
    print("\n3. MODEL CAPACITY:")
    if gating_results['mean_velocity'] > 0.3:
        print(f"   → Model learns fast dynamics well")
        print(f"   → Current architecture is appropriate")
    else:
        print(f"   → Model dynamics are slow")
        print(f"   → Consider increasing hidden size or checking data quality")
    
    print("\n" + "="*70)
    print(" RECOMMENDATIONS")
    print("="*70)
    
    # Generate recommendations based on results
    recommendations = []
    
    # Recommendation 1: Feature prioritization
    top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    recommendations.append(
        f"Prioritize control of {top_features[0][0]} and {top_features[1][0]} "
        f"to minimize energy costs"
    )
    
    # Recommendation 2: Control horizon
    if lag > 5:
        recommendations.append(
            f"Use predictive control with at least {lag} timestep lookahead "
            f"to account for system inertia"
        )
    
    # Recommendation 3: Model adequacy
    if gating_results['mean_velocity'] > 0.3:
        recommendations.append(
            "Model architecture is adequate for this problem; "
            "focus on hyperparameter tuning rather than capacity"
        )
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")
    
    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)
    
    print("\n1. Review visualizations:")
    print(f"   - {output_dir}/analysis_1_neural_saliency.png")
    print(f"   - {output_dir}/analysis_2_temporal_sensitivity.png")
    print(f"   - {output_dir}/analysis_3_internal_gating.png")
    
    print("\n2. Export results:")
    print(f"   - {output_dir}/analysis_summary.json")
    
    print("\n3. Implement control strategy:")
    print(f"   - Based on feature importance: {most_influential}")
    print(f"   - Account for system lag: {lag} timesteps")
    
    print("\n4. Iterate:")
    print("   - Adjust hyperparameters if needed")
    print("   - Re-run causality analysis after changes")
    print("   - Compare results across different models")
    
    print("\n✅ Workflow complete!")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    exit(main())
