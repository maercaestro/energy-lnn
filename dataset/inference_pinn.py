"""
Inference script for trained PINN model
Loads the best model and makes predictions on new data
"""

import torch
import pickle
import pandas as pd
import numpy as np
import argparse
import sys
import os

# Import model from train_pinn
sys.path.insert(0, os.path.dirname(__file__))
from train_pinn import FurnacePINN

def load_model(checkpoint_path, scaler_path):
    """Load trained model and scaler"""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config
    config = checkpoint['config']
    
    # Reconstruct model
    model = FurnacePINN(
        input_dim=len(config['feature_cols'].split(',')) if isinstance(config.get('feature_cols'), str) else 6,
        hidden_dim=config.get('hidden_dim', 64),
        layers=config.get('layers', 4)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"✅ Model loaded from: {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Val Loss: {checkpoint['val_loss']:.6f}")
    
    # Get learned parameters
    _, k, eta = model(torch.zeros(1, model.net[0].in_features))
    print(f"   Learned Efficiency η: {eta.item():.4f}")
    print(f"   Learned Leakage k: {k.item():.4f}")
    
    return model, scaler, config

def predict(model, scaler, input_data):
    """
    Make predictions on input data
    
    Args:
        model: Trained PINN model
        scaler: Fitted StandardScaler
        input_data: DataFrame or numpy array with features
    
    Returns:
        predictions: Dictionary with temperature and O2 predictions
    """
    # Convert to numpy if DataFrame
    if isinstance(input_data, pd.DataFrame):
        X = input_data.values
    else:
        X = input_data
    
    # Scale inputs
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Predict
    model.eval()
    with torch.no_grad():
        preds, k, eta = model(X_tensor)
    
    # Convert to numpy
    preds_np = preds.cpu().numpy()
    
    return {
        'outlet_temperature': preds_np[:, 0],
        'excess_o2': preds_np[:, 1],
        'efficiency': eta.item(),
        'leakage_coefficient': k.item()
    }

def main():
    parser = argparse.ArgumentParser(description='PINN Model Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--scaler', type=str, required=True,
                        help='Path to scaler file (.pkl file)')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--output_csv', type=str, default='predictions.csv',
                        help='Path to save predictions')
    parser.add_argument('--feature_cols', type=str,
                        default='InletT-Avg,InletFlow,FGFlow,AmbientT,DraftP,Density',
                        help='Comma-separated list of feature columns')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PINN Model Inference")
    print("=" * 80)
    
    # Load model
    model, scaler, config = load_model(args.checkpoint, args.scaler)
    
    # Load input data
    print(f"\n📥 Loading input data from: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"   Samples: {len(df)}")
    
    # Extract features
    feature_cols = args.feature_cols.split(',')
    print(f"   Features: {feature_cols}")
    
    # Verify columns exist
    missing = set(feature_cols) - set(df.columns)
    if missing:
        print(f"❌ Error: Missing columns in input CSV: {missing}")
        sys.exit(1)
    
    X = df[feature_cols]
    
    # Make predictions
    print("\n🔮 Making predictions...")
    results = predict(model, scaler, X)
    
    # Add predictions to dataframe
    df['Predicted_OutletT'] = results['outlet_temperature']
    df['Predicted_ExcessO2'] = results['excess_o2']
    
    # Save results
    df.to_csv(args.output_csv, index=False)
    print(f"\n✅ Predictions saved to: {args.output_csv}")
    
    # Summary statistics
    print("\n📊 Prediction Summary:")
    print(f"   Outlet Temperature: {results['outlet_temperature'].mean():.2f} ± {results['outlet_temperature'].std():.2f} °C")
    print(f"   Excess O2: {results['excess_o2'].mean():.2f} ± {results['excess_o2'].std():.2f} %")
    print(f"   Min/Max Temp: {results['outlet_temperature'].min():.2f} / {results['outlet_temperature'].max():.2f} °C")
    print(f"   Min/Max O2: {results['excess_o2'].min():.2f} / {results['excess_o2'].max():.2f} %")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
