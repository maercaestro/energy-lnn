#!/bin/bash
# Script to run WandB Sweep for PINN hyperparameter optimization

set -e

echo "🔥 Starting WandB Sweep for Furnace PINN"
echo "========================================"

# Check if sweep config exists
if [ ! -f "sweep_config.yaml" ]; then
    echo "❌ Error: sweep_config.yaml not found!"
    exit 1
fi

# Check if data file exists
if [ ! -f "furnace_data_cleaned.csv" ]; then
    echo "❌ Error: furnace_data_cleaned.csv not found!"
    exit 1
fi

# Initialize sweep and get sweep ID
echo "📝 Creating sweep..."
SWEEP_ID=$(wandb sweep sweep_config.yaml 2>&1 | grep "wandb agent" | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "❌ Error: Failed to create sweep"
    exit 1
fi

echo "✅ Sweep created: $SWEEP_ID"
echo ""
echo "To run the sweep agent:"
echo "  wandb agent $SWEEP_ID"
echo ""
echo "Or run multiple agents in parallel:"
echo "  wandb agent $SWEEP_ID --count 5"
echo ""
echo "To run on Azure VM, first ensure you're logged in:"
echo "  wandb login"
echo "  wandb agent $SWEEP_ID"
