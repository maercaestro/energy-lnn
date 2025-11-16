"""
Energy-Based Liquid Neural Network (EBLNN) Model Architecture
Hybrid model combining LNN (CfC) with Energy-Based Model (EBM) heads
"""

import numpy as np
import torch
import torch.nn as nn
from ncps.torch import CfC


# ============================================================================
# Energy Calculation Functions (Multi-Objective Cost)
# ============================================================================

def CO_model(o2_excess: np.ndarray) -> np.ndarray:
    """
    Calculates the CO (ppm) based on the excess O2 level.
    This function is vectorized to handle numpy arrays.
    
    CO Model (piecewise):
    - O2 < 1.5%: High CO emissions (exponential, max 100 ppm)
    - 1.5% <= O2 <= 2.5%: Optimal range, zero CO
    - O2 > 2.5%: Linear increase in CO
    
    Args:
        o2_excess: Excess oxygen percentage (%)
    
    Returns:
        CO emissions (ppm)
    """
    # Condition 1: O2 < 1.5%
    cond1 = (o2_excess < 1.5)
    val1 = np.minimum(100.0, 6.0 * np.exp(1.6 * (1.5 - o2_excess)))
    
    # Condition 2: 1.5% <= O2 <= 2.5%
    cond2 = (o2_excess >= 1.5) & (o2_excess <= 2.5)
    val2 = 0.0
    
    # Condition 3: O2 > 2.5%
    cond3 = (o2_excess > 2.5)
    val3 = 1.0 + (o2_excess - 2.5)
    
    # Combine conditions
    return np.where(cond1, val1, np.where(cond2, val2, val3))


def calculate_true_energy(
    fuel_flow: np.ndarray,
    next_excess_o2: np.ndarray,
    w_fuel: float = 1.0,
    w_safety: float = 100.0
) -> np.ndarray:
    """
    Calculates the "true" energy (cost) for a given state-action pair.
    This will be the target for our EBM head.
    
    Multi-objective cost function:
    E = W_FUEL × fuel_flow + W_SAFETY × CO_model(O₂)
    
    Objectives:
    1. Minimize energy consumption (fuel cost)
    2. Optimize O2 (1.5-2.5%)
    3. Maintain safety (minimize CO)
    
    Args:
        fuel_flow: Fuel flow rate (units/hour)
        next_excess_o2: Excess oxygen (%)
        w_fuel: Weight for fuel cost objective (default 1.0)
        w_safety: Weight for safety/CO objective (default 100.0)
    
    Returns:
        Total energy cost
    """
    # Objective B: Minimize energy consumption
    cost_fuel = w_fuel * fuel_flow
    
    # Objective A & C: Optimize O2 / Maintain Safety
    cost_safety = w_safety * CO_model(next_excess_o2)
    
    return cost_fuel + cost_safety


# ============================================================================
# EBLNN Model Architecture
# ============================================================================

class EBLNN(nn.Module):
    """
    Energy-Based Liquid Neural Network (EBLNN)
    
    Architecture:
        Input (5 features: fuel_flow, AFR, temp, inflow_temp, inflow_rate)
            ↓
        [CfC Body] - Liquid Neural Network core
            ↓
            ├─→ [Prediction Head] → (next_temp, next_excess_o2)  [Physics]
            └─→ [Energy Head]      → (energy_cost)               [EBM]
    
    The model is trained jointly with:
        L_total = L_LNN + α × L_EBM
    
    Where:
        - L_LNN: MSE for physics predictions
        - L_EBM: MSE for energy/cost predictions
        - α: Balance hyperparameter
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size_prediction: int = 2,  # [next_temp, next_excess_o2]
        output_size_energy: int = 1,      # [energy_cost]
        mixed_memory: bool = True,
        batch_first: bool = True
    ):
        """
        Initialize EBLNN model.
        
        Args:
            input_size: Number of input features (default 5)
            hidden_size: Hidden layer size for CfC
            output_size_prediction: Number of physics outputs (default 2)
            output_size_energy: Number of energy outputs (default 1)
            mixed_memory: Use mixed memory in CfC (default True)
            batch_first: Batch first format (default True)
        """
        super(EBLNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size_prediction = output_size_prediction
        self.output_size_energy = output_size_energy
        
        # 1. Body: The CfC (LNN) core
        # CfC = Closed-form Continuous-time network
        # Approximates ODE integration without solver
        self.cfc_body = CfC(
            input_size,
            hidden_size,
            mixed_memory=mixed_memory,
            batch_first=batch_first
        )
        
        # 2. Head 1: The LNN Prediction Head (Physics)
        # Predicts: [next_temp, next_excess_o2]
        self.predict_head = nn.Linear(hidden_size, output_size_prediction)
        
        # 3. Head 2: The EBM Energy Head (Multi-Objective Cost)
        # Predicts: [energy_cost]
        self.energy_head = nn.Linear(hidden_size, output_size_energy)
    
    def forward(self, x, hx=None):
        """
        Forward pass through the EBLNN.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            hx: Initial hidden state (optional)
        
        Returns:
            y_pred: Physics predictions (batch_size, seq_len, 2)
            e_pred: Energy predictions (batch_size, seq_len, 1)
            last_h: Final hidden state
        """
        # Pass input through the CfC body
        # h_seq shape: (batch_size, seq_len, hidden_size)
        h_seq, last_h = self.cfc_body(x, hx)
        
        # Pass the entire sequence of hidden states to the heads
        # This gives us a prediction for every timestep
        
        # Prediction Head Output (Physics)
        # y_pred shape: (batch_size, seq_len, output_size_prediction)
        y_pred = self.predict_head(h_seq)
        
        # Energy Head Output (EBM)
        # e_pred shape: (batch_size, seq_len, output_size_energy)
        e_pred = self.energy_head(h_seq)
        
        return y_pred, e_pred, last_h
    
    def get_num_parameters(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Model Factory Functions
# ============================================================================

def create_model(
    input_size: int = 5,
    hidden_size: int = 128,
    device: str = 'cpu'
) -> EBLNN:
    """
    Create and initialize EBLNN model.
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden layer size
        device: Device to place model on ('cpu' or 'cuda')
    
    Returns:
        Initialized EBLNN model
    """
    model = EBLNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size_prediction=2,
        output_size_energy=1
    )
    
    model = model.to(device)
    
    print(f"Created EBLNN model with {model.get_num_parameters():,} parameters")
    print(f"  - Input size: {input_size}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Prediction outputs: 2 (temp, O2)")
    print(f"  - Energy outputs: 1")
    print(f"  - Device: {device}")
    
    return model
