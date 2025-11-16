"""
Synthetic Data Generation for Furnace Thermodynamic System
Generates training data based on physics-based heat transfer models
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


class FurnaceDataGenerator:
    """
    Generates synthetic furnace operation data based on thermodynamic principles.
    """
    
    # Physical Constants
    AFR_OPT = 14.7                  # Optimal air-fuel ratio
    MAX_FUEL_ENERGY = 39000.0       # kJ per Nm³ (natural gas)
    FURNACE_MASS = 5000.0           # kg
    SPECIFIC_HEAT = 0.5             # kJ/(kg·°C)
    HEAT_LOSS_COEFF = 0.0005        # kJ/(°C·s)
    INLET_COEFF = 0.0002            # kJ/(unit·°C·s)
    AMBIENT_TEMP = 25.0             # °C
    TIME_STEP = 1.0                 # second
    
    # Heat transfer constants for O2 calculation
    HHV = 39000.0                   # Higher Heating Value (kJ per Nm³)
    U = 0.0005                      # Heat transfer coefficient (kJ/(s·m²·°C))
    A = 10.0                        # Heat transfer area (m²)
    T_FLAME = 1800.0                # Flame temperature (°C)
    SIGMA = 2.0                     # Efficiency curve width parameter
    
    def __init__(self, seed: int = 42):
        """
        Initialize data generator with random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        
        # Define variable ranges
        self.ranges = {
            'fuel_flow':      (1.0, 20.0),
            'air_fuel_ratio': (0.6, 25.0),
            'current_temp':   (25.0, 500.0),
            'inflow_temp':    (100.0, 200.0),
            'inflow_rate':    (50.0, 200.0)
        }
    
    def calculate_temperature(
        self,
        fuel_flow: float,
        air_flow: float,
        current_temp: float,
        inflow_temp: float,
        inflow_rate: float,
        noise_level: float = 1.0
    ) -> float:
        """
        Calculate next temperature using heat transfer model.
        
        Args:
            fuel_flow: Fuel flow rate (units/hour)
            air_flow: Air flow rate (units/hour)
            current_temp: Current furnace temperature (°C)
            inflow_temp: Incoming material temperature (°C)
            inflow_rate: Material inflow rate (units/hour)
            noise_level: Random noise magnitude (default 1.0)
        
        Returns:
            float: Next temperature (°C)
        """
        # Convert rates to SI per second (units/hour to units/second)
        fuel_s = fuel_flow / 3600.0
        inflow_s = inflow_rate / 3600.0
        
        # Combustion efficiency (Gaussian curve around optimal AFR)
        AFR = air_flow / max(fuel_flow, 1e-3)
        efficiency = np.exp(-(AFR - self.AFR_OPT)**2 / (2 * self.SIGMA**2))
        
        # Heat Balance Components
        Q_comb = fuel_s * self.MAX_FUEL_ENERGY * efficiency * self.TIME_STEP
        Q_loss = self.HEAT_LOSS_COEFF * (current_temp - self.AMBIENT_TEMP) * self.TIME_STEP
        Q_inflow = self.INLET_COEFF * inflow_s * (current_temp - inflow_temp) * self.TIME_STEP
        
        # Net energy and temperature change
        net_energy = Q_comb - Q_loss - Q_inflow
        temp_change = net_energy / (self.FURNACE_MASS * self.SPECIFIC_HEAT)
        
        # Add random noise
        noise = (np.random.rand() - 0.5) * 2 * noise_level
        new_temp = current_temp + temp_change + noise
        
        return max(self.AMBIENT_TEMP, new_temp)
    
    def calculate_excess_o2(
        self,
        air_fuel_ratio: float,
        fuel_flow: float,
        current_temp: float,
        noise_level: float = 0.2
    ) -> float:
        """
        Calculate excess oxygen based on air-fuel ratio and operating conditions.
        
        Args:
            air_fuel_ratio: Air-to-fuel ratio
            fuel_flow: Fuel flow rate (units/hour)
            current_temp: Current furnace temperature (°C)
            noise_level: Random noise magnitude (default 0.2)
        
        Returns:
            float: Excess oxygen percentage (%)
        """
        # Convert fuel flow to Nm³/s
        fuel_s = fuel_flow / 3600.0
        
        # Combustion efficiency (Gaussian)
        eta = np.exp(-(air_fuel_ratio - self.AFR_OPT)**2 / (2 * self.SIGMA**2))
        
        # Heat rates
        Q_comb = fuel_s * self.HHV * eta
        Q_trans = self.U * self.A * (self.T_FLAME - current_temp)
        
        # Calculate excess O2 based on heat loss
        frac_lost = max(0, 1 - Q_trans / max(Q_comb, 1e-6))
        excess_o2 = frac_lost * 21.0
        
        # Add small noise
        noise = (np.random.rand() - 0.5) * 2 * noise_level
        return max(0, excess_o2 + noise)
    
    def generate_data(
        self,
        num_sequences: int = 10000,
        sequence_length: int = 30,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Generate synthetic furnace operation data for ML training.
        
        Args:
            num_sequences: Number of operation scenarios to generate
            sequence_length: Number of timesteps per scenario
            verbose: Whether to print progress
        
        Returns:
            pandas.DataFrame: Synthetic data with columns:
                - sequence: Sequence ID
                - timestep: Timestep within sequence
                - fuel_flow: Fuel flow rate (units/hour)
                - air_fuel_ratio: Air-to-fuel ratio
                - current_temp: Current temperature (°C)
                - inflow_temp: Inflow temperature (°C)
                - inflow_rate: Inflow rate (units/hour)
                - next_temp: Next temperature (°C)
                - next_excess_o2: Next excess oxygen (%)
        """
        if verbose:
            print(f"Generating {num_sequences} sequences × {sequence_length} timesteps...")
            print(f"Total data points: {num_sequences * sequence_length:,}")
        
        records = []
        
        for seq in range(num_sequences):
            # Initialize random starting conditions for this sequence
            fuel_flow = np.random.uniform(*self.ranges['fuel_flow'])
            afr = np.random.uniform(*self.ranges['air_fuel_ratio'])
            current_temp = np.random.uniform(*self.ranges['current_temp'])
            inflow_temp = np.random.uniform(*self.ranges['inflow_temp'])
            inflow_rate = np.random.uniform(*self.ranges['inflow_rate'])
            air_flow = fuel_flow * afr
            
            # Generate timesteps
            for t in range(sequence_length):
                # Calculate next states
                next_temp = self.calculate_temperature(
                    fuel_flow, air_flow, current_temp, inflow_temp, inflow_rate
                )
                next_o2 = self.calculate_excess_o2(afr, fuel_flow, current_temp)
                
                # Store record
                records.append({
                    'sequence':        seq,
                    'timestep':        t,
                    'fuel_flow':       fuel_flow,
                    'air_fuel_ratio':  afr,
                    'current_temp':    current_temp,
                    'inflow_temp':     inflow_temp,
                    'inflow_rate':     inflow_rate,
                    'next_temp':       next_temp,
                    'next_excess_o2':  next_o2
                })
                
                # Update state for next timestep (random walk with bounds)
                current_temp = next_temp
                inflow_temp = np.clip(
                    inflow_temp + (np.random.rand() - 0.5) * 10,  # ±5°C variation
                    *self.ranges['inflow_temp']
                )
                inflow_rate = np.clip(
                    inflow_rate + (np.random.rand() - 0.5) * 20,  # ±10 units variation
                    *self.ranges['inflow_rate']
                )
            
            # Progress indicator
            if verbose and (seq + 1) % 1000 == 0:
                print(f"  Generated {seq + 1:,} sequences...")
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        if verbose:
            print(f"\n✅ Generation complete! Shape: {df.shape}")
        
        return df
    
    def get_data_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistics for generated data.
        
        Args:
            df: Generated dataframe
        
        Returns:
            DataFrame with statistics
        """
        variables = [
            'fuel_flow', 'air_fuel_ratio', 'current_temp',
            'inflow_temp', 'inflow_rate', 'next_temp', 'next_excess_o2'
        ]
        
        stats = []
        for var in variables:
            stats.append({
                'Variable': var,
                'Mean': df[var].mean(),
                'Std': df[var].std(),
                'Min': df[var].min(),
                'Max': df[var].max(),
                'Range': df[var].max() - df[var].min()
            })
        
        return pd.DataFrame(stats)


def load_or_generate_data(
    data_path: str,
    num_sequences: int = 10000,
    sequence_length: int = 30,
    seed: int = 42,
    force_regenerate: bool = False
) -> pd.DataFrame:
    """
    Load existing data or generate new data if not found.
    
    Args:
        data_path: Path to save/load data
        num_sequences: Number of sequences to generate
        sequence_length: Length of each sequence
        seed: Random seed
        force_regenerate: Force regeneration even if file exists
    
    Returns:
        DataFrame with generated data
    """
    import os
    
    if os.path.exists(data_path) and not force_regenerate:
        print(f"Loading existing data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} rows")
        return df
    
    print(f"Generating new data...")
    generator = FurnaceDataGenerator(seed=seed)
    df = generator.generate_data(num_sequences, sequence_length)
    
    # Save data
    df.to_csv(data_path, index=False)
    print(f"💾 Data saved to {data_path}")
    
    return df
