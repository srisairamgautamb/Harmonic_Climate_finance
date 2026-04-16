import config
import numpy as np
import pandas as pd
import os
from pathlib import Path

def generate_synthetic_data():
    print("Generating synthetic aligned monthly data...")
    
    # Generate dates
    dates = pd.date_range(start=config.START_DATE, end=config.END_DATE, freq=config.FREQ)
    n = len(dates)
    print(f"Time dimension: {n} months")
    
    # Get all required variables
    all_vars = (
        config.DRIVER_VARS + 
        config.CLIMATE_VARS + 
        config.CLIMATE_RISK_VARS + 
        config.FINANCIAL_RISK_VARS
    )
    
    # Generate random walk data for each variable to mimic real time-series
    np.random.seed(config.SEED)
    data = {}
    for var in all_vars:
        # Random walk with a bit of momentum/drift
        steps = np.random.randn(n) * 0.1
        walk = np.cumsum(steps) + np.random.randn(n) * 0.05
        # Add some seasonality (sine wave) just for realism
        seasonality = 0.5 * np.sin(2 * np.pi * np.arange(n) / 12.0)
        data[var] = walk + seasonality
        
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    
    # Ensure directory exists
    out_dir = Path(config.DATA_PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "aligned_monthly.csv"
    df.to_csv(out_path)
    print(f"Synthetic data saved to {out_path} with shape {df.shape}")
    
if __name__ == "__main__":
    generate_synthetic_data()
