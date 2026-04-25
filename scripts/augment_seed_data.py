"""
augment_seed_data.py  —  Hybrid data generator for IMU Smart Pen
================================================================
Takes a small handful of real "seed" recordings from data/seed/{0-9}/*.csv
and generates thousands of varied training samples using standard time-series
augmentation techniques (time-warping, scaling, rotation, noise).
"""

import os
import glob
import random
import math
import numpy as np
import pandas as pd
import scipy.interpolate as interp

LABELS = list("123ABC")
# Default to light augmentation: 0.5x of 1200 real samples.
SAMPLES_PER_CLASS = int(os.getenv("AUG_SAMPLES_PER_CLASS", "600"))

def time_warp(df, factor=1.0):
    """
    Stretch or squeeze the time series by a factor.
    factor > 1.0 means writing slower (signal gets longer).
    factor < 1.0 means writing faster (signal gets shorter).
    """
    if factor == 1.0:
        return df.copy()
        
    n = len(df)
    orig_steps = np.arange(n)
    new_steps = np.linspace(0, n - 1, int(n * factor))
    
    new_df = pd.DataFrame()
    # Interpolate each column separately
    for col in df.columns:
        if col == 'time_ms': 
            # Time is re-calculated uniformly at 10ms intervals
            new_df[col] = [i * 10 for i in range(len(new_steps))]
            continue
            
        func = interp.interp1d(orig_steps, df[col], kind='linear', bounds_error=False, fill_value="extrapolate")
        new_df[col] = func(new_steps)
        
    return new_df

def apply_rotation(df, roll_deg=0):
    """
    Applies a slight 3D rotation matrix to the accelerometer and gyroscope data.
    Assuming the pen shaft is the Y-axis, a 'roll' is a rotation around the Y-axis.
    """
    theta = math.radians(roll_deg)
    c, s = math.cos(theta), math.sin(theta)
    
    new_df = df.copy()
    
    # Rotate Accelerometer (X and Z around Y)
    ax = df['ax'] * c + df['az'] * s
    az = -df['ax'] * s + df['az'] * c
    new_df['ax'] = ax
    new_df['az'] = az
    
    # Rotate Gyroscope (X and Z around Y)
    gx = df['gx'] * c + df['gz'] * s
    gz = -df['gx'] * s + df['gz'] * c
    new_df['gx'] = gx
    new_df['gz'] = gz
    
    return new_df

def augment_sample(df):
    """
    Applies a chain of randomized augmentations to a single CSV dataframe.
    """
    # 1. Time Warp (Write speed variation +/- 20%)
    warp_f = random.uniform(0.8, 1.2)
    df_aug = time_warp(df, warp_f)
    
    # 2. Rotation Jitter (Roll around pen axis +/- 15 degrees)
    roll = random.uniform(-15.0, 15.0)
    df_aug = apply_rotation(df_aug, roll)
    
    # 3. Magnitude Scaling (Harder/softer strokes, larger/smaller writing)
    scale_a = random.uniform(0.85, 1.15)
    scale_g = random.uniform(0.85, 1.15)
    
    df_aug['ax'] *= scale_a
    df_aug['ay'] *= scale_a
    df_aug['az'] *= scale_a
    
    df_aug['gx'] *= scale_g
    df_aug['gy'] *= scale_g
    df_aug['gz'] *= scale_g
    
    # 4. White Noise (Sensor jitter & minor hand tremor)
    # Accel baseline noise ~ 0.01g, Gyro noise ~ 1.0 deg/s
    noise_a = np.random.normal(0, 0.015, size=len(df_aug))
    noise_g = np.random.normal(0, 1.5, size=len(df_aug))
    
    df_aug['ax'] += noise_a
    df_aug['ay'] += noise_a
    df_aug['az'] += noise_a
    
    df_aug['gx'] += noise_g
    df_aug['gy'] += noise_g
    df_aug['gz'] += noise_g
    
    return df_aug

def main():
    print("=== Hybrid Smart Pen Data Augmenter ===")
    
    seed_base = "data/seed"
    out_base = "data"
    
    if not os.path.exists(seed_base):
        print(f"[Error] Seed directory '{seed_base}' not found.")
        print(f"Please collect some seed data using the data collector tool.")
        print(f"Ensure files are placed in: {seed_base}/<label>/ (e.g. data/seed/3/)")
        return
        
    total_generated = 0
    
    for label in LABELS:
        seed_dir = os.path.join(seed_base, label)
        if not os.path.isdir(seed_dir):
            print(f"Skipping '{label}' - No seed directory {seed_dir}")
            continue
            
        seed_files = glob.glob(os.path.join(seed_dir, "*.csv"))
        if not seed_files:
            print(f"Skipping '{label}' - No seed CSVs found in {seed_dir}")
            continue
            
        print(f"Found {len(seed_files)} seed files for '{label}'. Generating {SAMPLES_PER_CLASS} augmented samples...")
        
        try:
            seed_dfs = [pd.read_csv(f) for f in seed_files]
        except Exception as e:
            print(f"Error reading CSVs for label '{label}': {e}")
            continue
            
        out_dir = os.path.join(out_base, label)
        os.makedirs(out_dir, exist_ok=True)
        
        for i in range(SAMPLES_PER_CLASS):
            df_base = random.choice(seed_dfs)
            df_aug = augment_sample(df_base)
            
            out_file = os.path.join(out_dir, f"sample_{i+1:03d}.csv")
            df_aug.to_csv(out_file, index=False)
            
        total_generated += SAMPLES_PER_CLASS
            
    if total_generated > 0:
        print(f"\n[SUCCESS] Augmented a total of {total_generated} training samples.")
        print("You can now train the model by running:")
        print("  python scripts/train_bilstm.py")
    else:
        print("\n[WARNING] No augmented samples were generated. Make sure seed folders contain CSV files.")

if __name__ == '__main__':
    main()
