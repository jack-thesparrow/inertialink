import os
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dtw_distance(s1, s2):
    """Simple Dynamic Time Warping (DTW) distance metric."""
    n, m = len(s1), len(s2)
    dtw = np.full((n + 1, m + 1), float('inf'))
    dtw[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
            
    return dtw[n, m]

def load_random_samples(digit, count=2):
    """Load a set number of random CSVs from the seed directory."""
    path = f"data/seed/{digit}/*.csv"
    files = glob.glob(path)
    if not files:
        print(f"Skipping {digit} - no files found.")
        return []
    
    samples = []
    for f in random.sample(files, min(count, len(files))):
        df = pd.read_csv(f)
        # We focus on the accelerometer (ax, ay, az) and gyro (gx, gy, gz)
        data = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values
        samples.append(data)
    return samples

def main():
    print("=== Stroke Comparison Analyzer ===")
    
    digitsToTest = ["1", "2", "3"]
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    plt.figure(figsize=(14, 12))
    
    for idx, digit in enumerate(digitsToTest):
        samples = load_random_samples(digit, count=3)
        if len(samples) < 2:
            continue
            
        plt.subplot(len(digitsToTest), 2, (idx*2) + 1)
        plt.title(f"Digit {digit} - Accelerometer Raw")
        for i, data in enumerate(samples):
            plt.plot(data[:, 0], label=f'ax #{i}', alpha=0.6, color=colors[i])
            plt.plot(data[:, 2], '--', label=f'az #{i}', alpha=0.4, color=colors[i])
        plt.ylabel("G-Force")
        if idx == 0: plt.legend()
            
        plt.subplot(len(digitsToTest), 2, (idx*2) + 2)
        plt.title(f"Digit {digit} - Gyroscope Raw")
        for i, data in enumerate(samples):
            plt.plot(data[:, 3], label=f'gx #{i}', alpha=0.6, color=colors[i])
        plt.ylabel("Deg/s")
        if idx == 0: plt.legend()
        
        # Calculate Match Distance
        print(f"--- Digit {digit} Similarity ---")
        dist = dtw_distance(samples[0], samples[1])
        print(f"DTW Score (Sample 0 vs 1): {dist:.2f} (lower is more similar)")

    plt.tight_layout()
    plot_path = "comparison_plot.png"
    plt.savefig(plot_path)
    print(f"\nSaved plot to {plot_path}.")

if __name__ == "__main__":
    main()
