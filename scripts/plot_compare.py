import os
import glob
import random
import itertools
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


def normalized_dtw(s1, s2):
    """Length-normalized DTW so scores are comparable across different durations."""
    raw = dtw_distance(s1, s2)
    denom = max(1, (len(s1) + len(s2)) / 2.0)
    return raw / denom

def load_random_samples(label, count=2):
    """Load a set number of random CSVs from the seed directory."""
    path = f"data/seed/{label}/*.csv"
    files = glob.glob(path)
    if not files:
        print(f"Skipping '{label}' - no files found.")
        return []
    
    samples = []
    for f in random.sample(files, min(count, len(files))):
        df = pd.read_csv(f)
        # We focus on the accelerometer (ax, ay, az) and gyro (gx, gy, gz)
        data = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values
        samples.append(data)
    return samples


def summarize_similarity(samples):
    """Return pairwise normalized DTW stats across all sample pairs."""
    if len(samples) < 2:
        return None
    dists = [
        normalized_dtw(a, b)
        for a, b in itertools.combinations(samples, 2)
    ]
    return {
        "pairs": len(dists),
        "mean": float(np.mean(dists)),
        "std": float(np.std(dists)),
        "min": float(np.min(dists)),
        "max": float(np.max(dists)),
    }


def interclass_separation(sample_map):
    """Compute mean normalized DTW between every pair of classes."""
    rows = []
    labels = sorted(sample_map.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a_label, b_label = labels[i], labels[j]
            a_samples = sample_map[a_label]
            b_samples = sample_map[b_label]
            if not a_samples or not b_samples:
                continue
            dists = [
                normalized_dtw(a, b)
                for a in a_samples
                for b in b_samples
            ]
            if not dists:
                continue
            rows.append({
                "pair": f"{a_label}-{b_label}",
                "mean": float(np.mean(dists)),
                "std": float(np.std(dists)),
                "min": float(np.min(dists)),
                "max": float(np.max(dists)),
            })
    return rows

def main():
    print("=== Stroke Comparison Analyzer ===")
    
    labels_to_test = ["1", "2", "3", "A", "B", "C"]
    sample_count = 4
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    plt.figure(figsize=(14, 3 * len(labels_to_test)))
    
    sample_map = {}
    intra_stats = {}
    for idx, label in enumerate(labels_to_test):
        samples = load_random_samples(label, count=sample_count)
        if len(samples) < 2:
            continue
        sample_map[label] = samples
            
        plt.subplot(len(labels_to_test), 2, (idx * 2) + 1)
        plt.title(f"Class {label} - Accelerometer Raw")
        for i, data in enumerate(samples):
            color = colors[i % len(colors)]
            plt.plot(data[:, 0], label=f'ax #{i}', alpha=0.6, color=color)
            plt.plot(data[:, 2], '--', label=f'az #{i}', alpha=0.4, color=color)
        plt.ylabel("G-Force")
        if idx == 0: plt.legend()
            
        plt.subplot(len(labels_to_test), 2, (idx * 2) + 2)
        plt.title(f"Class {label} - Gyroscope Raw")
        for i, data in enumerate(samples):
            color = colors[i % len(colors)]
            plt.plot(data[:, 3], label=f'gx #{i}', alpha=0.6, color=color)
        plt.ylabel("Deg/s")
        if idx == 0: plt.legend()
        
        sim = summarize_similarity(samples)
        intra_stats[label] = sim
        print(f"--- Class {label} Similarity ---")
        print(
            f"nDTW over {sim['pairs']} pairs: "
            f"mean={sim['mean']:.2f}  std={sim['std']:.2f}  "
            f"min={sim['min']:.2f}  max={sim['max']:.2f}  (lower is more similar)"
        )

    # Inter-class separability summary
    inter_rows = interclass_separation(sample_map)
    if inter_rows:
        print("\n=== Inter-class Separation (nDTW) ===")
        inter_rows_sorted = sorted(inter_rows, key=lambda r: r["mean"])
        for row in inter_rows_sorted:
            print(
                f"{row['pair']}: mean={row['mean']:.2f}  std={row['std']:.2f}  "
                f"min={row['min']:.2f}  max={row['max']:.2f}"
            )

        print("\n=== Confusion Risk (closest class pairs first) ===")
        for row in inter_rows_sorted[:5]:
            print(f"{row['pair']}  mean_nDTW={row['mean']:.2f}")

        # Margin view: how far each class is from its nearest other class
        print("\n=== Per-class Margin (nearest other class distance) ===")
        for label in sorted(intra_stats.keys()):
            own_mean = intra_stats[label]["mean"]
            nearest = None
            nearest_dist = float("inf")
            for row in inter_rows:
                a, b = row["pair"].split("-")
                if label in (a, b) and row["mean"] < nearest_dist:
                    nearest_dist = row["mean"]
                    nearest = row["pair"]
            if nearest is None:
                continue
            margin = nearest_dist - own_mean
            print(
                f"{label}: intra_mean={own_mean:.2f}  nearest={nearest}({nearest_dist:.2f})  "
                f"margin={margin:.2f}"
            )

    plt.tight_layout()
    plot_path = "comparison_plot.png"
    plt.savefig(plot_path)
    print(f"\nSaved plot to {plot_path}.")

if __name__ == "__main__":
    main()
