"""
generate_synthetic_data.py — augment existing stroke CSVs for training.

With no physical hardware you only have a handful of real samples per word.
This script inflates your dataset by creating augmented copies of each CSV
with realistic variation:

  - Random affine transform  (slight rotation + uniform scale)
  - Gaussian noise           (proportional to the stroke's own range)
  - Smooth time warping      (simulates writing faster or slower)
  - accel_z re-spike         (randomises impact magnitude)

Usage:
    python3 scripts/generate_synthetic_data.py
    python3 scripts/generate_synthetic_data.py --samples-per-source 30
    python3 scripts/generate_synthetic_data.py --data-dir data --samples-per-source 20

After running, retrain:
    python3 scripts/train_bilstm.py
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# -----------------------------------------------------------------------
# Augmentation helpers
# -----------------------------------------------------------------------

def _smooth_warp(length: int, sigma: float) -> np.ndarray:
    """Return a monotone warp map from [0,1] to [0,1] with smooth perturbation."""
    num_knots = max(4, length // 15)
    t = np.linspace(0, 1, num_knots)
    # Perturb knots then enforce monotonicity by sorting
    delta = np.random.normal(0, sigma, num_knots)
    delta[0] = delta[-1] = 0.0          # keep endpoints fixed
    warped = np.clip(t + delta, 0, 1)
    warped = np.sort(warped)             # monotone
    warped[0], warped[-1] = 0.0, 1.0

    t_dense = np.linspace(0, 1, length)
    return interp1d(t, warped, kind='linear')(t_dense)


def augment(df: pd.DataFrame,
            noise_frac: float = 0.03,
            warp_sigma: float = 0.12,
            scale_range: tuple = (0.88, 1.12),
            angle_std_rad: float = 0.06) -> pd.DataFrame:
    """Return one augmented copy of a stroke DataFrame."""
    x  = df['x'].to_numpy(dtype=float)
    y  = df['y'].to_numpy(dtype=float)
    az = df['accel_z'].to_numpy(dtype=float)
    n  = len(x)

    # 1. Affine: random scale + small in-plane rotation
    scale = np.random.uniform(*scale_range)
    theta = np.random.normal(0.0, angle_std_rad)
    c, s  = np.cos(theta), np.sin(theta)
    x2 = scale * (c * x - s * y)
    y2 = scale * (s * x + c * y)
    x, y = x2, y2

    # 2. Additive Gaussian noise scaled to stroke extent
    x_span = np.ptp(x) + 1.0           # avoid zero-span edge case
    y_span = np.ptp(y) + 1.0
    x += np.random.normal(0, noise_frac * x_span, n)
    y += np.random.normal(0, noise_frac * y_span, n)
    az += np.random.normal(0, noise_frac * 0.15, n)

    # 3. Smooth time warp — resample x and y along a perturbed time axis
    #    (accel_z is an event signal; warping it separately would distort the
    #     impact spike, so we just add noise to it instead)
    t_orig   = np.linspace(0, 1, n)
    t_warped = _smooth_warp(n, warp_sigma)

    x  = interp1d(t_orig, x,  kind='linear')(t_warped)
    y  = interp1d(t_orig, y,  kind='linear')(t_warped)

    # 4. Re-randomise the accel_z impact spike amplitude (it varies per tap)
    peak_idx = int(np.argmax(az))
    az[peak_idx] *= np.random.uniform(0.7, 1.3)
    az = np.clip(az, 0.0, None)

    out = df.copy()
    out['x']       = x
    out['y']       = y
    out['accel_z'] = az
    return out


# -----------------------------------------------------------------------
# Per-folder generation
# -----------------------------------------------------------------------

def _next_sample_number(folder: str) -> int:
    existing = glob.glob(os.path.join(folder, 'sample_*.csv'))
    nums = []
    for f in existing:
        stem = os.path.splitext(os.path.basename(f))[0]  # e.g. "sample_001"
        try:
            nums.append(int(stem.split('_')[1]))
        except (IndexError, ValueError):
            pass
    return max(nums, default=0) + 1


def generate_for_word(folder: str, samples_per_source: int) -> int:
    sources = sorted(glob.glob(os.path.join(folder, 'sample_*.csv')))
    if not sources:
        print(f"  [skip] no sample_*.csv files found")
        return 0

    next_n   = _next_sample_number(folder)
    created  = 0

    for src in sources:
        try:
            df = pd.read_csv(src)
        except Exception as e:
            print(f"  [warn] could not read {src}: {e}")
            continue

        if 'x' not in df.columns or 'y' not in df.columns:
            print(f"  [warn] {src} missing x/y columns — skipping")
            continue

        for _ in range(samples_per_source):
            aug  = augment(df)
            path = os.path.join(folder, f'sample_{next_n:03d}.csv')
            aug.to_csv(path, index=False)
            next_n  += 1
            created += 1

    return created


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Augment stroke CSV data to expand your training set.')
    parser.add_argument('--data-dir', default='data',
                        help='Root data directory (default: data)')
    parser.add_argument('--samples-per-source', type=int, default=20,
                        help='Augmented copies to create per source CSV (default: 20)')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"[Error] '{args.data_dir}' not found. "
              "Collect at least one real sample first via ./bin/data_collector.")
        return

    word_folders = sorted(
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    )

    if not word_folders:
        print(f"[Error] No word folders found inside '{args.data_dir}'.")
        return

    print(f"Found words: {word_folders}")
    print(f"Generating {args.samples_per_source} augmented copies per source CSV.\n")

    total = 0
    for word in word_folders:
        path = os.path.join(args.data_dir, word)
        print(f"  '{word}'  ...", end=' ', flush=True)
        n = generate_for_word(path, args.samples_per_source)
        print(f"+{n} samples")
        total += n

    print(f"\n[SUCCESS] Created {total} new samples across {len(word_folders)} word(s).")
    print("Now retrain:  python3 scripts/train_bilstm.py")


if __name__ == '__main__':
    main()
