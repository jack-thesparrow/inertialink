"""
augment_hard_samples.py — Targeted augmentation from misclassified samples
==========================================================================
Reads models/eval_results.csv, finds wrong predictions, and generates extra
augmented samples from those hard examples into data/hard/<label>/.

Usage:
  python3 scripts/augment_hard_samples.py
  python3 scripts/augment_hard_samples.py --per-mistake 30 --max-per-label 400
"""

import argparse
import csv
import glob
import math
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.interpolate as interp

DEFAULT_RESULTS = "models/eval_results.csv"
OUT_BASE = "data/hard"
FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=DEFAULT_RESULTS, help="Path to eval_results.csv")
    p.add_argument("--per-mistake", type=int, default=20, help="Augmented samples per wrong row")
    p.add_argument("--max-per-label", type=int, default=500, help="Cap hard samples generated per label")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def _read_eval_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing results file: {path}")
    with open(path, newline="") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        rows = list(reader)
    if not rows:
        raise ValueError("No rows found in eval results.")
    return rows


def time_warp(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    if abs(factor - 1.0) < 1e-6:
        return df.copy()
    n = len(df)
    if n < 4:
        return df.copy()
    orig_steps = np.arange(n)
    new_steps = np.linspace(0, n - 1, max(4, int(n * factor)))
    out = pd.DataFrame()
    for col in df.columns:
        if col == "time_ms":
            out[col] = [i * 10 for i in range(len(new_steps))]
            continue
        fn = interp.interp1d(orig_steps, df[col], kind="linear", bounds_error=False, fill_value="extrapolate")
        out[col] = fn(new_steps)
    return out


def apply_rotation(df: pd.DataFrame, roll_deg: float) -> pd.DataFrame:
    t = math.radians(roll_deg)
    c, s = math.cos(t), math.sin(t)
    out = df.copy()
    ax = df["ax"] * c + df["az"] * s
    az = -df["ax"] * s + df["az"] * c
    gx = df["gx"] * c + df["gz"] * s
    gz = -df["gx"] * s + df["gz"] * c
    out["ax"] = ax
    out["az"] = az
    out["gx"] = gx
    out["gz"] = gz
    return out


def augment_hard(df: pd.DataFrame) -> pd.DataFrame:
    # Stronger than base augmentation to push boundary coverage on hard cases.
    warp_f = random.uniform(0.7, 1.3)
    out = time_warp(df, warp_f)
    out = apply_rotation(out, random.uniform(-20.0, 20.0))

    scale_a = random.uniform(0.8, 1.2)
    scale_g = random.uniform(0.8, 1.2)
    out["ax"] *= scale_a
    out["ay"] *= scale_a
    out["az"] *= scale_a
    out["gx"] *= scale_g
    out["gy"] *= scale_g
    out["gz"] *= scale_g

    noise_a = np.random.normal(0, 0.02, size=len(out))
    noise_g = np.random.normal(0, 2.0, size=len(out))
    out["ax"] += noise_a
    out["ay"] += noise_a
    out["az"] += noise_a
    out["gx"] += noise_g
    out["gy"] += noise_g
    out["gz"] += noise_g
    return out


def _next_index(folder: str) -> int:
    existing = glob.glob(os.path.join(folder, "hard_*.csv"))
    max_i = 0
    for p in existing:
        name = os.path.basename(p).replace("hard_", "").replace(".csv", "")
        try:
            max_i = max(max_i, int(name))
        except ValueError:
            pass
    return max_i + 1


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    rows = _read_eval_rows(args.results)
    hard_rows = [r for r in rows if str(r.get("correct", "")).lower() in {"false", "0"}]
    if not hard_rows:
        print("No misclassified rows found. Nothing to augment.")
        return

    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in hard_rows:
        by_label[r["word"]].append(r)

    total_written = 0
    print(f"Found {len(hard_rows)} misclassified rows in {args.results}")

    for label, mistakes in sorted(by_label.items()):
        out_dir = os.path.join(OUT_BASE, label)
        os.makedirs(out_dir, exist_ok=True)
        idx = _next_index(out_dir)
        written_for_label = 0

        for m in mistakes:
            src_path = m.get("path", "")
            if not src_path or not os.path.exists(src_path):
                continue
            try:
                df = pd.read_csv(src_path)
            except Exception:
                continue
            if any(c not in df.columns for c in FEATURES):
                continue
            if len(df) < 6:
                continue

            for _ in range(args.per_mistake):
                if written_for_label >= args.max_per_label:
                    break
                out_df = augment_hard(df)
                out_path = os.path.join(out_dir, f"hard_{idx:04d}.csv")
                out_df.to_csv(out_path, index=False)
                idx += 1
                written_for_label += 1
                total_written += 1
            if written_for_label >= args.max_per_label:
                break

        print(f"  {label}: wrote {written_for_label} hard samples -> {out_dir}")

    print(f"\n[SUCCESS] Wrote {total_written} targeted hard samples under {OUT_BASE}/")
    print("Next steps:")
    print("  1) python3 scripts/train_bilstm.py")
    print("  2) python3 scripts/eval_model.py --source both")


if __name__ == "__main__":
    main()
