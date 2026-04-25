"""
flag_anomalies.py — Identify suspicious eval samples for quarantine review
==========================================================================
Reads models/eval_results.csv, computes per-label signal stats, and flags
rows that look anomalous (not just hard) based on:
  - wrong + very high confidence
  - frame-length outlier within class
  - abnormal sensor spike behavior

Outputs:
  models/anomaly_candidates.csv
  models/quarantine_manifest.txt

Usage:
  python3 scripts/flag_anomalies.py
  python3 scripts/flag_anomalies.py --source both --only-wrong 1
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

DEFAULT_RESULTS = "models/eval_results.csv"
OUT_CSV = "models/anomaly_candidates.csv"
OUT_MANIFEST = "models/quarantine_manifest.txt"
FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]


@dataclass
class Thresholds:
    high_conf_wrong: float = 95.0
    z_len: float = 3.0
    z_spike: float = 3.0
    min_score: int = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=DEFAULT_RESULTS, help="Path to eval_results.csv")
    p.add_argument("--out", default=OUT_CSV, help="Output CSV path")
    p.add_argument("--manifest", default=OUT_MANIFEST, help="Output quarantine manifest path")
    p.add_argument("--source", default="both", choices=["augmented", "seed", "both"], help="Filter by source column")
    p.add_argument("--only-wrong", type=int, default=1, help="1=only wrong rows, 0=scan all rows")
    p.add_argument("--high-conf-wrong", type=float, default=95.0, help="Wrong row confidence threshold")
    p.add_argument("--z-len", type=float, default=3.0, help="Absolute z-score threshold for frame length")
    p.add_argument("--z-spike", type=float, default=3.0, help="Absolute z-score threshold for spike score")
    p.add_argument("--min-score", type=int, default=2, help="Minimum anomaly score to flag")
    return p.parse_args()


def load_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing results file: {path}")
    with open(path, newline="") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        rows = list(reader)
    if not rows:
        raise ValueError("No rows found in results file.")
    required = {"word", "predicted", "correct", "confidence", "n_frames", "source", "path"}
    got = set(rows[0].keys())
    if not required.issubset(got):
        missing = sorted(required - got)
        raise ValueError(f"Results file is missing required columns: {missing}")
    return rows


def robust_z(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.array([])
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad < 1e-8:
        std = np.std(values)
        if std < 1e-8:
            return np.zeros_like(values)
        return (values - np.mean(values)) / std
    return 0.6745 * (values - med) / mad


def compute_spike_score(csv_path: str) -> float:
    if not os.path.exists(csv_path):
        return np.nan
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return np.nan
    if any(col not in df.columns for col in FEATURES):
        return np.nan
    x = df[FEATURES].values.astype(np.float32)
    if x.shape[0] < 4:
        return np.nan
    # Approximate jerk-like behavior as frame-to-frame derivative magnitude.
    dx = np.diff(x, axis=0)
    mag = np.linalg.norm(dx, axis=1)
    if mag.size == 0:
        return np.nan
    p95 = np.percentile(mag, 95)
    med = np.median(mag)
    return float(p95 / (med + 1e-6))


def main() -> None:
    args = parse_args()
    th = Thresholds(
        high_conf_wrong=args.high_conf_wrong,
        z_len=args.z_len,
        z_spike=args.z_spike,
        min_score=args.min_score,
    )

    rows = load_rows(args.results)

    # Filter source scope
    if args.source != "both":
        rows = [r for r in rows if r.get("source") == args.source]
    if args.only_wrong == 1:
        rows = [r for r in rows if str(r.get("correct", "")).lower() in {"false", "0"}]

    if not rows:
        print("No rows after filtering. Nothing to analyze.")
        return

    # Compute spike scores and per-label aggregates
    per_label_len: Dict[str, List[float]] = defaultdict(list)
    per_label_spike: Dict[str, List[float]] = defaultdict(list)
    enriched = []

    for r in rows:
        label = r["word"]
        n_frames = float(r.get("n_frames", "0") or 0)
        spike = compute_spike_score(r["path"])
        rec = dict(r)
        rec["n_frames_f"] = n_frames
        rec["spike_score"] = spike
        enriched.append(rec)
        per_label_len[label].append(n_frames)
        if not np.isnan(spike):
            per_label_spike[label].append(spike)

    # Per-label robust z normalization baselines
    label_len_z_lookup: Dict[str, np.ndarray] = {}
    label_spike_z_lookup: Dict[str, np.ndarray] = {}
    for label, vals in per_label_len.items():
        label_len_z_lookup[label] = robust_z(np.array(vals, dtype=np.float32))
    for label, vals in per_label_spike.items():
        label_spike_z_lookup[label] = robust_z(np.array(vals, dtype=np.float32))

    # Build index trackers to map row->z within each label array
    len_idx: Dict[str, int] = defaultdict(int)
    spike_idx: Dict[str, int] = defaultdict(int)

    flagged = []
    for rec in enriched:
        label = rec["word"]
        conf = float(rec.get("confidence", "0") or 0)
        correct = str(rec.get("correct", "")).lower() in {"true", "1"}
        z_len = float(label_len_z_lookup[label][len_idx[label]])
        len_idx[label] += 1

        spike = rec["spike_score"]
        if np.isnan(spike) or len(per_label_spike[label]) == 0:
            z_spike = np.nan
        else:
            z_spike = float(label_spike_z_lookup[label][spike_idx[label]])
            spike_idx[label] += 1

        reasons = []
        score = 0

        if (not correct) and conf >= th.high_conf_wrong:
            reasons.append("high_conf_wrong")
            score += 2
        if abs(z_len) >= th.z_len:
            reasons.append("length_outlier")
            score += 1
        if not np.isnan(z_spike) and abs(z_spike) >= th.z_spike:
            reasons.append("spike_outlier")
            score += 1

        if score >= th.min_score:
            rec_out = {
                "word": rec["word"],
                "predicted": rec["predicted"],
                "correct": rec["correct"],
                "confidence": f"{conf:.1f}",
                "source": rec["source"],
                "sample": rec["sample"],
                "n_frames": int(rec["n_frames_f"]),
                "z_len": f"{z_len:.2f}",
                "spike_score": "" if np.isnan(spike) else f"{spike:.3f}",
                "z_spike": "" if np.isnan(z_spike) else f"{z_spike:.2f}",
                "anomaly_score": score,
                "reasons": "|".join(reasons),
                "path": rec["path"],
            }
            flagged.append(rec_out)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        fields = [
            "word", "predicted", "correct", "confidence", "source", "sample",
            "n_frames", "z_len", "spike_score", "z_spike",
            "anomaly_score", "reasons", "path",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(flagged, key=lambda x: (-int(x["anomaly_score"]), float(x["confidence"]))))

    with open(args.manifest, "w") as f:
        for row in flagged:
            f.write(f"{row['path']}\n")

    print(f"Analyzed rows: {len(rows)}")
    print(f"Flagged anomalies: {len(flagged)}")
    print(f"Candidates CSV: {args.out}")
    print(f"Quarantine manifest: {args.manifest}")


if __name__ == "__main__":
    main()
