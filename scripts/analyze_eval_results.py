"""
analyze_eval_results.py — Analyze eval_model.py detailed output
===============================================================
Reads models/eval_results.csv and prints:
  1) confusion matrix (actual rows, predicted columns)
  2) per-class precision / recall / F1 / support
  3) macro and weighted averages

Usage:
  python3 scripts/analyze_eval_results.py
  python3 scripts/analyze_eval_results.py models/eval_results.csv
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List

DEFAULT_RESULTS_PATH = "models/eval_results.csv"


def _safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def _load_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")

    rows: List[Dict[str, str]] = []
    with open(path, newline="") as f:
        filtered_lines = (line for line in f if not line.startswith("#"))
        reader = csv.DictReader(filtered_lines)
        required = {"word", "predicted"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must contain columns {sorted(required)}; got {reader.fieldnames}"
            )
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No evaluation rows found in {path}")

    return rows


def _print_confusion_matrix(labels: List[str], cm: Dict[str, Dict[str, int]]) -> None:
    print("\nConfusion Matrix (actual x predicted)")
    row_label_w = max(6, max(len(l) for l in labels))
    cell_w = max(6, max(len(l) for l in labels))

    header = "actual\\pred".ljust(row_label_w) + " | " + " ".join(
        l.rjust(cell_w) for l in labels
    )
    print(header)
    print("-" * len(header))

    for actual in labels:
        values = " ".join(str(cm[actual][pred]).rjust(cell_w) for pred in labels)
        print(f"{actual.ljust(row_label_w)} | {values}")


def _print_classification_report(
    labels: List[str], cm: Dict[str, Dict[str, int]], support: Dict[str, int]
) -> None:
    print("\nPer-class metrics")
    print("class    precision   recall      f1   support")
    print("---------------------------------------------")

    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0
    weighted_p = 0.0
    weighted_r = 0.0
    weighted_f1 = 0.0
    total = sum(support.values())

    for label in labels:
        tp = cm[label][label]
        fp = sum(cm[actual][label] for actual in labels if actual != label)
        fn = sum(cm[label][pred] for pred in labels if pred != label)

        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * p * r, p + r)
        s = support[label]

        macro_p += p
        macro_r += r
        macro_f1 += f1
        weighted_p += p * s
        weighted_r += r * s
        weighted_f1 += f1 * s

        print(f"{label:<6} {p:>9.3f} {r:>8.3f} {f1:>8.3f} {s:>9d}")

    n = len(labels)
    print("---------------------------------------------")
    print(
        f"{'macro':<6} {macro_p / n:>9.3f} {macro_r / n:>8.3f} {macro_f1 / n:>8.3f} {total:>9d}"
    )
    print(
        f"{'weighted':<6} {weighted_p / total:>9.3f} {weighted_r / total:>8.3f} {weighted_f1 / total:>8.3f} {total:>9d}"
    )


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_PATH
    rows = _load_rows(path)

    labels = sorted(
        set(r["word"] for r in rows) | set(r["predicted"] for r in rows if r["predicted"])
    )
    cm: Dict[str, Dict[str, int]] = {
        actual: {pred: 0 for pred in labels} for actual in labels
    }
    support: Dict[str, int] = defaultdict(int)

    for r in rows:
        actual = r["word"]
        pred = r["predicted"] if r["predicted"] else "<blank>"
        if pred not in cm[actual]:
            for a in labels:
                cm[a][pred] = 0
            labels.append(pred)
        cm[actual][pred] += 1
        support[actual] += 1

    total = len(rows)
    correct = sum(cm[label][label] for label in labels if label in cm[label])
    acc = _safe_div(correct, total)

    print(f"Loaded {total} rows from {path}")
    print(f"Overall accuracy: {acc * 100:.2f}% ({correct}/{total})")

    _print_confusion_matrix(labels, cm)
    _print_classification_report(labels, cm, support)


if __name__ == "__main__":
    main()
