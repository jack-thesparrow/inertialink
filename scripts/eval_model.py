"""
eval_model.py — Batch evaluation of the exported ONNX model
============================================================
Runs every synthetic sample (12 words × 200 samples = 2400 total) through
the ONNX model, collects per-sample predictions and confidence, and writes
two output files:

  models/eval_results.csv   — one row per sample (for detailed analysis)
  models/eval_summary.csv   — one row per word   (accuracy / confidence)

Usage:
  python3 scripts/eval_model.py                            # all words, augmented only
  python3 scripts/eval_model.py 1 2 3                      # specific labels
  python3 scripts/eval_model.py --source seed              # evaluate only seed data
  python3 scripts/eval_model.py --source both 1 2 3 A B C  # evaluate both sources
"""

import os
import sys
import glob
import csv
import numpy as np
import pandas as pd
import onnxruntime as ort
from datetime import datetime

# ── Constants ─────────────────────────────────────────────────────────────────
ALPHABET      = "~ 123ABC"
MODEL_PATH    = "models/pen_model.onnx"
DATA_DIR      = "data"
RESULTS_PATH  = "models/eval_results.csv"
SUMMARY_PATH  = "models/eval_summary.csv"

ALL_WORDS = ["1", "2", "3", "A", "B", "C"]
VALID_SOURCES = {"augmented", "seed", "both"}


def create_ort_session(model_path: str) -> ort.InferenceSession:
    """Create an ONNX Runtime session with Intel Arc-first provider order.

    MODEL_DEVICE env:
      - auto (default): try OpenVINO GPU first, then CPU
      - xpu/gpu: force OpenVINO GPU first, then CPU fallback
      - cpu: CPU only
    """
    requested = os.getenv("MODEL_DEVICE", "auto").strip().lower()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available = ort.get_available_providers()
    providers = []

    wants_gpu = requested in {"auto", "xpu", "gpu"}
    if wants_gpu and "OpenVINOExecutionProvider" in available:
        providers.append(("OpenVINOExecutionProvider", {"device_type": "GPU"}))
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")

    # If OpenVINO isn't present and GPU was requested, fall back cleanly.
    if requested in {"xpu", "gpu"} and not any(
        (p == "OpenVINOExecutionProvider" or (isinstance(p, tuple) and p[0] == "OpenVINOExecutionProvider"))
        for p in providers
    ):
        print("[Warning] OpenVINOExecutionProvider not available. Falling back to CPU.")

    if requested == "cpu":
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    print(f"ONNX providers requested: {providers}")
    print(f"ONNX providers active   : {sess.get_providers()}")
    return sess


def parse_args(argv: list[str]) -> tuple[list[str], str]:
    """Parse positional words and --source flag without breaking old usage."""
    words: list[str] = []
    source = "augmented"

    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--source":
            if i + 1 >= len(argv):
                print("[Error] --source requires one value: augmented|seed|both")
                sys.exit(1)
            source = argv[i + 1].strip().lower()
            i += 2
            continue
        if token.startswith("--source="):
            source = token.split("=", 1)[1].strip().lower()
            i += 1
            continue
        words.append(token)
        i += 1

    if source not in VALID_SOURCES:
        print(f"[Error] Invalid source '{source}'. Choose from: augmented, seed, both.")
        sys.exit(1)

    if not words:
        words = ALL_WORDS.copy()
    return words, source


def get_source_folders(word: str, source: str) -> list[tuple[str, str]]:
    """Return [(source_tag, folder_path), ...] for a label."""
    folders: list[tuple[str, str]] = []
    if source in {"augmented", "both"}:
        folders.append(("augmented", os.path.join(DATA_DIR, word)))
    if source in {"seed", "both"}:
        folders.append(("seed", os.path.join(DATA_DIR, "seed", word)))
    return folders


# ── CTC helpers ───────────────────────────────────────────────────────────────
def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax.  Input shape: (T, C)."""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def ctc_greedy_decode(logits: np.ndarray):
    """CTC greedy decoder — mirrors decoder_main.cpp exactly.

    Returns
    -------
    predicted   : str   decoded text
    confidence  : float mean probability of emitted characters (0–100)
    per_char    : str   "h=99% e=49% l=95%" style breakdown
    """
    probs       = _softmax(logits)           # (T, C)
    best_cls    = probs.argmax(axis=-1)      # (T,)
    best_prob   = probs.max(axis=-1)         # (T,)

    chars: list[tuple[str, float]] = []
    last = 0
    for cls, prob in zip(best_cls, best_prob):
        if cls != 0 and cls != last:
            chars.append((ALPHABET[cls], float(prob)))
        last = int(cls)

    predicted  = "".join(c for c, _ in chars)
    confidence = (sum(p for _, p in chars) / len(chars) * 100) if chars else 0.0
    per_char   = "  ".join(f"{c}={int(p * 100)}%" for c, p in chars)
    return predicted, confidence, per_char


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    words, source = parse_args(sys.argv[1:])

    # Validate requested words
    unknown = [w for w in words if w not in ALL_WORDS]
    if unknown:
        print(f"[Warning] Not in training vocab: {', '.join(unknown)}")

    if not os.path.exists(MODEL_PATH):
        print(f"[Error] {MODEL_PATH} not found — run train_bilstm.py first.")
        sys.exit(1)

    print(f"Loading {MODEL_PATH} ...")
    sess       = create_ort_session(MODEL_PATH)
    input_name = sess.get_inputs()[0].name
    print(f"Model input : {input_name}  shape {sess.get_inputs()[0].shape}")
    print(f"Source      : {source}")
    print(f"Evaluating  : {', '.join(words)}\n")

    all_rows   : list[dict] = []
    word_stats : list[dict] = []
    errors     : list[dict] = []   # wrong predictions for the summary

    for word in words:
        csvs: list[tuple[str, str]] = []
        for src_tag, folder in get_source_folders(word, source):
            for csv_path in sorted(glob.glob(f"{folder}/sample_*.csv")):
                csvs.append((src_tag, csv_path))
        if not csvs:
            print(f"  [Skip] No samples found for '{word}' in selected source(s)")
            continue

        correct     = 0
        confidences = []
        source_counts: dict[str, int] = {}

        for src_tag, csv_path in csvs:
            sample_num = os.path.splitext(os.path.basename(csv_path))[0].replace("sample_", "")

            df      = pd.read_csv(csv_path)
            x       = df[["ax", "ay", "az", "gx", "gy", "gz"]].values.astype(np.float32)[np.newaxis]  # (1,T,6)
            logits  = sess.run(None, {input_name: x})[0][0]                              # (T,C)

            predicted, conf, per_char = ctc_greedy_decode(logits)
            is_correct = (predicted == word)
            if is_correct:
                correct += 1
            confidences.append(conf)

            row = {
                "word":       word,
                "sample":     sample_num,
                "source":     src_tag,
                "path":       csv_path,
                "predicted":  predicted,
                "correct":    is_correct,
                "confidence": f"{conf:.1f}",
                "per_char":   per_char,
                "n_frames":   len(df),
            }
            all_rows.append(row)
            if not is_correct:
                errors.append(row)
            source_counts[src_tag] = source_counts.get(src_tag, 0) + 1

        n          = len(csvs)
        accuracy   = correct / n * 100
        avg_conf   = sum(confidences) / n
        min_conf   = min(confidences)
        max_conf   = max(confidences)

        word_stats.append({
            "word":        word,
            "samples":     n,
            "correct":     correct,
            "wrong":       n - correct,
            "accuracy_%":  f"{accuracy:.1f}",
            "avg_conf_%":  f"{avg_conf:.1f}",
            "min_conf_%":  f"{min_conf:.1f}",
            "max_conf_%":  f"{max_conf:.1f}",
        })

        bar_filled = int(accuracy / 5)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        source_txt = ", ".join(f"{k}:{v}" for k, v in sorted(source_counts.items()))
        print(f"  {word:<8}  [{bar}] {accuracy:5.1f}%  "
              f"({correct}/{n})  avg conf {avg_conf:.0f}%  sources[{source_txt}]")

    if not word_stats:
        print("[Error] No data found. Run augment_seed_data.py first.")
        return

    # ── Totals ────────────────────────────────────────────────────────────────
    total_samples = sum(int(s["samples"]) for s in word_stats)
    total_correct = sum(int(s["correct"]) for s in word_stats)
    total_acc     = total_correct / total_samples * 100 if total_samples else 0
    avg_conf_all  = sum(float(s["avg_conf_%"]) for s in word_stats) / len(word_stats)

    print(f"\n{'─'*60}")
    print(f"  {'TOTAL':<8}  {total_correct}/{total_samples} correct  "
          f"overall accuracy: {total_acc:.1f}%  avg conf: {avg_conf_all:.1f}%")
    print(f"{'─'*60}")

    # ── Worst predictions ─────────────────────────────────────────────────────
    if errors:
        print(f"\n  Wrong predictions ({len(errors)} / {total_samples}):")
        # Show at most 20 examples, sorted by confidence ascending (most uncertain first)
        errors_sorted = sorted(errors, key=lambda r: float(r["confidence"]))
        for e in errors_sorted[:20]:
            print(f"    {e['word']:<8} → \"{e['predicted']}\"  "
                  f"conf={e['confidence']}%  sample={e['sample']}  src={e['source']}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more (see {RESULTS_PATH})")

    # ── Write files ───────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(RESULTS_PATH, "w", newline="") as f:
        fields = ["word", "sample", "source", "path", "predicted", "correct",
                  "confidence", "per_char", "n_frames"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        # Metadata comment line (readable in text editor, ignored by CSV parsers
        # that skip non-matching rows)
        f.write(f"# generated {ts}  model={MODEL_PATH}  source={source}\n")
        w.writerows(all_rows)

    with open(SUMMARY_PATH, "w", newline="") as f:
        fields = ["word", "samples", "correct", "wrong",
                  "accuracy_%", "avg_conf_%", "min_conf_%", "max_conf_%"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(word_stats)
        # Totals row
        w.writerow({
            "word":       "TOTAL",
            "samples":    total_samples,
            "correct":    total_correct,
            "wrong":      total_samples - total_correct,
            "accuracy_%": f"{total_acc:.1f}",
            "avg_conf_%": f"{avg_conf_all:.1f}",
            "min_conf_%": "",
            "max_conf_%": "",
        })

    print(f"\n  Detailed results : {RESULTS_PATH}")
    print(f"  Summary          : {SUMMARY_PATH}")
    print(f"  Evaluated at     : {ts}")


if __name__ == "__main__":
    main()
