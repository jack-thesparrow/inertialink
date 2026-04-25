# scripts / train_bilstm.py
import os

# ── Intel threading knobs — must be set BEFORE torch is imported ─────────────
# KMP_BLOCKTIME=0 : idle OpenMP threads sleep immediately (no spin-wait CPU burn)
# KMP_AFFINITY    : pin threads compactly to physical cores for better cache reuse
os.environ.setdefault("OMP_NUM_THREADS",  str(os.cpu_count() or 4))
os.environ.setdefault("KMP_BLOCKTIME",    "0")
os.environ.setdefault("KMP_AFFINITY",     "granularity=fine,compact,1,0")

import glob
import random
import math
import sys
import datetime
import time
from itertools import groupby
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
import numpy as np


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------
# 0. DEVICE SELECTION
#    Priority: Intel Arc native XPU → IPEX XPU → NVIDIA CUDA → CPU
#
#    PyTorch 2.4+ has built-in XPU support for Intel Arc — no IPEX needed.
#    IPEX is tried as a fallback for older PyTorch versions.
#    Install Intel drivers: https://dgpu-docs.intel.com/driver/installation.html
# ---------------------------------------------------------
def get_device() -> torch.device:
    forced = os.getenv("TRAIN_DEVICE", "auto").strip().lower()
    if forced in {"cpu", "cuda", "xpu"}:
        return torch.device(forced)
    # PyTorch 2.4+ native XPU — works without IPEX on modern Intel Arc drivers
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
    except Exception:
        pass
    # Older PyTorch: try IPEX (version must match PyTorch exactly)
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        if torch.xpu.is_available():
            return torch.device("xpu")
    except Exception:
        pass
    # NVIDIA GPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    # CPU fallback
    return torch.device("cpu")


DEVICE = get_device()

# ── CPU performance flags ────────────────────────────────────
# All cores for BLAS/MKL (matrix multiply inside LSTM)
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(max(1, os.cpu_count() // 2))
# Allow slightly lower float32 precision for faster matmul on Intel
torch.set_float32_matmul_precision("high")
# CUDA-specific throughput flags
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
# Fuse adjacent oneDNN ops (free ~5-10% on Intel CPUs)
torch.jit.enable_onednn_fusion(True)

# Mixed precision (AMP): automatically uses fp16 for CUDA, bf16 for XPU/CPU.
# Falls back to float32 silently on standard CPUs.
if DEVICE.type == "cuda":
    _cuda_bf16_ok = torch.cuda.is_bf16_supported()
    USE_AMP = True
    AMP_DTYPE = torch.bfloat16 if _cuda_bf16_ok else torch.float16
elif DEVICE.type == "xpu":
    USE_AMP = True
    AMP_DTYPE = torch.bfloat16
elif DEVICE.type == "cpu":
    # oneDNN bf16 autocast on CPU is often faster on modern Intel chips.
    USE_AMP = True
    AMP_DTYPE = torch.bfloat16
else:
    USE_AMP = False
    AMP_DTYPE = torch.float32
USE_BF16 = AMP_DTYPE == torch.bfloat16

# ---------------------------------------------------------
# 1. ALPHABET & HYPERPARAMETERS
# ---------------------------------------------------------
# Index 0 is the CTC blank token (single placeholder char '~', never printed).
# Restrict training to the target classes only.
TARGET_LABELS = ["1", "2", "3", "A", "B", "C"]
ALPHABET = "~ " + "".join(TARGET_LABELS)
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALPHABET)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(ALPHABET)}

INPUT_FEATURES = 6   # (ax, ay, az, gx, gy, gz)
HIDDEN_SIZE    = 64  # Smaller vocabulary = can run an even smaller network
NUM_LAYERS     = 2    
NUM_CLASSES    = len(ALPHABET)
CONV_CHANNELS  = 64   # Conv1d frontend output channels (feeds into LSTM)

EPOCHS             = int(os.getenv("TRAIN_EPOCHS", "150"))
BASE_BATCH_SIZE    = int(os.getenv("TRAIN_BATCH_SIZE", "64"))
WARMUP_EPOCHS      = 5     # Short warmup with higher base LR
BASE_LR            = 1e-3  # Scaled for batch 512 (sqrt scaling from 5e-4 @ 128)
PATIENCE           = 60    # Tighter early stop; conv frontend converges faster
CHECKPOINT_EVERY   = 10    # Reduce checkpoint I/O overhead (was 2)
CHECKPOINT_PATH    = "models/checkpoint.pt"
LOG_PATH           = "models/training_log.csv"
VAL_SPLIT          = 0.15
SPLIT_SEED         = 42
EMA_DECAY          = float(os.getenv("TRAIN_EMA_DECAY", "0.999"))
GPU_FAST_MODE      = os.getenv("TRAIN_GPU_FAST", "1").strip().lower() not in {"0", "false", "no"}
VAL_EVERY          = max(1, _env_int("TRAIN_VAL_EVERY", 3))
PROGRESS_EVERY     = max(1, _env_int("TRAIN_PROGRESS_EVERY", 25))
FORCE_COMPILE      = os.getenv("TRAIN_FORCE_COMPILE", "0").strip().lower() in {"1", "true", "yes"}
FROM_SCRATCH       = os.getenv("TRAIN_FROM_SCRATCH", "0").strip().lower() in {"1", "true", "yes"}
DISABLE_COMPILE    = os.getenv("TRAIN_DISABLE_COMPILE", "0").strip().lower() in {"1", "true", "yes"}
VAL_METRICS_EVERY  = max(1, _env_int("TRAIN_VAL_METRICS_EVERY", 3))


# ---------------------------------------------------------
# 2. NEURAL NETWORK ARCHITECTURE
# ---------------------------------------------------------
class SmartPenDecoder(nn.Module):
    """Conv1d + BiLSTM CTC decoder with baked-in z-score normalization.

    The Conv1d frontend (stride 2) halves the sequence length before the
    LSTM, cutting LSTM compute by ~2x — the single biggest training speed
    win.  Normalization stats are registered as buffers so they travel
    into the ONNX export transparently.
    """

    def __init__(self, input_mean: torch.Tensor, input_std: torch.Tensor):
        super(SmartPenDecoder, self).__init__()
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std",  input_std)

        # Conv frontend: (B, 6, T) → (B, 64, T//2)
        # Halves sequence length, enriches features 6→64 for the LSTM.
        self.conv = nn.Sequential(
            nn.Conv1d(INPUT_FEATURES, CONV_CHANNELS,
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(CONV_CHANNELS),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

        self.lstm = nn.LSTM(
            input_size=CONV_CHANNELS,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.fc = nn.Linear(HIDDEN_SIZE * 2, NUM_CLASSES)

    @staticmethod
    def conv_out_len(length):
        """Output length after stride-2 conv: (L-1)//2 + 1."""
        return (length - 1) // 2 + 1

    def forward(self, x, lengths=None):
        # x: (B, T, 6) raw sensor values

        x_accel = x[:, :, :3]
        x_gyro  = x[:, :, 3:]

        if lengths is not None:
            # Create mask for valid frames
            mask = (torch.arange(x.size(1), device=x.device)[None, :] < lengths.to(x.device)[:, None]).float()
            
            # Local DC Offset Removal for Accelerometer (removes pen tilt/gravity)
            # Sum only valid frames, then divide by the valid length
            accel_sum = (x_accel * mask.unsqueeze(-1)).sum(dim=1)
            local_accel_mean = accel_sum / lengths.to(x.device).float().unsqueeze(-1)
            x_accel = x_accel - local_accel_mean.unsqueeze(1)
            
            # Gyroscope just uses the global mean (sensor bias)
            x_gyro = x_gyro - self.input_mean[3:]
            
            x_centered = torch.cat([x_accel, x_gyro], dim=-1)
            
            # Zero out padding frames so they remain exactly 0.0
            x_centered = x_centered * mask.unsqueeze(-1)
        else:
            # Inference mode (unpadded)
            local_accel_mean = x_accel.mean(dim=1, keepdim=True)
            x_accel = x_accel - local_accel_mean
            x_gyro = x_gyro - self.input_mean[3:]
            x_centered = torch.cat([x_accel, x_gyro], dim=-1)

        # Apply global standard deviation
        x = x_centered / (self.input_std + 1e-8)

        # Conv1d wants (B, C, T)
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # → (B, T//2, 64)

        if lengths is not None:
            lengths = self.conv_out_len(lengths).clamp(min=1)
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        return logits


DATASET_CACHE = "data/dataset_cache_123ABC.pt"

# Directories to scan for training data.  Both are merged into one dataset.
#   data/       — augmented synthetic data from augment_seed_data.py
#   data/seed/  — real hardware collected seed data from data_collector
#   data/hard/  — targeted hard-example augmentation from eval mistakes
DATA_DIRS = ["data", "data/seed", "data/hard"]

# ---------------------------------------------------------
# 3. DATA LOADING
# ---------------------------------------------------------
def _any_csv_newer_than(dirs: list[str], cache_path: str) -> bool:
    """Return True if any CSV in `dirs` is newer than `cache_path`."""
    if not os.path.exists(cache_path):
        return True
    cache_mtime = os.path.getmtime(cache_path)
    for d in dirs:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(".csv") and os.path.getmtime(os.path.join(root, f)) > cache_mtime:
                    return True
    return False


def load_dataset():
    """Load training tensors from cache or build from CSVs.

    Scans both data/ (synthetic) and data_real/ (hardware-collected) and
    merges every CSV into one dataset.  A single cache file is saved for
    fast reloads.  The cache is automatically invalidated if any CSV is
    newer than the cache file.

    Delete data/dataset_cache.pt manually if you want to force a rebuild.
    
    Returns:
        sequences: list of sensor tensors
        targets: list of target label tensors
        weights: list of sample weights (3.0 for seed data, 1.0 for synthetic)
    """
    if os.path.exists(DATASET_CACHE) and not _any_csv_newer_than(DATA_DIRS, DATASET_CACHE):
        print(f"[Dataset] Loading cache from {DATASET_CACHE} ...")
        cached = torch.load(DATASET_CACHE, map_location="cpu")
        return cached["X"], cached["Y"], cached.get("weights", [1.0] * len(cached["X"]))

    print("[Dataset] Building dataset from CSVs ...")
    sequences, targets, weights = [], [], []
    source_counts: dict[str, int] = {}

    valid_folders = set(TARGET_LABELS)
    for data_dir in DATA_DIRS:
        if not os.path.exists(data_dir):
            continue
        for label_folder in sorted(os.listdir(data_dir)):
            # Include seed folder when data_dir is "data" (synthetic data)
            # This allows training on both synthetic and real data
            if data_dir == "data" and label_folder == "seed":
                # This is the real seed data - include it!
                pass
            if label_folder not in valid_folders:
                continue
            folder_path = os.path.join(data_dir, label_folder)
            if not os.path.isdir(folder_path):
                continue
            csv_files = glob.glob(f"{folder_path}/*.csv")
            if not csv_files:
                continue
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                if any(col not in df.columns for col in ["ax", "ay", "az", "gx", "gy", "gz"]):
                    continue
                target_chars = [c for c in label_folder if c in CHAR_TO_IDX]
                if not target_chars:
                    continue
                
                # Weight real data more heavily than synthetic
                sample_weight = 3.0 if "seed" in csv_file else 1.0
                
                sequences.append(
                    torch.tensor(df[["ax", "ay", "az", "gx", "gy", "gz"]].values, dtype=torch.float32)
                )
                targets.append(
                    torch.tensor(
                        [CHAR_TO_IDX[c] for c in target_chars],
                        dtype=torch.long,
                    )
                )
                weights.append(sample_weight)
            count = len(csv_files)
            key = f"{data_dir}/{label_folder}"
            source_counts[key] = count

    if not sequences:
        print("[Error] No training data found!  Collect data using ./bin/data_collector first")
        return [], [], []

    # Print source breakdown
    for src, cnt in sorted(source_counts.items()):
        tag = "seed" if "seed" in src else "aug "
        print(f"  [{tag}] {src}: {cnt} samples")

    os.makedirs(os.path.dirname(DATASET_CACHE), exist_ok=True)
    torch.save({"X": sequences, "Y": targets, "weights": weights}, DATASET_CACHE)
    print(f"[Dataset] Cached {len(sequences)} total samples to {DATASET_CACHE}")
    return sequences, targets, weights


def _build_balanced_indices(Y_train):
    """Build class-balanced sample indices for one epoch."""
    class_to_indices: dict[int, list[int]] = {}
    for idx, y in enumerate(Y_train):
        if y.numel() == 0:
            continue
        class_id = int(y[0].item())
        class_to_indices.setdefault(class_id, []).append(idx)

    if not class_to_indices:
        return []

    max_count = max(len(v) for v in class_to_indices.values())
    balanced = []
    for _, idxs in class_to_indices.items():
        if len(idxs) < max_count:
            balanced.extend(random.choices(idxs, k=max_count))
        else:
            balanced.extend(idxs)
    random.shuffle(balanced)
    return balanced


def _stratified_train_val_split(X, Y, W, val_ratio=0.15, seed=42):
    """Split samples into train/val per class to avoid skew."""
    rng = random.Random(seed)
    class_to_indices: dict[int, list[int]] = {}
    for idx, y in enumerate(Y):
        if y.numel() == 0:
            continue
        class_id = int(y[0].item())
        class_to_indices.setdefault(class_id, []).append(idx)

    train_idx, val_idx = [], []
    for _, idxs in class_to_indices.items():
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio)))
        n_val = min(n_val, max(0, len(idxs) - 1))
        if n_val == 0:
            train_idx.extend(idxs)
        else:
            val_idx.extend(idxs[:n_val])
            train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    X_train = [X[i] for i in train_idx]
    Y_train = [Y[i] for i in train_idx]
    W_train = [W[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    Y_val = [Y[i] for i in val_idx]
    W_val = [W[i] for i in val_idx]
    return X_train, Y_train, W_train, X_val, Y_val, W_val


# ---------------------------------------------------------
# 4. GLOBAL BATCH PRE-BUILDER
# ---------------------------------------------------------
def prep_global_tensors(X_train, Y_train, device):
    """Pad the entire dataset to a single fixed tensor up-front.
    Eliminates PyTorch padding overhead during training and locks the
    batch sequence length so torch.compile only compiles exactly once.
    """
    print("[Dataset] Padding entire dataset globally...")
    # Keep tensors on host for GPU/XPU runs to avoid VRAM spikes on smaller GPUs.
    keep_on_host = device.type in ("cuda", "xpu")
    xs = X_train if keep_on_host else [x.to(device) for x in X_train]
    X_pad = pad_sequence(xs, batch_first=True)  # (N, T_max, 6)
    if keep_on_host and torch.cuda.is_available():
        X_pad = X_pad.pin_memory()
    
    lengths_t = torch.tensor([x.size(0) for x in X_train], dtype=torch.long)
    targets_flat = torch.cat(Y_train)
    tgt_lengths_t = torch.tensor([y.size(0) for y in Y_train], dtype=torch.long)
    
    # Pre-calculate slice indices for the flattened targets
    target_start_idx = torch.zeros(len(Y_train) + 1, dtype=torch.long)
    target_start_idx[1:] = tgt_lengths_t.cumsum(dim=0)
    
    return X_pad, lengths_t, targets_flat, tgt_lengths_t, target_start_idx


def resolve_batch_size(base_batch_size: int, device: torch.device) -> int:
    """Choose a safe high-throughput batch size for the current backend.

    You can still override manually with TRAIN_BATCH_SIZE.
    """
    # Respect explicit override from env first.
    env_bs = os.getenv("TRAIN_BATCH_SIZE")
    if env_bs is not None:
        return max(8, _env_int("TRAIN_BATCH_SIZE", base_batch_size))

    bs = base_batch_size
    if device.type == "cuda":
        total_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        if total_mem_gb <= 4.5:
            bs = min(bs, 64)
        elif total_mem_gb <= 8.5:
            bs = min(bs, 128)
        else:
            bs = min(bs, 256)
    elif device.type == "xpu":
        # Arc/mobile XPU cards commonly sit in this range; keep it conservative.
        # You can raise via TRAIN_BATCH_SIZE if your card is stable.
        # No hardcoded limit - let user control via environment variable
        pass
    else:
        # CPU benefits from moderate batch size; too large hurts cache locality.
        cpu_cores = os.cpu_count() or 8
        if cpu_cores <= 8:
            bs = min(bs, 64)
        elif cpu_cores <= 16:
            bs = min(bs, 128)
        else:
            bs = min(bs, 192)
    return max(8, bs)


# ---------------------------------------------------------
# 5. TRAINING LOG HELPERS
# ---------------------------------------------------------
def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_session_start(start_epoch: int) -> None:
    """Write a SESSION START marker.

    First checks whether the previous session ended with a SESSION END marker.
    If not (power cut, force restart, OOM kill), it writes a SESSION KILLED
    line so the log always explains every gap.
    """
    os.makedirs("models", exist_ok=True)
    needs_header = not os.path.exists(LOG_PATH)

    if not needs_header:
        # Scan backward for the last non-blank line
        with open(LOG_PATH) as f:
            lines = [l.rstrip() for l in f if l.strip()]
        if lines and lines[-1].startswith("--- SESSION START"):
            # Previous run had no SESSION END → was killed externally
            with open(LOG_PATH, "a") as f:
                f.write(f"--- SESSION KILLED | detected {_now()} ---\n")

    with open(LOG_PATH, "a") as f:
        if needs_header:
                f.write("epoch,train_loss,val_loss,best_val_loss,lr,no_improve,timestamp\n")
        action = f"resuming from epoch {start_epoch}" if start_epoch > 0 else "fresh start"
        f.write(f"--- SESSION START | {_now()} | {action} ---\n")


def _log_session_end(reason: str, best_loss: float, epoch: int) -> None:
    """Write a SESSION END marker with the reason training stopped."""
    with open(LOG_PATH, "a") as f:
        f.write(
            f"--- SESSION END | {reason} | "
            f"stopped at epoch {epoch} | best_loss={best_loss:.6f} | {_now()} ---\n"
        )


def _levenshtein(a: list[int], b: list[int]) -> int:
    """Classic DP edit-distance for token sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _ctc_greedy_decode_ids(logits_btq: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    """Greedy CTC decode -> token id lists, stripping blanks/repeats."""
    pred = logits_btq.argmax(dim=2)  # (B, T)
    out: list[list[int]] = []
    for i in range(pred.size(0)):
        t = int(lengths[i].item())
        seq = pred[i, :t].tolist()
        # Collapse repeats, then remove CTC blank (0).
        seq = [k for k, _ in groupby(seq)]
        seq = [tok for tok in seq if tok != 0]
        out.append(seq)
    return out


# ---------------------------------------------------------
# 6. LR SCHEDULE
# ---------------------------------------------------------
def get_lr(epoch: int, total_epochs: int, warmup: int, base_lr: float) -> float:
    """Linear warm-up then cosine annealing to 5 % of base_lr."""
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total_epochs - warmup)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr   = base_lr * 0.05
    return min_lr + (base_lr - min_lr) * cosine


# ---------------------------------------------------------
# 5. TRAINING LOOP
# ---------------------------------------------------------
def train_and_export():
    print(f"=== Smart Pen CTC Trainer ({HIDDEN_SIZE}-unit BiLSTM × {NUM_LAYERS}, cosine LR) ===")
    print(f"Device: {DEVICE}")
    
    # ── Training Time Tracking ────────────────────────────────────────
    training_start_time = time.time()
    epoch_start_times = []
    
    X_all, Y_all, W_all = load_dataset()

    if len(X_all) == 0:
        return

    print(f"Loaded {len(X_all)} stroke samples.")
    X_train, Y_train, W_train, X_val, Y_val, W_val = _stratified_train_val_split(
        X_all, Y_all, W_all, val_ratio=VAL_SPLIT, seed=SPLIT_SEED
    )
    if len(X_train) == 0 or len(X_val) == 0:
        print("[Error] Train/val split failed. Please check class data availability.")
        return
    print(f"Split: train={len(X_train)} | val={len(X_val)} (stratified)")

    # --- Compute normalization stats from the whole training set (on CPU) ---
    all_frames = torch.cat(X_train, dim=0)
    feat_mean  = all_frames.mean(dim=0)     # (6,)
    feat_std   = all_frames.std(dim=0)      # (6,)
    print(f"Feature mean: {feat_mean.tolist()}")
    print(f"Feature std:  {feat_std.tolist()}")

    model = SmartPenDecoder(feat_mean, feat_std).to(DEVICE)
    # Prefer fused AdamW where supported; otherwise fallback to foreach Adam.
    if DEVICE.type == "cuda":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=BASE_LR,
            weight_decay=1e-5,
            fused=True,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=BASE_LR,
            weight_decay=1e-5,
            foreach=True,
        )
    ctc_loss  = nn.CTCLoss(blank=0, zero_infinity=True, reduction='none')
    scaler = GradScaler(device="cuda", enabled=(DEVICE.type == "cuda"))

    # Test whether CTCLoss runs natively on this device (CUDA yes, XPU maybe).
    # If it works, logits never leave the device — no GPU stalls.
    # If it fails, we fall back to .cpu() transfer per batch.
    _ctc_native = False
    try:
        _dummy_log  = torch.zeros(5, 1, NUM_CLASSES, device=DEVICE).log_softmax(2)
        _dummy_tgt  = torch.tensor([1, 2], dtype=torch.long)
        _dummy_ilen = torch.tensor([5], dtype=torch.long)
        _dummy_tlen = torch.tensor([2], dtype=torch.long)
        ctc_loss(_dummy_log, _dummy_tgt, _dummy_ilen, _dummy_tlen)
        _ctc_native = True
        print(f"CTCLoss: running natively on {DEVICE} (no CPU transfers)")
    except Exception:
        print(f"CTCLoss: falling back to CPU (XPU kernel unavailable)")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    batch_size = resolve_batch_size(BASE_BATCH_SIZE, DEVICE)
    run_fast = GPU_FAST_MODE and DEVICE.type in ("cuda", "xpu")
    checkpoint_every = max(CHECKPOINT_EVERY, 25) if run_fast else CHECKPOINT_EVERY
    val_every = VAL_EVERY if run_fast else 1
    print(f"Training for {EPOCHS} epochs, batch size {batch_size}.")
    print(
        f"Fast mode: {'ON' if run_fast else 'OFF'} | "
        f"val_every={val_every} | checkpoint_every={checkpoint_every}"
    )
    print("-" * 60)

    # Pad the entire dataset upfront globally. 
    # This locks sequence lengths to T_max and speeds up torch.compile massively.
    X_pad_global, lengths_global, targets_global, tgt_lengths_global, tgt_starts_global = prep_global_tensors(X_train, Y_train, DEVICE)
    X_pad_val, lengths_val, targets_val, tgt_lengths_val, tgt_starts_val = prep_global_tensors(X_val, Y_val, DEVICE)
    
    # Convert weights to tensors for efficient indexing
    W_train_tensor = torch.tensor(W_train, dtype=torch.float32)
    W_val_tensor = torch.tensor(W_val, dtype=torch.float32)
    
    n_train = len(X_train)
    n_val = len(X_val)

    indices        = list(range(n_train))
    best_loss      = float("inf")
    best_weights   = None
    no_improve        = 0
    start_epoch       = 0
    bar_width         = 30
    training_complete = False  # True only on natural finish; False on Ctrl+C

    # --- Resume from checkpoint if one exists ---
    os.makedirs("models", exist_ok=True)
    if (not FROM_SCRATCH) and os.path.exists(CHECKPOINT_PATH):
        print(f"[Checkpoint] Resuming from {CHECKPOINT_PATH} ...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

        # Backward-compat: old checkpoints were saved from a compiled model and
        # have "_orig_mod." prefixed keys. Strip the prefix so the weights load
        # into the raw model cleanly. After this run, checkpoints are saved from
        # raw_model so the strip will be a no-op on every future resume.
        def _clean(sd):
            return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                    for k, v in sd.items()}

        try:
            model.load_state_dict(_clean(ckpt["model"]))
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch  = ckpt["epoch"] + 1
            best_loss    = ckpt["best_loss"]
            no_improve   = ckpt["no_improve"]
            bw = ckpt.get("best_weights")
            best_weights = _clean(bw) if bw is not None else None
            print(f"[Checkpoint] Resuming at epoch {start_epoch}, best loss {best_loss:.4f}")
        except (RuntimeError, ValueError):
            print("[Checkpoint] Architecture changed — starting fresh training")
    elif FROM_SCRATCH and os.path.exists(CHECKPOINT_PATH):
        print("[Checkpoint] TRAIN_FROM_SCRATCH=1 -> ignoring existing checkpoint")

    # Write SESSION START (also detects previous kill and notes it in the log)
    _log_session_start(start_epoch)

    # Keep a reference to the raw (uncompiled) model for ONNX export later.
    # torch.onnx.export uses jit.trace which conflicts with dynamo-compiled models.
    # We use raw_model only at export time — training always uses the compiled one.
    raw_model = model
    ema_model = SmartPenDecoder(feat_mean, feat_std).to(DEVICE)
    ema_model.load_state_dict(raw_model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False

    # torch.compile() AFTER checkpoint load — compiling changes key names
    # (_orig_mod. prefix) so the checkpoint must be loaded into the raw model first.
    # XPU + torch.compile can fail in some container toolchains (Triton/headers).
    # Keep compile on by default for CUDA/CPU, and opt-in on XPU via TRAIN_FORCE_COMPILE=1.
    compile_enabled = (FORCE_COMPILE or DEVICE.type != "xpu") and (not DISABLE_COMPILE)
    if compile_enabled:
        try:
            # dynamic=True prevents recompilation stalls when pack_padded_sequence is used
            model = torch.compile(model, dynamic=True, mode="max-autotune")
            print("torch.compile: enabled (dynamic=True, mode=max-autotune)")
        except Exception:
            print("torch.compile: not available, running normally")
    else:
        print("torch.compile: disabled on XPU (set TRAIN_FORCE_COMPILE=1 to override)")

    # Print which acceleration features are active
    accel = []
    accel.append("torch.compile[max-autotune]")
    accel.append("oneDNN-fusion")
    accel.append("bf16" if USE_BF16 else "fp32")
    accel.append("fused-adamw" if DEVICE.type == "cuda" else "foreach-adam")
    accel.append("global-memory-slice")
    print(f"Acceleration: {' | '.join(accel)}")

    model.train()
    try:
        for epoch in range(start_epoch, EPOCHS):
            # ── Epoch Time Tracking ─────────────────────────────────
            epoch_start_time = time.time()
            
            # --- Set LR manually (warm-up + cosine) ---
            lr = get_lr(epoch, EPOCHS, WARMUP_EPOCHS, BASE_LR)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Slice batches directly from pre-padded memory
            indices = _build_balanced_indices(Y_train)
            if not indices:
                print("[Error] Balanced sampler produced no indices.")
                return
            n_bal = len(indices)
            n_batches = (n_bal + batch_size - 1) // batch_size
            batch_count = 0
            total_loss  = 0.0

            for start_idx in range(0, n_bal, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                
                # Slicing from global tensors
                x_pad = X_pad_global[batch_idx]
                lengths = lengths_global[batch_idx]
                target_lengths = tgt_lengths_global[batch_idx]
                if DEVICE.type in ("cuda", "xpu"):
                    x_pad = x_pad.to(DEVICE, non_blocking=True)
                    lengths = lengths.to(DEVICE, non_blocking=True)
                    target_lengths = target_lengths.to(DEVICE, non_blocking=True)
                
                # Gather targets dynamically from the flat array using start offsets
                ys = []
                for idx in batch_idx:
                    s = tgt_starts_global[idx]
                    e = s + tgt_lengths_global[idx]
                    ys.append(targets_global[s:e])
                targets = torch.cat(ys)
                if DEVICE.type in ("cuda", "xpu"):
                    targets = targets.to(DEVICE, non_blocking=True)
                
                # Get sample weights for this batch
                batch_weights = W_train_tensor[batch_idx].to(DEVICE)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=DEVICE.type,
                                    dtype=AMP_DTYPE,
                                    enabled=USE_AMP or USE_BF16):
                    logits = model(x_pad, lengths)           # (B, T_max, C)

                logits_ctc = logits.float().permute(1, 0, 2) # (T_out, B, C) fp32

                # Conv stride-2 halves T; CTC needs the actual output lengths
                ctc_lengths = SmartPenDecoder.conv_out_len(lengths).clamp(min=1)

                # CTCLoss(reduction='none') returns shape (B,) — one loss per sample.
                # Multiply elementwise by per-sample weights (B,) then reduce to scalar.
                if _ctc_native:
                    loss = ctc_loss(
                        logits_ctc.log_softmax(2),
                        targets,
                        ctc_lengths.to(DEVICE),
                        target_lengths,
                    )
                    # Apply sample weights: real data (seed) gets 3x weight, then mean
                    loss = (loss * batch_weights).mean()
                else:
                    loss = ctc_loss(
                        logits_ctc.cpu().log_softmax(2),
                        targets.cpu(),
                        ctc_lengths.cpu(),
                        target_lengths.cpu(),
                    )
                    # Apply sample weights: real data (seed) gets 3x weight, then mean
                    loss = (loss * batch_weights.cpu()).mean()

                if DEVICE.type == "cuda":
                    scaler.scale(loss).backward()
                    if not run_fast:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if not run_fast:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                # EMA weights often improve final validation and export accuracy.
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), raw_model.parameters()):
                        ema_p.mul_(EMA_DECAY).add_(p, alpha=(1.0 - EMA_DECAY))

                total_loss  += loss.item() * x_pad.size(0)
                batch_count += 1

                if batch_count % PROGRESS_EVERY == 0 or batch_count == n_batches:
                    filled = int(bar_width * batch_count / n_batches)
                    bar    = "#" * filled + "-" * (bar_width - filled)
                    
                    # ── Calculate Time Estimates ────────────────────────
                    current_time = time.time()
                    epoch_elapsed = current_time - epoch_start_time
                    training_elapsed = current_time - training_start_time
                    
                    # Estimate time remaining for current epoch
                    if batch_count > 0:
                        epoch_eta = epoch_elapsed * (n_batches / batch_count - 1)
                        epoch_eta_str = f"{epoch_eta/60:.1f}m" if epoch_eta > 60 else f"{epoch_eta:.0f}s"
                    else:
                        epoch_eta_str = "--s"
                    
                    # Estimate time remaining for total training
                    epochs_completed = epoch - start_epoch
                    epochs_remaining = EPOCHS - epoch - 1
                    if epochs_completed > 0:
                        avg_epoch_time = training_elapsed / epochs_completed
                        total_eta = avg_epoch_time * epochs_remaining
                        total_eta_str = f"{total_eta/3600:.1f}h" if total_eta > 3600 else f"{total_eta/60:.1f}m"
                    else:
                        total_eta_str = "--m"
                    
                    sys.stdout.write(
                        f"\rEpoch {epoch+1:>4}/{EPOCHS}  [{bar}]  "
                        f"batch {batch_count}/{n_batches}  "
                        f"lr={lr:.2e}  "
                        f"epoch_eta={epoch_eta_str}  "
                        f"total_eta={total_eta_str}"
                    )
                    sys.stdout.flush()

            avg_loss = total_loss / n_bal
            
            # ── Epoch Completion Time ───────────────────────────────
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_start_times.append(epoch_duration)
            
            # Print epoch completion with time
            epochs_completed = epoch - start_epoch + 1
            training_elapsed = epoch_end_time - training_start_time
            avg_epoch_time = training_elapsed / epochs_completed
            
            if epochs_completed > 1:
                remaining_epochs = EPOCHS - epoch - 1
                est_remaining_time = avg_epoch_time * remaining_epochs
                est_remaining_str = f"{est_remaining_time/3600:.1f}h" if est_remaining_time > 3600 else f"{est_remaining_time/60:.1f}m"
            else:
                est_remaining_str = "Calculating..."
            
            print(f"\nEpoch {epoch+1:>4}/{EPOCHS} completed in {epoch_duration/60:.1f}m ({epoch_duration:.0f}s)")
            print(f"Training time: {training_elapsed/60:.1f}m total, avg: {avg_epoch_time/60:.1f}m/epoch, ETA: {est_remaining_str}")

            # Validation pass on untouched holdout split (can be throttled in fast mode).
            compute_val_metrics = ((epoch + 1) % VAL_METRICS_EVERY == 0) or (epoch + 1 == EPOCHS)
            if ((epoch + 1) % val_every == 0) or (epoch == start_epoch) or (epoch + 1 == EPOCHS):
                model.eval()
                val_total_loss = 0.0
                val_char_edits = 0
                val_char_total = 0
                val_word_hits = 0
                val_word_total = 0
                with torch.inference_mode():
                    for start_idx in range(0, n_val, batch_size):
                        batch_idx = list(range(start_idx, min(start_idx + batch_size, n_val)))
                        x_pad = X_pad_val[batch_idx]
                        lengths = lengths_val[batch_idx]
                        target_lengths = tgt_lengths_val[batch_idx]
                        if DEVICE.type in ("cuda", "xpu"):
                            x_pad = x_pad.to(DEVICE, non_blocking=True)
                            lengths = lengths.to(DEVICE, non_blocking=True)
                            target_lengths = target_lengths.to(DEVICE, non_blocking=True)

                        ys = []
                        for idx in batch_idx:
                            s = tgt_starts_val[idx]
                            e = s + tgt_lengths_val[idx]
                            ys.append(targets_val[s:e])
                        targets = torch.cat(ys)
                        if DEVICE.type in ("cuda", "xpu"):
                            targets = targets.to(DEVICE, non_blocking=True)
                        
                        # Get sample weights for this batch
                        batch_weights = W_val_tensor[batch_idx].to(DEVICE)

                        logits = ema_model(x_pad, lengths)
                        logits_ctc = logits.float().permute(1, 0, 2)
                        ctc_lengths = SmartPenDecoder.conv_out_len(lengths).clamp(min=1)

                        # CTCLoss(reduction='none') returns shape (B,).
                        # Multiply by weights (B,) and sum for accumulation across batches.
                        # We divide by n_val at the end to get the weighted mean.
                        if _ctc_native:
                            vloss = ctc_loss(
                                logits_ctc.log_softmax(2),
                                targets,
                                ctc_lengths.to(DEVICE),
                                target_lengths,
                            )
                            val_total_loss += (vloss * batch_weights).sum().item()
                        else:
                            vloss = ctc_loss(
                                logits_ctc.cpu().log_softmax(2),
                                targets.cpu(),
                                ctc_lengths.cpu(),
                                target_lengths.cpu(),
                            )
                            val_total_loss += (vloss * batch_weights.cpu()).sum().item()

                        if compute_val_metrics:
                            pred_seqs = _ctc_greedy_decode_ids(logits, ctc_lengths)
                            gt_seqs = [y.tolist() for y in ys]
                            for pseq, gseq in zip(pred_seqs, gt_seqs):
                                val_char_edits += _levenshtein(pseq, gseq)
                                val_char_total += max(1, len(gseq))
                                val_word_hits += int(pseq == gseq)
                                val_word_total += 1
                model.train()
                val_loss = val_total_loss / n_val
            else:
                val_loss = best_loss
            if val_loss < best_loss:
                best_loss    = val_loss
                # Save from raw_model — keys are always clean (no _orig_mod. prefix)
                # even though training runs through the compiled wrapper.
                best_weights = {k: v.cpu().clone() for k, v in ema_model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1

            # Move to new line and print the epoch summary
            stop_marker = f"  [no improvement {no_improve}/{PATIENCE}]" if no_improve > 0 else ""
            line = (
                f"Epoch {epoch+1:>4}/{EPOCHS}  "
                f"train={avg_loss:.4f}  val={val_loss:.4f}  best={best_loss:.4f}  lr={lr:.2e}"
                + stop_marker
            )
            if ((epoch + 1) % val_every == 0) or (epoch == start_epoch) or (epoch + 1 == EPOCHS):
                if compute_val_metrics and val_word_total > 0:
                    cer = val_char_edits / max(1, val_char_total)
                    wer = 1.0 - (val_word_hits / val_word_total)
                    line += f"  cer={cer:.3f}  wer={wer:.3f}"
            # Pad to 80 chars so any leftover progress-bar text is overwritten
            print(f"\r{line:<80}")

            # Append one row to the training log (creates file + header if new).
            log_exists = os.path.exists(LOG_PATH)
            with open(LOG_PATH, "a") as log_f:
                if not log_exists:
                    log_f.write("epoch,train_loss,val_loss,best_val_loss,lr,no_improve,timestamp\n")
                log_f.write(
                    f"{epoch+1},{avg_loss:.6f},{val_loss:.6f},{best_loss:.6f},"
                    f"{lr:.2e},{no_improve},"
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )

            # Save checkpoint every N epochs so training can resume after a crash.
            # Always use raw_model.state_dict() — clean keys, no _orig_mod. prefix.
            if (epoch + 1) % checkpoint_every == 0:
                torch.save({
                    "epoch":        epoch,
                    "model":        raw_model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "best_loss":    best_loss,
                    "no_improve":   no_improve,
                    "best_weights": best_weights,
                }, CHECKPOINT_PATH)

            # Early stopping: plateau detected
            if no_improve >= PATIENCE:
                print(f"\n[Early stop] No improvement for {PATIENCE} epochs. Best loss: {best_loss:.4f}")
                training_complete = True
                _log_session_end("completed normally — early stop", best_loss, epoch + 1)
                break

        else:
            # for-loop completed all EPOCHS without a break — not caught by early stop
            training_complete = True
            _log_session_end("completed normally — all epochs done", best_loss, EPOCHS)

    except KeyboardInterrupt:
        print(f"\n\n[Interrupted] ", end="")
        if best_weights is None:
            print("No epoch completed — nothing to export. Re-run and let at least 1 epoch finish.")
            _log_session_end("ctrl+c (no weights)", float("inf"), start_epoch)
            return
        _log_session_end(f"ctrl+c", best_loss, epoch + 1)
        print(f"Exporting model at best loss {best_loss:.4f} (checkpoint kept — re-run to continue training) ...")

    finally:
        pass

    # Restore the best weights seen during training.
    # best_weights is always saved from raw_model so keys are clean — load directly.
    if best_weights is not None:
        raw_model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})

    # ---------------------------------------------------------
    # 6. ONNX EXPORT FOR C++
    # ---------------------------------------------------------
    import warnings
    import onnx

    warnings.filterwarnings("ignore")
    os.makedirs("models", exist_ok=True)
    onnx_path = "models/pen_model.onnx"

    # Use raw_model (not compiled model) — torch.onnx.export uses jit.trace which
    # crashes on dynamo-optimized (torch.compile'd) models.
    raw_model = raw_model.to("cpu")
    raw_model.eval()
    dummy_input = torch.randn(1, 150, INPUT_FEATURES, device="cpu")

    # Use the legacy TorchScript-based exporter (dynamo=False) with dynamic_axes.
    # The newer dynamo exporter conflicts when the dummy input has a fixed sequence
    # length (150) but the dimension is declared dynamic via Dim objects.
    torch.onnx.export(
        raw_model,
        (dummy_input,),
        onnx_path,
        dynamo=False,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_stroke"],
        output_names=["predicted_logits"],
        dynamic_axes={
            "input_stroke":     {1: "input_seq_len"},
            "predicted_logits": {1: "output_seq_len"},
        },
    )

    # Downgrade IR version so the C++ system ONNX Runtime library can read it.
    onnx_model = onnx.load(onnx_path)
    onnx_model.ir_version = 9
    onnx.save(onnx_model, onnx_path)

    print(f"\n[SUCCESS] AI Brain exported to: {onnx_path} (IR Version 9)")
    print(f"Final training loss: {best_loss:.4f}")
    print("The C++ decoder is now ready to receive this brain.")
    
    # ── Final Training Time Summary ───────────────────────────────
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    print(f"\n{'='*60}")
    print(f"TRAINING TIME SUMMARY")
    print(f"{'='*60}")
    print(f"Total training time: {total_training_time/3600:.2f} hours ({total_training_time/60:.1f} minutes)")
    print(f"Epochs completed: {epoch - start_epoch + 1}")
    
    if epoch_start_times:
        avg_epoch_time = sum(epoch_start_times) / len(epoch_start_times)
        fastest_epoch = min(epoch_start_times)
        slowest_epoch = max(epoch_start_times)
        
        print(f"Average epoch time: {avg_epoch_time/60:.2f} minutes")
        print(f"Fastest epoch: {fastest_epoch/60:.2f} minutes")
        print(f"Slowest epoch: {slowest_epoch/60:.2f} minutes")
        print(f"Epoch time variance: {(slowest_epoch - fastest_epoch)/60:.2f} minutes")
    
    print(f"Training completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Only remove checkpoint on natural completion (early stop / all epochs done).
    # On Ctrl+C we keep it so the next run can resume and continue training.
    if training_complete and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print(f"[Checkpoint] Removed {CHECKPOINT_PATH} (training complete)")


if __name__ == "__main__":
    train_and_export()