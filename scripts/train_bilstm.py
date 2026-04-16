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
import queue
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# 0. DEVICE SELECTION
#    Priority: Intel Arc native XPU → IPEX XPU → NVIDIA CUDA → CPU
#
#    PyTorch 2.4+ has built-in XPU support for Intel Arc — no IPEX needed.
#    IPEX is tried as a fallback for older PyTorch versions.
#    Install Intel drivers: https://dgpu-docs.intel.com/driver/installation.html
# ---------------------------------------------------------
def get_device() -> torch.device:
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
# Fuse adjacent oneDNN ops (free ~5-10% on Intel CPUs)
torch.jit.enable_onednn_fusion(True)

# Mixed precision (AMP): automatically uses fp16 for CUDA, bf16 for XPU/CPU.
# Falls back to float32 silently on standard CPUs.
USE_AMP = DEVICE.type in ["cuda", "xpu"]
USE_BF16 = getattr(torch.backends, "mkldnn", None) is not None
AMP_DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.bfloat16

# ---------------------------------------------------------
# 1. ALPHABET & HYPERPARAMETERS
# ---------------------------------------------------------
# Index 0 is the CTC blank token (single placeholder char '~', never printed).
# ALPHABET for digits-only prototype (0 is CTC blank)
ALPHABET = "~ 0123456789"
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALPHABET)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(ALPHABET)}

INPUT_FEATURES = 6   # (ax, ay, az, gx, gy, gz)
HIDDEN_SIZE    = 64  # Smaller vocabulary = can run an even smaller network
NUM_LAYERS     = 2    
NUM_CLASSES    = len(ALPHABET)  # 11
CONV_CHANNELS  = 64   # Conv1d frontend output channels (feeds into LSTM)

EPOCHS             = 150   # Conv+BiLSTM converges faster; halved from 300
BATCH_SIZE         = 128   # 2x larger batches = fewer steps/epoch
WARMUP_EPOCHS      = 5     # Short warmup with higher base LR
BASE_LR            = 5e-4  # Aggressive LR — cosine decays it safely
PATIENCE           = 60    # Tighter early stop; conv frontend converges faster
CHECKPOINT_EVERY   = 10    # Reduce checkpoint I/O overhead (was 2)
CHECKPOINT_PATH    = "models/checkpoint.pt"
LOG_PATH           = "models/training_log.csv"


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


DATASET_CACHE = "data/dataset_cache.pt"

# Directories to scan for training data.  Both are merged into one dataset.
#   data/       — augmented synthetic data from augment_seed_data.py
#   data/seed/  — real hardware collected seed data from data_collector
DATA_DIRS = ["data", "data/seed"]

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
    """
    if os.path.exists(DATASET_CACHE) and not _any_csv_newer_than(DATA_DIRS, DATASET_CACHE):
        print(f"[Dataset] Loading cache from {DATASET_CACHE} ...")
        cached = torch.load(DATASET_CACHE, map_location="cpu")
        return cached["X"], cached["Y"]

    print("[Dataset] Building dataset from CSVs ...")
    sequences, targets = [], []
    source_counts: dict[str, int] = {}

    for data_dir in DATA_DIRS:
        if not os.path.exists(data_dir):
            continue
        for label_folder in sorted(os.listdir(data_dir)):
            if data_dir == "data" and label_folder == "seed":
                continue
            folder_path = os.path.join(data_dir, label_folder)
            if not os.path.isdir(folder_path):
                continue
            csv_files = glob.glob(f"{folder_path}/*.csv")
            if not csv_files:
                continue
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                sequences.append(
                    torch.tensor(df[["ax", "ay", "az", "gx", "gy", "gz"]].values, dtype=torch.float32)
                )
                targets.append(
                    torch.tensor(
                        [CHAR_TO_IDX[c] for c in label_folder if c in CHAR_TO_IDX],
                        dtype=torch.long,
                    )
                )
            count = len(csv_files)
            key = f"{data_dir}/{label_folder}"
            source_counts[key] = count

    if not sequences:
        print("[Error] No training data found!  Run generate_synthetic_data.py or data_collector first.")
        return [], []

    # Print source breakdown
    for src, cnt in sorted(source_counts.items()):
        tag = "seed" if "seed" in src else "aug "
        print(f"  [{tag}] {src}: {cnt} samples")

    os.makedirs(os.path.dirname(DATASET_CACHE), exist_ok=True)
    torch.save({"X": sequences, "Y": targets}, DATASET_CACHE)
    print(f"[Dataset] Cached {len(sequences)} total samples to {DATASET_CACHE}")
    return sequences, targets


# ---------------------------------------------------------
# 4. GLOBAL BATCH PRE-BUILDER
# ---------------------------------------------------------
def prep_global_tensors(X_train, Y_train, device):
    """Pad the entire dataset to a single fixed tensor up-front.
    Eliminates PyTorch padding overhead during training and locks the
    batch sequence length so torch.compile only compiles exactly once.
    """
    print("[Dataset] Padding entire dataset globally...")
    xs = [x.to(device) for x in X_train]
    X_pad = pad_sequence(xs, batch_first=True)  # (N, T_max, 6)
    
    lengths_t = torch.tensor([x.size(0) for x in X_train], dtype=torch.long)
    targets_flat = torch.cat(Y_train)
    tgt_lengths_t = torch.tensor([y.size(0) for y in Y_train], dtype=torch.long)
    
    # Pre-calculate slice indices for the flattened targets
    target_start_idx = torch.zeros(len(Y_train) + 1, dtype=torch.long)
    target_start_idx[1:] = tgt_lengths_t.cumsum(dim=0)
    
    return X_pad, lengths_t, targets_flat, tgt_lengths_t, target_start_idx


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
            f.write("epoch,loss,best_loss,lr,no_improve,timestamp\n")
        action = f"resuming from epoch {start_epoch}" if start_epoch > 0 else "fresh start"
        f.write(f"--- SESSION START | {_now()} | {action} ---\n")


def _log_session_end(reason: str, best_loss: float, epoch: int) -> None:
    """Write a SESSION END marker with the reason training stopped."""
    with open(LOG_PATH, "a") as f:
        f.write(
            f"--- SESSION END | {reason} | "
            f"stopped at epoch {epoch} | best_loss={best_loss:.6f} | {_now()} ---\n"
        )


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
    X_train, Y_train = load_dataset()

    if len(X_train) == 0:
        return

    n = len(X_train)
    print(f"Loaded {n} stroke samples.")

    # --- Compute normalization stats from the whole training set (on CPU) ---
    all_frames = torch.cat(X_train, dim=0)  # (N_total_frames, 3)
    feat_mean  = all_frames.mean(dim=0)     # (3,)
    feat_std   = all_frames.std(dim=0)      # (3,)
    print(f"Feature mean: {feat_mean.tolist()}")
    print(f"Feature std:  {feat_std.tolist()}")

    model     = SmartPenDecoder(feat_mean, feat_std).to(DEVICE)
    # foreach=True: Adam updates all parameter tensors in a single fused kernel
    # call instead of a Python loop — ~10-20% faster optimizer step.
    optimizer = optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-5,
                           foreach=True)
    ctc_loss  = nn.CTCLoss(blank=0, zero_infinity=True)

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
    print(f"Training for {EPOCHS} epochs, batch size {BATCH_SIZE}.")
    print("-" * 60)

    # Pad the entire dataset upfront globally. 
    # This locks sequence lengths to T_max and speeds up torch.compile massively.
    X_pad_global, lengths_global, targets_global, tgt_lengths_global, tgt_starts_global = prep_global_tensors(X_train, Y_train, DEVICE)

    indices        = list(range(n))
    best_loss      = float("inf")
    best_weights   = None
    no_improve        = 0
    start_epoch       = 0
    bar_width         = 30
    training_complete = False  # True only on natural finish; False on Ctrl+C

    # --- Resume from checkpoint if one exists ---
    os.makedirs("models", exist_ok=True)
    if os.path.exists(CHECKPOINT_PATH):
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
        except RuntimeError:
            print("[Checkpoint] Architecture changed — starting fresh training")

    # Write SESSION START (also detects previous kill and notes it in the log)
    _log_session_start(start_epoch)

    # Keep a reference to the raw (uncompiled) model for ONNX export later.
    # torch.onnx.export uses jit.trace which conflicts with dynamo-compiled models.
    # We use raw_model only at export time — training always uses the compiled one.
    raw_model = model

    # torch.compile() AFTER checkpoint load — compiling changes key names
    # (_orig_mod. prefix) so the checkpoint must be loaded into the raw model first.
    try:
        # dynamic=True prevents recompilation stalls when pack_padded_sequence is used
        model = torch.compile(model, dynamic=True)
        print("torch.compile: enabled (dynamic=True)")
    except Exception:
        print("torch.compile: not available, running normally")

    # Print which acceleration features are active
    accel = []
    accel.append("torch.compile[max-autotune]")
    accel.append("oneDNN-fusion")
    accel.append("bf16" if USE_BF16 else "fp32")
    accel.append("foreach-adam")
    accel.append("global-memory-slice")
    print(f"Acceleration: {' | '.join(accel)}")

    model.train()
    try:
        for epoch in range(start_epoch, EPOCHS):
            # --- Set LR manually (warm-up + cosine) ---
            lr = get_lr(epoch, EPOCHS, WARMUP_EPOCHS, BASE_LR)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Slice batches directly from pre-padded memory
            random.shuffle(indices)
            n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
            batch_count = 0
            total_loss  = 0.0

            for start_idx in range(0, n, BATCH_SIZE):
                batch_idx = indices[start_idx : start_idx + BATCH_SIZE]
                
                # Slicing from global tensors
                x_pad = X_pad_global[batch_idx]
                lengths = lengths_global[batch_idx]
                target_lengths = tgt_lengths_global[batch_idx]
                
                # Gather targets dynamically from the flat array using start offsets
                ys = []
                for idx in batch_idx:
                    s = tgt_starts_global[idx]
                    e = s + tgt_lengths_global[idx]
                    ys.append(targets_global[s:e])
                targets = torch.cat(ys)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=DEVICE.type,
                                    dtype=AMP_DTYPE,
                                    enabled=USE_AMP or USE_BF16):
                    logits = model(x_pad, lengths)           # (B, T_max, C)

                logits_ctc = logits.float().permute(1, 0, 2) # (T_out, B, C) fp32

                # Conv stride-2 halves T; CTC needs the actual output lengths
                ctc_lengths = SmartPenDecoder.conv_out_len(lengths).clamp(min=1)

                if _ctc_native:
                    loss = ctc_loss(
                        logits_ctc.log_softmax(2),
                        targets.to(DEVICE),
                        ctc_lengths.to(DEVICE),
                        target_lengths.to(DEVICE),
                    )
                else:
                    loss = ctc_loss(
                        logits_ctc.cpu().log_softmax(2),
                        targets,
                        ctc_lengths,
                        target_lengths,
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss  += loss.item() * x_pad.size(0)
                batch_count += 1

                if batch_count % 10 == 0 or batch_count == n_batches:
                    filled = int(bar_width * batch_count / n_batches)
                    bar    = "#" * filled + "-" * (bar_width - filled)
                    sys.stdout.write(
                        f"\rEpoch {epoch+1:>4}/{EPOCHS}  [{bar}]  "
                        f"batch {batch_count}/{n_batches}  "
                        f"lr={lr:.2e}"
                    )
                    sys.stdout.flush()

            avg_loss = total_loss / n
            if avg_loss < best_loss:
                best_loss    = avg_loss
                # Save from raw_model — keys are always clean (no _orig_mod. prefix)
                # even though training runs through the compiled wrapper.
                best_weights = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1

            # Move to new line and print the epoch summary
            stop_marker = f"  [no improvement {no_improve}/{PATIENCE}]" if no_improve > 0 else ""
            line = (
                f"Epoch {epoch+1:>4}/{EPOCHS}  "
                f"loss={avg_loss:.4f}  best={best_loss:.4f}  lr={lr:.2e}"
                + stop_marker
            )
            # Pad to 80 chars so any leftover progress-bar text is overwritten
            print(f"\r{line:<80}")

            # Append one row to the training log (creates file + header if new).
            log_exists = os.path.exists(LOG_PATH)
            with open(LOG_PATH, "a") as log_f:
                if not log_exists:
                    log_f.write("epoch,loss,best_loss,lr,no_improve,timestamp\n")
                log_f.write(
                    f"{epoch+1},{avg_loss:.6f},{best_loss:.6f},"
                    f"{lr:.2e},{no_improve},"
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )

            # Save checkpoint every N epochs so training can resume after a crash.
            # Always use raw_model.state_dict() — clean keys, no _orig_mod. prefix.
            if (epoch + 1) % CHECKPOINT_EVERY == 0:
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
    raw_model.eval()
    dummy_input = torch.randn(1, 150, INPUT_FEATURES)

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

    # Only remove checkpoint on natural completion (early stop / all epochs done).
    # On Ctrl+C we keep it so the next run can resume and continue training.
    if training_complete and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print(f"[Checkpoint] Removed {CHECKPOINT_PATH} (training complete)")


if __name__ == "__main__":
    train_and_export()
