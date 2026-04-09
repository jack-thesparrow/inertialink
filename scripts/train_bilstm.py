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

# bfloat16 autocast: Intel 12th/13th gen CPUs run bf16 matmul natively.
# Forward pass uses bf16; CTCLoss stays float32 (cast back before loss).
# Falls back to float32 silently on CPUs that don't support it.
USE_BF16 = (DEVICE.type == "cpu") and torch.cpu.is_available() and \
           getattr(torch.backends, "mkldnn", None) is not None

# ---------------------------------------------------------
# 1. ALPHABET & HYPERPARAMETERS
# ---------------------------------------------------------
# Index 0 is the CTC blank token (single placeholder char '~', never printed).
# Real characters start at index 1.  Total classes = 64.
ALPHABET = "~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALPHABET)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(ALPHABET)}

INPUT_FEATURES = 3   # (x, y, accel_z)
HIDDEN_SIZE    = 128  # Sufficient for 12-word synthetic vocab; ~4x faster than 256
NUM_LAYERS     = 2    # 2 layers is enough; 3rd layer added marginal accuracy
NUM_CLASSES    = len(ALPHABET)  # 64

EPOCHS             = 300   # Cosine decays to ~1e-4 by epoch 200 (was too slow at 500)
BATCH_SIZE         = 64   # 128 OOMs on Arc 530M (shared LPDDR5); use 32 if still OOM
WARMUP_EPOCHS      = 20   # Linear LR warm-up before cosine decay
BASE_LR            = 3e-4
PATIENCE           = 150  # Must be > EPOCHS/2 so cosine has room to work
CHECKPOINT_EVERY   = 2    # Save a resume checkpoint every N epochs
CHECKPOINT_PATH    = "models/checkpoint.pt"


# ---------------------------------------------------------
# 2. NEURAL NETWORK ARCHITECTURE
# ---------------------------------------------------------
class SmartPenDecoder(nn.Module):
    """BiLSTM CTC decoder with baked-in z-score normalization.

    Normalization stats are registered as buffers so they are part of the
    ONNX export — the C++ decoder sends raw mm/accel values and gets
    logits back without needing any preprocessing code on its side.
    """

    def __init__(self, input_mean: torch.Tensor, input_std: torch.Tensor):
        super(SmartPenDecoder, self).__init__()
        # Buffers travel with the model into ONNX; they are NOT trainable.
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std",  input_std)

        self.lstm = nn.LSTM(
            input_size=INPUT_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        # Bi-LSTM outputs 2x hidden size
        self.fc = nn.Linear(HIDDEN_SIZE * 2, NUM_CLASSES)

    def forward(self, x, lengths=None):
        # x shape: (Batch, Seq_Len, Features) — raw values from C++
        x = (x - self.input_mean) / (self.input_std + 1e-8)
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        # logits shape: (Batch, Seq_Len, Num_Classes)
        return logits


DATASET_CACHE = "data/dataset_cache.pt"

# ---------------------------------------------------------
# 3. DATA LOADING
# ---------------------------------------------------------
def load_dataset(data_dir="data"):
    """Load training tensors from cache or build from CSVs.

    First run: reads all 2400 CSVs, converts to tensors, saves a single
    dataset_cache.pt.  Every subsequent run loads that one file — startup
    drops from ~10 s to < 1 s.

    Delete data/dataset_cache.pt whenever you regenerate synthetic data so
    the cache is rebuilt from the new CSVs.
    """
    if os.path.exists(DATASET_CACHE):
        print(f"[Dataset] Loading cache from {DATASET_CACHE} ...")
        cached = torch.load(DATASET_CACHE, map_location="cpu")
        return cached["X"], cached["Y"]

    print("[Dataset] Cache not found — reading CSVs (first run only) ...")
    if not os.path.exists(data_dir):
        print(f"[Error] '{data_dir}' not found! Run generate_synthetic_data.py first.")
        return [], []

    sequences, targets = [], []
    for label_folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue
        csv_files = glob.glob(f"{folder_path}/*.csv")
        if not csv_files:
            continue
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            sequences.append(
                torch.tensor(df[["x", "y", "accel_z"]].values, dtype=torch.float32)
            )
            targets.append(
                torch.tensor(
                    [CHAR_TO_IDX[c] for c in label_folder if c in CHAR_TO_IDX],
                    dtype=torch.long,
                )
            )

    torch.save({"X": sequences, "Y": targets}, DATASET_CACHE)
    print(f"[Dataset] Cached {len(sequences)} samples to {DATASET_CACHE}")
    return sequences, targets


# ---------------------------------------------------------
# 4. BATCH PRE-BUILDER
# ---------------------------------------------------------
def build_batches(X_train, Y_train, indices, batch_size):
    """Pre-assemble all padded batches for one epoch.

    Runs once in Python before each epoch so there is zero Python
    overhead *between* GPU kernel launches during the actual training
    loop — the GPU gets a new pre-built tensor immediately after each
    optimizer step.

    Mini-bucket sorting: within each window of 4×batch_size samples we
    sort by sequence length.  This keeps similar-length sequences together
    so each padded batch is only as long as its longest member rather than
    the global maximum — typically cuts padding by ~25 %, reducing wasted
    LSTM compute without removing the per-epoch shuffle randomness.
    """
    bucket = batch_size * 4
    sorted_indices: list[int] = []
    for start in range(0, len(indices), bucket):
        chunk = indices[start : start + bucket]
        chunk.sort(key=lambda i: X_train[i].size(0))
        sorted_indices.extend(chunk)

    batches = []
    for start in range(0, len(sorted_indices), batch_size):
        idx = sorted_indices[start : start + batch_size]
        xs  = [X_train[i] for i in idx]
        ys  = [Y_train[i] for i in idx]
        lengths        = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
        x_pad          = pad_sequence(xs, batch_first=True)        # on DEVICE already
        targets        = torch.cat(ys)                              # CPU
        target_lengths = torch.tensor([y.size(0) for y in ys], dtype=torch.long)
        batches.append((x_pad, lengths, targets, target_lengths))
    return batches


def prefetch_batches_async(X_train, Y_train, indices, batch_size, result):
    """Build next epoch's batches in a background thread.

    Runs concurrently while the current epoch trains so there is zero
    idle time between epochs waiting for batch assembly.
    result[0] is set to the built batch list when done.
    """
    shuffled = indices[:]
    random.shuffle(shuffled)
    result[0] = (shuffled, build_batches(X_train, Y_train, shuffled, batch_size))


# ---------------------------------------------------------
# 5. LR SCHEDULE
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

    # Pre-move all training tensors to the device once at startup.
    # This eliminates per-batch CPU→GPU transfers that stall the GPU.
    X_train = [x.to(DEVICE) for x in X_train]
    # Targets stay on CPU — CTCLoss backward requires CPU target indices.
    # We only transfer the padded input and lengths to the device.

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

        model.load_state_dict(_clean(ckpt["model"]))
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt["epoch"] + 1
        best_loss    = ckpt["best_loss"]
        no_improve   = ckpt["no_improve"]
        bw = ckpt.get("best_weights")
        best_weights = _clean(bw) if bw is not None else None
        print(f"[Checkpoint] Resuming at epoch {start_epoch}, best loss {best_loss:.4f}")

    # Keep a reference to the raw (uncompiled) model for ONNX export later.
    # torch.onnx.export uses jit.trace which conflicts with dynamo-compiled models.
    # We use raw_model only at export time — training always uses the compiled one.
    raw_model = model

    # torch.compile() AFTER checkpoint load — compiling changes key names
    # (_orig_mod. prefix) so the checkpoint must be loaded into the raw model first.
    try:
        # max-autotune: profiles multiple kernel implementations and keeps the
        # fastest one — best for repeated fixed-shape workloads (our batches).
        # First epoch is slower while it benchmarks; every subsequent epoch wins.
        model = torch.compile(model, mode="max-autotune")
        print("torch.compile: enabled (max-autotune, faster after 1st epoch)")
    except Exception:
        print("torch.compile: not available, running normally")

    # Print which acceleration features are active
    accel = []
    accel.append("torch.compile[max-autotune]")
    accel.append("oneDNN-fusion")
    accel.append("bf16" if USE_BF16 else "fp32")
    accel.append("foreach-adam")
    accel.append("async-prefetch")
    accel.append("bucket-batching")
    print(f"Acceleration: {' | '.join(accel)}")

    # Kick off the first prefetch before epoch 1 starts
    _prefetch_result = [None]
    _prefetch_thread = threading.Thread(
        target=prefetch_batches_async,
        args=(X_train, Y_train, indices, BATCH_SIZE, _prefetch_result),
        daemon=True,
    )
    _prefetch_thread.start()

    model.train()
    try:
        for epoch in range(start_epoch, EPOCHS):
            # --- Set LR manually (warm-up + cosine) ---
            lr = get_lr(epoch, EPOCHS, WARMUP_EPOCHS, BASE_LR)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Wait for prefetch thread to finish, grab its batches
            _prefetch_thread.join()
            indices, batches = _prefetch_result[0]
            n_batches   = len(batches)
            batch_count = 0
            total_loss  = 0.0

            # Start prefetching NEXT epoch's batches immediately in background
            _prefetch_result = [None]
            _prefetch_thread = threading.Thread(
                target=prefetch_batches_async,
                args=(X_train, Y_train, indices, BATCH_SIZE, _prefetch_result),
                daemon=True,
            )
            _prefetch_thread.start()

            for x_pad, lengths, targets, target_lengths in batches:
                # set_to_none=True: skips the memset that zero_grad() does —
                # None gradients are treated as zero by the backward pass but
                # avoid touching memory unnecessarily (small but free speedup).
                optimizer.zero_grad(set_to_none=True)

                # bf16 autocast: faster matmul on Intel 12th/13th gen CPUs.
                # CTCLoss needs float32 so we cast logits back before loss.
                with torch.autocast(device_type=DEVICE.type,
                                    dtype=torch.bfloat16,
                                    enabled=USE_BF16):
                    logits = model(x_pad, lengths)           # (B, T_max, C)

                logits_ctc = logits.float().permute(1, 0, 2) # (T_max, B, C) fp32

                if _ctc_native:
                    loss = ctc_loss(
                        logits_ctc.log_softmax(2),
                        targets.to(DEVICE),
                        lengths.to(DEVICE),
                        target_lengths.to(DEVICE),
                    )
                else:
                    loss = ctc_loss(
                        logits_ctc.cpu().log_softmax(2),
                        targets,
                        lengths,
                        target_lengths,
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss  += loss.item() * x_pad.size(0)
                batch_count += 1

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
                break

    except KeyboardInterrupt:
        print(f"\n\n[Interrupted] ", end="")
        if best_weights is None:
            print("No epoch completed — nothing to export. Re-run and let at least 1 epoch finish.")
            return
        print(f"Exporting model at best loss {best_loss:.4f} (checkpoint kept — re-run to continue training) ...")

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
            "input_stroke":     {1: "seq_len"},
            "predicted_logits": {1: "seq_len"},
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
