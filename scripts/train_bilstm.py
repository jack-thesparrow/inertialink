# scripts / train_bilstm.py
import os
import glob
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. ALPHABET & HYPERPARAMETERS
# ---------------------------------------------------------
# Index 0 is the CTC blank token (single placeholder char '~', never printed).
# Real characters start at index 1.  Total classes = 64.
ALPHABET = "~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALPHABET)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(ALPHABET)}

INPUT_FEATURES = 3   # (x, y, accel_z)
HIDDEN_SIZE    = 256  # Doubled from 128 — more capacity for 12 words
NUM_LAYERS     = 3    # 3 stacked BiLSTM layers
NUM_CLASSES    = len(ALPHABET)  # 64

EPOCHS         = 500
BATCH_SIZE     = 16   # Mini-batch gradient descent
WARMUP_EPOCHS  = 20   # Linear LR warm-up before cosine decay
BASE_LR        = 3e-4


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

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features) — raw values from C++
        x = (x - self.input_mean) / (self.input_std + 1e-8)
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        # logits shape: (Batch, Seq_Len, Num_Classes)
        return logits


# ---------------------------------------------------------
# 3. DATA LOADING
# ---------------------------------------------------------
def load_dataset(data_dir="data"):
    """Reads all CSVs exported by the C++ Autonomous Collector"""
    sequences = []
    targets = []

    if not os.path.exists(data_dir):
        print(f"[Error] Directory '{data_dir}' not found! Run generate_synthetic_data.py first.")
        return sequences, targets

    for label_folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue

        csv_files = glob.glob(f"{folder_path}/*.csv")
        if not csv_files:
            continue

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)

            # Extract the 3 physical features (Skip timestamp)
            tensor_data = torch.tensor(
                df[["x", "y", "accel_z"]].values, dtype=torch.float32
            )

            # Convert the folder name (e.g., "hello") into target IDs
            target_ids = [
                CHAR_TO_IDX[char] for char in label_folder if char in CHAR_TO_IDX
            ]

            sequences.append(tensor_data)
            targets.append(torch.tensor(target_ids, dtype=torch.long))

    return sequences, targets


# ---------------------------------------------------------
# 4. LR SCHEDULE
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
    print("=== Smart Pen CTC Trainer (v2 — 256-unit BiLSTM × 3, cosine LR) ===")
    X_train, Y_train = load_dataset()

    if len(X_train) == 0:
        return

    n = len(X_train)
    print(f"Loaded {n} stroke samples.")

    # --- Compute normalization stats from the whole training set ---
    all_frames = torch.cat(X_train, dim=0)  # (N_total_frames, 3)
    feat_mean  = all_frames.mean(dim=0)     # (3,)
    feat_std   = all_frames.std(dim=0)      # (3,)
    print(f"Feature mean: {feat_mean.tolist()}")
    print(f"Feature std:  {feat_std.tolist()}")

    model     = SmartPenDecoder(feat_mean, feat_std)
    optimizer = optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-5)
    ctc_loss  = nn.CTCLoss(blank=0, zero_infinity=True)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(f"Training for {EPOCHS} epochs, batch size {BATCH_SIZE}.")
    print("-" * 60)

    indices   = list(range(n))
    best_loss = float("inf")
    n_batches = math.ceil(n / BATCH_SIZE)
    bar_width = 30

    import sys, time

    model.train()
    for epoch in range(EPOCHS):
        # --- Set LR manually (warm-up + cosine) ---
        lr = get_lr(epoch, EPOCHS, WARMUP_EPOCHS, BASE_LR)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        total_loss  = 0.0
        batch_count = 0
        random.shuffle(indices)

        # --- Mini-batch loop with in-line progress bar ---
        for batch_start in range(0, n, BATCH_SIZE):
            batch_idx = indices[batch_start : batch_start + BATCH_SIZE]

            optimizer.zero_grad()
            batch_loss = 0.0

            for i in batch_idx:
                x, y = X_train[i], Y_train[i]

                # Add Batch Dimension: (Seq_Len, 3) -> (1, Seq_Len, 3)
                x_batched = x.unsqueeze(0)
                logits = model(x_batched)

                # CTC Loss expects: (Seq_Len, Batch, Num_Classes)
                logits_ctc = logits.transpose(0, 1)

                input_lengths  = torch.tensor([logits_ctc.size(0)], dtype=torch.long)
                target_lengths = torch.tensor([y.size(0)],          dtype=torch.long)

                loss = ctc_loss(
                    logits_ctc.log_softmax(2), y, input_lengths, target_lengths
                )
                batch_loss += loss

            # Average the loss over the mini-batch then back-prop once
            (batch_loss / len(batch_idx)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss  += batch_loss.item()
            batch_count += 1

            # Overwrite same line with a mini progress bar
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
            best_loss = avg_loss

        # Move to new line and print the epoch summary
        print(
            f"\rEpoch {epoch+1:>4}/{EPOCHS}  "
            f"loss={avg_loss:.4f}  best={best_loss:.4f}  lr={lr:.2e}"
            + " " * 10  # clear any leftover bar characters
        )

    # ---------------------------------------------------------
    # 6. ONNX EXPORT FOR C++
    # ---------------------------------------------------------
    import warnings
    import onnx

    warnings.filterwarnings("ignore")
    os.makedirs("models", exist_ok=True)
    onnx_path = "models/pen_model.onnx"

    model.eval()
    dummy_input = torch.randn(1, 150, INPUT_FEATURES)

    # Use the legacy TorchScript-based exporter (dynamo=False) with dynamic_axes.
    # The newer dynamo exporter conflicts when the dummy input has a fixed sequence
    # length (150) but the dimension is declared dynamic via Dim objects.
    torch.onnx.export(
        model,
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


if __name__ == "__main__":
    train_and_export()
