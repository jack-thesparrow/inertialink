# scripts / train_bilstm.py
import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. ALPHABET & HYPERPARAMETERS
# ---------------------------------------------------------
# Index 0 is strictly reserved for the CTC "Blank" token.
# We include lowercase, uppercase, numbers, and space.
ALPHABET = "<BLANK> abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALPHABET)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(ALPHABET)}

INPUT_FEATURES = 3  # (x, y, accel_z)
HIDDEN_SIZE = 128  # Brain capacity
NUM_LAYERS = 2  # Stacked LSTMs
NUM_CLASSES = len(ALPHABET)


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
            dropout=0.2,
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
        print(f"[Error] Directory '{data_dir}' not found! Collect data via C++ first.")
        return sequences, targets

    for label_folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue

        for csv_file in glob.glob(f"{folder_path}/*.csv"):
            df = pd.read_csv(csv_file)

            # Extract the 3 physical features (Skip timestamp)
            tensor_data = torch.tensor(
                df[["x", "y", "accel_z"]].values, dtype=torch.float32
            )

            # Convert the folder name (e.g., "Hello123") into target IDs
            target_ids = [
                CHAR_TO_IDX[char] for char in label_folder if char in CHAR_TO_IDX
            ]

            sequences.append(tensor_data)
            targets.append(torch.tensor(target_ids, dtype=torch.long))

    return sequences, targets


# ---------------------------------------------------------
# 4. TRAINING LOOP
# ---------------------------------------------------------
def train_and_export():
    print("=== Smart Pen CTC Trainer ===")
    X_train, Y_train = load_dataset()

    if len(X_train) == 0:
        return

    print(f"Loaded {len(X_train)} stroke samples.")

    # --- Compute normalization stats from the whole training set ---
    # Stack all frames to get a (TotalFrames, 3) tensor, then take mean/std
    # per feature.  These stats are baked into the model so C++ sends raw
    # values and the ONNX graph normalizes them internally.
    all_frames = torch.cat(X_train, dim=0)  # (N_total_frames, 3)
    feat_mean = all_frames.mean(dim=0)      # (3,)
    feat_std  = all_frames.std(dim=0)       # (3,)
    print(f"Feature mean: {feat_mean.tolist()}")
    print(f"Feature std:  {feat_std.tolist()}")

    model = SmartPenDecoder(feat_mean, feat_std)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # CTC Loss is the magic that allows continuous variable-length reading
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    model.train()
    epochs = 150  # More epochs — CTC needs loss well below 0.1 to decode reliably

    indices = list(range(len(X_train)))

    for epoch in range(epochs):
        total_loss = 0

        # Shuffle each epoch so the model learns the signal, not the order
        random.shuffle(indices)

        for i in indices:
            x, y = X_train[i], Y_train[i]
            optimizer.zero_grad()

            # Add Batch Dimension: (Seq_Len, 3) -> (1, Seq_Len, 3)
            x_batched = x.unsqueeze(0)
            logits = model(x_batched)

            # CTC Loss expects: (Seq_Len, Batch, Num_Classes)
            logits_ctc = logits.transpose(0, 1)

            input_lengths = torch.tensor([logits_ctc.size(0)], dtype=torch.long)
            target_lengths = torch.tensor([y.size(0)], dtype=torch.long)

            loss = ctc_loss(logits_ctc.log_softmax(2), y, input_lengths, target_lengths)
            loss.backward()
            # Gradient clipping — LSTMs blow up without it
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(X_train):.4f}")

    # ---------------------------------------------------------
    # 5. ONNX EXPORT FOR C++
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
            "input_stroke":    {1: "seq_len"},
            "predicted_logits": {1: "seq_len"},
        },
    )

    # --- THE MAGIC FIX FOR C++ ---
    # PyTorch's exporter is too modern for your C++ system library.
    # We will manually downgrade the internal file version so C++ can read it.
    onnx_model = onnx.load(onnx_path)
    onnx_model.ir_version = 9
    onnx.save(onnx_model, onnx_path)

    print(f"\n[SUCCESS] AI Brain exported to: {onnx_path} (IR Version 9)")
    print("The C++ decoder is now ready to receive this brain.")


if __name__ == "__main__":
    train_and_export()
