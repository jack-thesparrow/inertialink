import os
import glob
import pickle
import time
import logging
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import (
    Conv1D,
    Input,
    Dense,
    Dropout,
    Bidirectional,
    LSTM,
    GlobalAveragePooling1D,
)
from keras.utils import to_categorical
from tabulate import tabulate

# --- CONFIGURATION ---
DATA_DIR = "data"
MODEL_FILE = "data/smart_pen_bilstm.keras"
TARGET_SEQUENCE_LENGTH = 100
FEATURES = 3  # Pitch, Roll, Yaw
NUM_EPOCHS = 50
BATCH_SIZE = 32
DISPLAY_ALL_RESULTS = False

logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_message(message: str) -> None:
    logging.info(message)


def load_modular_dataset(data_dir):
    """Crawls the data directory and loads all individual character pickles."""
    X_list, Y_list = [], []
    x_files = glob.glob(os.path.join(data_dir, "*", "*_x_dat.pkl"))

    if not x_files:
        raise ValueError(
            f"No pickle files found in {data_dir}. Run build_pickles.py first!"
        )

    for x_file in x_files:
        y_file = x_file.replace("_x_dat.pkl", "_gt.pkl")
        with open(x_file, "rb") as f:
            X_list.append(pickle.load(f))
        with open(y_file, "rb") as f:
            Y_list.append(pickle.load(f))

    return np.concatenate(X_list, axis=0), np.concatenate(Y_list, axis=0)


# --- 1. DATA LOADING & PREPROCESSING ---
start_time = time.time()
log_message("Loading modular IMU data...")
imu_data_padded, gt_data_raw = load_modular_dataset(DATA_DIR)

# Convert string labels ('A', 'B') to categorical integers
label_encoder = LabelEncoder()
gt_data_integers = label_encoder.fit_transform(gt_data_raw)
num_classes = len(label_encoder.classes_)
gt_data_categorical = to_categorical(gt_data_integers, num_classes=num_classes)

# --- 2. DEDUPLICATION ---
log_message("Removing duplicates from dataset...")
unique_indices = []
seen_hashes = set()

for i in range(len(imu_data_padded)):
    sample_hash = imu_data_padded[i].tobytes()
    if sample_hash not in seen_hashes:
        seen_hashes.add(sample_hash)
        unique_indices.append(i)

imu_data_padded = imu_data_padded[unique_indices]
gt_data_categorical = gt_data_categorical[unique_indices]
gt_data_integers = gt_data_integers[unique_indices]

log_message(f"Remaining unique samples: {len(imu_data_padded)}")

# --- 3. TRAIN / TEST SPLIT ---
log_message("Splitting data into Training (80%) and Test (20%)...")
X_train, X_test, y_class_train, y_class_test = train_test_split(
    imu_data_padded,
    gt_data_categorical,
    test_size=0.2,
    random_state=42,
    stratify=gt_data_integers,
)

# Autoencoder targets are exactly identical to the inputs
y_ae_train = X_train
y_ae_test = X_test

# --- 4. MODEL ARCHITECTURE (CNN + Bi-LSTM) ---
log_message("Building CNN + Bi-LSTM architecture...")

input_layer = Input(shape=(TARGET_SEQUENCE_LENGTH, FEATURES))

# CNN Extractor (Padding 'same' keeps our 100 frames intact)
x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(input_layer)
x = Conv1D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
x = Dropout(0.3)(x)

# Bidirectional LSTM for temporal sequence memory
x = Bidirectional(LSTM(128, return_sequences=True))(x)

# Head 1: Classification (What letter is this?)
y_class = GlobalAveragePooling1D()(x)
classification_output = Dense(num_classes, activation="softmax", name="classification")(
    y_class
)

# Head 2: Autoencoder (Reconstruct the Pitch, Roll, Yaw to ensure rich feature learning)
autoencoder_output = Dense(FEATURES, activation="linear", name="autoencoder")(x)

mtl_model = Model(
    inputs=input_layer, outputs=[classification_output, autoencoder_output]
)

mtl_model.compile(
    optimizer="adam",
    loss={"classification": "categorical_crossentropy", "autoencoder": "mse"},
    loss_weights={
        "classification": 1.0,
        "autoencoder": 0.5,  # Give slightly less priority to reconstruction vs prediction
    },
    metrics={"classification": ["accuracy"]},
)

mtl_model.summary()

# --- 5. TRAINING ---
log_message("\nStarting model training...")
mtl_model.fit(
    X_train,
    [y_class_train, y_ae_train],
    validation_data=(X_test, [y_class_test, y_ae_test]),
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
)

mtl_model.save(MODEL_FILE)

# Save Class Map for C++ inference later
with open(MODEL_FILE.replace(".keras", "_classes.pkl"), "wb") as f:
    pickle.dump(label_encoder.classes_, f)
log_message(f"Model and Class Map saved successfully.")

# --- 6. EVALUATION ---
log_message("\nEvaluating model...")
predicted_labels, _ = mtl_model.predict(X_test)
predicted_chars = label_encoder.inverse_transform(np.argmax(predicted_labels, axis=1))
ground_truth_chars = label_encoder.inverse_transform(np.argmax(y_class_test, axis=1))

evaluation = mtl_model.evaluate(
    X_test, [y_class_test, y_ae_test], verbose=0, return_dict=True
)
classification_accuracy = evaluation.get("classification_accuracy", 0.0)

mismatched_data = []
for i in range(len(predicted_chars)):
    if predicted_chars[i] != ground_truth_chars[i]:
        mismatched_data.append([i + 1, predicted_chars[i], ground_truth_chars[i]])

log_message(f"\nClassification Accuracy: {classification_accuracy * 100:.2f}%")

if mismatched_data:
    log_message("\nMismatched Characters:")
    log_message(
        tabulate(
            mismatched_data,
            headers=["Sample #", "Predicted", "Actual"],
            tablefmt="grid",
        )
    )
else:
    log_message("\nPerfect Accuracy! No mismatched characters found.")
