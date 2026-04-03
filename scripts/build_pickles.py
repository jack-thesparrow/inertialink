import os
import glob
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

TARGET_SEQUENCE_LENGTH = 100


def process_stroke(csv_path):
    """Resamples and normalizes a single stroke CSV."""
    df = pd.read_csv(csv_path)

    # Extract raw data
    raw_pitch = df["pitch"].values
    raw_roll = df["roll"].values
    raw_yaw = df["yaw"].values

    old_time = np.linspace(0, 1, num=len(df))
    new_time = np.linspace(0, 1, num=TARGET_SEQUENCE_LENGTH)

    # Cubic Interpolation
    resampled_pitch = interp1d(old_time, raw_pitch, kind="cubic")(new_time)
    resampled_roll = interp1d(old_time, raw_roll, kind="cubic")(new_time)
    resampled_yaw = interp1d(old_time, raw_yaw, kind="cubic")(new_time)

    resampled_data = np.column_stack((resampled_pitch, resampled_roll, resampled_yaw))

    # Normalization
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(resampled_data)

    return normalized_data


def build_dataset(data_dir):
    all_x_data = []  # Holds the 100x3 arrays
    all_gt = []  # Holds the labels ('A', 'B', etc.)

    # Find all subdirectories (which act as our labels: 'A', 'B', '1', '2')
    labels = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]

    print(f"Found labels: {labels}")

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        # Find all CSVs in this folder
        csv_files = glob.glob(os.path.join(label_dir, "*.csv"))

        print(f"Processing {len(csv_files)} samples for letter '{label}'...")

        for csv_file in csv_files:
            try:
                processed_array = process_stroke(csv_file)
                all_x_data.append(processed_array)
                all_gt.append(label)
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

    # Convert to NumPy arrays for efficient ML training
    X = np.array(all_x_data)
    Y = np.array(all_gt)

    return X, Y


if __name__ == "__main__":
    # Point this to where your C++ app is saving the folders
    data_directory = "../data"

    print("Building Smart Pen Dataset...")
    X, Y = build_dataset(data_directory)

    print("\n=== Dataset Summary ===")
    print(f"IMU Data Shape (X): {X.shape}")  # Should be (Samples, 100, 3)
    print(f"Labels Shape (Y):   {Y.shape}")  # Should be (Samples,)

    # --- SAVE THE EXACT PICKLES THE REPO EXPECTS ---
    output_dir = "../data"

    x_file = os.path.join(output_dir, "all_x_dat_imu.pkl")
    with open(x_file, "wb") as f:
        pickle.dump(X, f)

    y_file = os.path.join(output_dir, "all_gt.pkl")
    with open(y_file, "wb") as f:
        pickle.dump(Y, f)

    print(f"\n[SUCCESS] Saved {x_file}")
    print(f"[SUCCESS] Saved {y_file}")
