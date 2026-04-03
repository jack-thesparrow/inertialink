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


def build_distributed_pickles(data_dir):
    # Find all subdirectories ('A', 'B', etc.)
    labels = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]

    print(f"Found folders: {labels}")

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        csv_files = glob.glob(os.path.join(label_dir, "*.csv"))

        if not csv_files:
            continue

        print(f"\nProcessing {len(csv_files)} samples for letter '{label}'...")

        label_x_data = []
        label_gt_data = []

        for csv_file in csv_files:
            try:
                processed_array = process_stroke(csv_file)
                label_x_data.append(processed_array)
                label_gt_data.append(label)
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

        # Convert to NumPy arrays
        X = np.array(label_x_data)
        Y = np.array(label_gt_data)

        # Define paths INSIDE the character folder
        x_file = os.path.join(label_dir, f"{label}_x_dat.pkl")
        y_file = os.path.join(label_dir, f"{label}_gt.pkl")

        # Save the pickles
        with open(x_file, "wb") as f:
            pickle.dump(X, f)
        with open(y_file, "wb") as f:
            pickle.dump(Y, f)

        print(f"  -> Saved {x_file}")
        print(f"  -> Saved {y_file}")


if __name__ == "__main__":
    # Point this to the data folder inside the project root
    data_directory = "data"

    print("Building Modular Smart Pen Dataset...")
    build_distributed_pickles(data_directory)
    print("\n[SUCCESS] All characters processed into their respective folders.")
