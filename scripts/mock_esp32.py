import math
import os
import sys
import glob
import random
import time
import socket
import pandas as pd

UDP_IP   = "127.0.0.1"
UDP_PORT = 5005
LEVER_ARM_MM = 150.0  # Must match pen::Defaults::leverArmMm in io.hpp

# All words the model was trained on
TRAINED_WORDS = ["hello", "world", "pen", "123", "write",
                 "note", "data", "code", "test", "abc", "xyz", "open"]

print("=== ESP32 Hardware Simulator ===")
print(f"Target: {UDP_IP}:{UDP_PORT} (WiFi Mode)")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def stream_csv(filepath: str) -> pd.Series:
    df = pd.read_csv(filepath)
    print(f"[Hardware] Streaming {filepath} at 100Hz...")

    for _, row in df.iterrows():
        # REVERSE PHYSICS: CSV (x, y) → IMU angles (degrees) for io.cpp
        # data_collector did:  x_mm = -yaw_rad * 150  → yaw_rad = -x / 150
        #                      y_mm =  pitch_rad * 150 → pitch_rad = y / 150
        # io.cpp expects DEGREES (like the real ESP32 firmware).
        yaw   = math.degrees(-(row["x"] / LEVER_ARM_MM))
        pitch = math.degrees( row["y"] / LEVER_ARM_MM)
        roll  = 0.0
        msg = f"{pitch:.4f},{roll:.4f},{yaw:.4f},{row['accel_z']:.4f}\n".encode()
        sock.sendto(msg, (UDP_IP, UDP_PORT))
        time.sleep(0.01)  # 10 ms = 100 Hz

    return df.iloc[-1]


def simulate_word(word: str, sample: str = "random") -> None:
    """Stream one sample CSV for `word` and print the expected prediction.

    sample: "random" picks a random sample, "001" uses sample_001.csv, etc.
    """
    folder = f"data/{word}"
    if not os.path.isdir(folder):
        print(f"[Error] No training data for '{word}'. "
              f"Run generate_synthetic_data.py first.")
        return

    csvs = sorted(glob.glob(f"{folder}/sample_*.csv"))
    if not csvs:
        print(f"[Error] No CSV files found in {folder}/")
        return

    if sample == "random":
        filepath = random.choice(csvs)
    else:
        filepath = f"{folder}/sample_{sample.zfill(3)}.csv"
        if not os.path.exists(filepath):
            print(f"[Error] {filepath} not found.")
            return

    print(f"\n[Hardware] Target word : \"{word}\"  ← compare to AI prediction")
    print(f"[Hardware] Sample file : {filepath}")

    # 1. Idle burst so C++ can reach its wake-on-impact state
    print("[Hardware] Sending idle data...")
    for _ in range(100):
        sock.sendto(b"0.0,0.0,0.0,0.0\n", (UDP_IP, UDP_PORT))
        time.sleep(0.01)

    # 2. Stream the actual stroke
    last_row = stream_csv(filepath)

    # 3. Final resting position (NOT zeros) for the idle tail.
    # Training CSVs kept recording for 2 s after pen lift so the model
    # expects the pen to stay at the end-of-stroke position, not jump to 0.
    final_yaw   = math.degrees(-(last_row["x"] / LEVER_ARM_MM))
    final_pitch = math.degrees( last_row["y"] / LEVER_ARM_MM)
    idle_msg = f"{final_pitch:.4f},0.0000,{final_yaw:.4f},0.0000\n".encode()

    print(f"[Hardware] Pen lifted — waiting for AI to process...")
    for _ in range(250):
        sock.sendto(idle_msg, (UDP_IP, UDP_PORT))
        time.sleep(0.01)

    print("[Hardware] Simulation complete.\n")


def print_usage() -> None:
    print("\nUsage:")
    print("  python3 scripts/mock_esp32.py                → test 'hello' sample_001")
    print("  python3 scripts/mock_esp32.py <word>         → test word, random sample")
    print("  python3 scripts/mock_esp32.py <word> <N>     → test word, sample_00N.csv")
    print("  python3 scripts/mock_esp32.py all            → cycle through all 12 words")
    print(f"\nTrained words: {', '.join(TRAINED_WORDS)}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        # Default: hello sample_001
        simulate_word("hello", "001")

    elif args[0] == "all":
        # Cycle all 12 trained words with a random sample each
        for w in TRAINED_WORDS:
            simulate_word(w, "random")
            time.sleep(1.0)

    elif args[0] in ("-h", "--help"):
        print_usage()

    else:
        word   = args[0]
        sample = args[1] if len(args) > 1 else "random"
        if word not in TRAINED_WORDS:
            print(f"[Warning] '{word}' was not in training data — prediction may be wrong.")
            print(f"Trained words: {', '.join(TRAINED_WORDS)}")
        simulate_word(word, sample)
