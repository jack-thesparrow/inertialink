import math
import os
import time
import socket
import pandas as pd

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
LEVER_ARM_MM = 150.0  # Must match your C++ decoder exactly

print("=== ESP32 Hardware Simulator ===")
print(f"Target: {UDP_IP}:{UDP_PORT} (WiFi Mode)")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def stream_csv(filepath):
    df = pd.read_csv(filepath)
    print(f"\n[Hardware] Streaming {filepath} at 100Hz...")

    for index, row in df.iterrows():
        # REVERSE PHYSICS: Convert the 2D CSV data back into raw 3D IMU angles.
        # C++ did: x = -yaw_rad * 150  -> yaw_rad = -x / 150
        # C++ did: y = pitch_rad * 150 -> pitch_rad = y / 150
        # io.cpp multiplies by (pi/180) expecting DEGREES (like real ESP32 firmware),
        # so we must convert our radian angles to degrees before sending.

        yaw = math.degrees(-(row["x"] / LEVER_ARM_MM))
        pitch = math.degrees(row["y"] / LEVER_ARM_MM)
        roll = 0.0  # We didn't simulate roll
        accel_z = row["accel_z"]

        # Format EXACTLY like the physical ESP32 firmware will send it
        msg = f"{pitch:.4f},{roll:.4f},{yaw:.4f},{accel_z:.4f}\n".encode()
        sock.sendto(msg, (UDP_IP, UDP_PORT))

        time.sleep(0.01)  # 10ms = 100Hz hardware speed

    # Return the last row so the caller can use its position for idle frames
    return df.iloc[-1]


if __name__ == "__main__":
    # Let's test the AI by sending it the word "hello"
    sample_file = "data/hello/sample_001.csv"
    target_word = os.path.basename(os.path.dirname(sample_file))

    if os.path.exists(sample_file):
        print(f"\n[Hardware] Target word: \"{target_word}\"  <-- compare this to the AI prediction")

        # 1. Send idle data for a second to let C++ calibrate
        print("[Hardware] Sending idle data...")
        for _ in range(100):
            sock.sendto(b"0.0,0.0,0.0,0.0\n", (UDP_IP, UDP_PORT))
            time.sleep(0.01)

        # 2. Stream the actual stroke; get final pen position
        last_row = stream_csv(sample_file)

        # 3. Send the pen's FINAL RESTING POSITION as idle data.
        #
        # Why not zeros: training CSVs were collected by the data_collector which
        # kept recording for 2 s after the pen lifted — during that time the pen
        # stayed at the end of the stroke (x ≈ final_x, not x ≈ 0).  Sending
        # zeros here made the decoder's trailing 200 frames jump back to x ≈ 0,
        # a pattern the model never saw in training, causing wrong predictions.
        final_yaw   = math.degrees(-(last_row["x"] / LEVER_ARM_MM))
        final_pitch = math.degrees( last_row["y"] / LEVER_ARM_MM)
        idle_msg = f"{final_pitch:.4f},0.0000,{final_yaw:.4f},0.0000\n".encode()

        print(f"[Hardware] Pen lifted. Sent: \"{target_word}\" — waiting for AI to process...")
        for _ in range(250):
            sock.sendto(idle_msg, (UDP_IP, UDP_PORT))
            time.sleep(0.01)

        print("[Hardware] Simulation complete.")
    else:
        print(
            f"Error: Could not find {sample_file}. Run generate_synthetic_data.py first."
        )
