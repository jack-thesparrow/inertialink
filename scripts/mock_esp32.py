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
        # REVERSE PHYSICS: Convert the 2D CSV data back into raw 3D IMU angles
        # C++ did: x = -yaw * 150  -> So we do: yaw = -x / 150
        # C++ did: y = pitch * 150 -> So we do: pitch = y / 150

        yaw = -(row["x"] / LEVER_ARM_MM)
        pitch = row["y"] / LEVER_ARM_MM
        roll = 0.0  # We didn't simulate roll
        accel_z = row["accel_z"]

        # Format EXACTLY like the physical ESP32 firmware will send it
        msg = f"{pitch:.4f},{roll:.4f},{yaw:.4f},{accel_z:.4f}\n".encode()
        sock.sendto(msg, (UDP_IP, UDP_PORT))

        time.sleep(0.01)  # 10ms = 100Hz hardware speed


if __name__ == "__main__":
    # Let's test the AI by sending it the word "hello"
    sample_file = "data/hello/sample_001.csv"

    if os.path.exists(sample_file):
        # 1. Send idle data for a second to let C++ calibrate
        print("[Hardware] Sending idle data...")
        for _ in range(100):
            sock.sendto(b"0.0,0.0,0.0,0.0\n", (UDP_IP, UDP_PORT))
            time.sleep(0.01)

        # 2. Stream the actual stroke!
        stream_csv(sample_file)

        # 3. Send idle data so the C++ triggers the 2-second timeout
        print("[Hardware] Pen lifted. Waiting for AI to process...")
        for _ in range(250):
            sock.sendto(b"0.0,0.0,0.0,0.0\n", (UDP_IP, UDP_PORT))
            time.sleep(0.01)

        print("[Hardware] Simulation complete.")
    else:
        print(
            f"Error: Could not find {sample_file}. Run generate_synthetic_data.py first."
        )
