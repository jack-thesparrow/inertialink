import time
import math
import random
import os

# Connect to the "ESP32" end of our virtual cable
PORT = "/tmp/vtty_esp32"

print(f"[Simulator] Booting Virtual ESP32 on {PORT}...")
print("[Simulator] Transmitting 6-DOF data at 100Hz. Press Ctrl+C to stop.")

try:
    with open(PORT, "w") as f:
        t = 0
        while True:
            # Accelerometer (g-force) — simulates gentle pen movement + gravity on Z
            acc_x = math.sin(t) * 0.15 + random.uniform(-0.02, 0.02)
            acc_y = math.cos(t * 0.7) * 0.10 + random.uniform(-0.02, 0.02)
            acc_z = 1.0 + math.sin(t * 0.3) * 0.05 + random.uniform(-0.01, 0.01)

            # Gyroscope (deg/s) — simulates pen tilting/rotating while writing
            gyro_x = math.sin(t * 1.5) * 30.0 + random.uniform(-2, 2)
            gyro_y = math.cos(t * 0.8) * 20.0 + random.uniform(-2, 2)
            gyro_z = math.sin(t * 0.4) * 10.0 + random.uniform(-1, 1)

            # Format: ax,ay,az,gx,gy,gz
            f.write(f"{acc_x:.4f},{acc_y:.4f},{acc_z:.4f},{gyro_x:.4f},{gyro_y:.4f},{gyro_z:.4f}\n")
            f.flush()

            t += 0.05
            time.sleep(0.01)  # 10ms delay = 100Hz

except FileNotFoundError:
    print(f"Error: {PORT} not found. Did you run the socat command first?")
except KeyboardInterrupt:
    print("\n[Simulator] Shutting down.")
