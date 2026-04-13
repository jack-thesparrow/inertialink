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
        # Simulated pen tilt angle (~35° from horizontal)
        TILT_RAD = math.radians(35.0)
        cos_t = math.cos(TILT_RAD)
        sin_t = math.sin(TILT_RAD)

        while True:
            # === Gravity projection for tilted pen ===
            # At 35° tilt: gravity projects onto sensor X and Z
            grav_x = -math.sin(TILT_RAD)  # ~-0.57g along pen shaft
            grav_z =  math.cos(TILT_RAD)  # ~0.82g perpendicular to PCB

            # Accelerometer (g-force) — gravity + simulated writing dynamics
            acc_x = grav_x + math.sin(t) * 0.15 + random.uniform(-0.02, 0.02)
            acc_y = math.cos(t * 0.7) * 0.10 + random.uniform(-0.02, 0.02)
            acc_z = grav_z + math.sin(t * 0.3) * 0.05 + random.uniform(-0.01, 0.01)

            # Gyroscope (deg/s) — pen-shaft-aligned mounting:
            #   GX = pitch (pen tilts forward/back → vertical stroke)
            #   GY = yaw   (wrist turns left/right → horizontal stroke, DOMINANT)
            #   GZ = roll  (pen twist → minimal during writing)
            gyro_x = math.sin(t * 1.5) * 20.0 + random.uniform(-2, 2)    # pitch
            gyro_y = math.cos(t * 0.8) * 30.0 + random.uniform(-2, 2)    # yaw (dominant)
            gyro_z = math.sin(t * 0.4) * 5.0  + random.uniform(-1, 1)    # roll (small)

            # Format: ax,ay,az,gx,gy,gz
            f.write(f"{acc_x:.4f},{acc_y:.4f},{acc_z:.4f},{gyro_x:.4f},{gyro_y:.4f},{gyro_z:.4f}\n")
            f.flush()

            t += 0.05
            time.sleep(0.01)  # 10ms delay = 100Hz

except FileNotFoundError:
    print(f"Error: {PORT} not found. Did you run the socat command first?")
except KeyboardInterrupt:
    print("\n[Simulator] Shutting down.")
