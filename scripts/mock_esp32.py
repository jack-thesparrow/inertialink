import time
import math
import random
import os

# Connect to the "ESP32" end of our virtual cable
PORT = "/tmp/vtty_esp32"

print(f"[Simulator] Booting Virtual ESP32 on {PORT}...")
print("[Simulator] Transmitting data at 100Hz. Press Ctrl+C to stop.")

try:
    with open(PORT, "w") as f:
        t = 0
        while True:
            # Generate a baseline "idle" wave with some random high-frequency noise
            # This simulates a hand holding a pen while breathing
            pitch = math.sin(t) * 15.0 + random.uniform(-1, 1)
            roll = math.cos(t * 0.5) * 10.0 + random.uniform(-1, 1)
            yaw = math.sin(t * 0.2) * 5.0 + random.uniform(-0.5, 0.5)

            # Format EXACTLY like the Arduino code: Pitch,Roll,Yaw\n
            f.write(f"{pitch:.2f},{roll:.2f},{yaw:.2f}\n")
            f.flush()

            t += 0.05
            time.sleep(0.01)  # 10ms delay = 100Hz

except FileNotFoundError:
    print(f"Error: {PORT} not found. Did you run the socat command first?")
except KeyboardInterrupt:
    print("\n[Simulator] Shutting down.")
