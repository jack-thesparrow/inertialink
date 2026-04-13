"""
generate_synthetic_data.py  —  High-quality synthetic stroke generator (6-DOF)
================================================================================
Generates training data in the 6-channel raw sensor format:
    time_ms, ax, ay, az, gx, gy, gz

Each channel simulates the physical signature of writing:
  - ax, ay: lateral accelerations from pen movement (proportional to character
    shape oscillation velocity changes)
  - az: vertical acceleration including gravity (~1.0g at rest, spikes on impact,
    drops near zero when pen lifts)
  - gx, gy, gz: angular velocities from wrist rotation during writing
    (proportional to character shape — sharper turns = higher angular velocity)

Design goals
------------
1. Every character has a UNIQUE, physically-motivated 6-DOF signature.
2. Oscillation uses LOCAL phase (fraction of the character's own duration).
3. Augmentation: drift, noise, speed/pressure jitter applied per-sample.
4. 200 samples per word, 12 words → 2400 total CSVs.
"""

import math
import os
import random

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# WORDS TO TRAIN ON
# ---------------------------------------------------------------------------
WORDS = [
    "hello",
    "world",
    "pen",
    "123",
    "write",
    "note",
    "data",
    "code",
    "test",
    "abc",
    "xyz",
    "open",
]

SAMPLES_PER_WORD = 200   # 200 × 12 words = 2400 samples
SAMPLE_RATE_MS   = 10    # 100 Hz, 10 ms per frame
DT               = SAMPLE_RATE_MS / 1000.0  # seconds per frame

# ---------------------------------------------------------------------------
# PER-CHARACTER PROFILES
# (n_cycles, y_center_mm, y_amp_mm, x_width_mult, base_duration_frames)
# ---------------------------------------------------------------------------
CHAR_PROFILE = {
    # --- LOWERCASE baseline glyphs (small, round) ---
    'a': (1.5,  2.0,  5.0, 1.0, 22),
    'c': (0.8,  2.5,  6.0, 0.9, 18),
    'e': (1.2,  3.0,  5.5, 0.9, 20),
    'i': (0.5,  3.0,  3.0, 0.6, 14),
    'm': (3.0,  2.0,  4.5, 1.4, 26),
    'n': (2.0,  2.0,  4.5, 1.1, 22),
    'o': (1.0,  2.5,  6.5, 1.0, 20),
    'r': (1.0,  3.5,  3.5, 0.8, 16),
    's': (1.8,  2.0,  5.0, 0.9, 20),
    'u': (1.0,  1.5,  5.5, 1.0, 20),
    'v': (1.0,  2.0,  6.0, 0.9, 18),
    'w': (2.0,  2.0,  5.5, 1.3, 24),
    'x': (2.0,  2.0,  6.0, 1.0, 18),
    'z': (1.5,  2.0,  4.5, 1.0, 18),

    # --- ASCENDERS (tall upward strokes) ---
    'b': (2.0, 12.0,  8.0, 1.1, 24),
    'd': (1.5, 13.0,  8.5, 1.1, 24),
    'f': (1.0, 14.0,  7.0, 0.8, 22),
    'h': (2.5, 12.0,  7.5, 1.2, 26),
    'k': (2.0, 13.0,  8.0, 1.1, 24),
    'l': (0.5, 14.0,  5.0, 0.7, 20),
    't': (1.5, 11.0,  6.5, 0.9, 22),

    # --- DESCENDERS (tail below baseline) ---
    'g': (1.5, -5.0,  9.0, 1.1, 24),
    'j': (0.5, -5.0,  7.0, 0.6, 18),
    'p': (1.5, -4.0,  8.5, 1.1, 24),
    'q': (1.5, -5.0,  8.0, 1.1, 24),
    'y': (1.0, -4.0,  8.5, 1.1, 22),

    # --- DIGITS (drop far below baseline) ---
    '0': (1.0, -15.0, 10.0, 1.0, 22),
    '1': (0.3, -13.0,  4.0, 0.6, 16),
    '2': (1.5, -15.0,  9.0, 1.0, 22),
    '3': (2.0, -15.0,  8.5, 1.0, 22),
    '4': (1.5, -14.0,  8.0, 1.0, 20),
    '5': (2.0, -15.0,  9.0, 1.0, 22),
    '6': (1.5, -15.0,  9.5, 1.0, 24),
    '7': (0.8, -13.0,  7.0, 0.9, 18),
    '8': (2.5, -15.0,  9.0, 1.0, 24),
    '9': (1.5, -15.0,  9.0, 1.0, 22),

    # --- CAPITALS ---
    'A': (2.0, 18.0, 10.0, 1.2, 26),
    'B': (2.5, 18.0,  9.5, 1.2, 26),
    'C': (0.8, 17.0, 10.0, 1.1, 22),
    'D': (1.5, 18.0, 10.0, 1.2, 24),
    'E': (2.0, 18.0,  9.0, 1.1, 24),
    'H': (2.5, 17.0,  9.5, 1.3, 26),
    'N': (2.5, 17.0,  9.0, 1.3, 26),
    'O': (1.0, 17.5, 11.0, 1.1, 22),
    'P': (1.5, 18.0,  8.5, 1.1, 24),
    'R': (1.5, 18.0,  9.0, 1.1, 24),
    'T': (1.0, 18.0,  7.0, 1.0, 22),
    'W': (2.5, 17.0,  9.5, 1.4, 28),
    'X': (2.0, 17.0,  9.5, 1.1, 22),
}


def _default_profile(char):
    if char.isupper():
        return (1.5, 16.0, 8.0, 1.0, 22)
    if char.isdigit():
        return (1.5, -14.0, 8.0, 1.0, 20)
    return (1.0, 2.0, 5.0, 1.0, 20)


def get_profile(char):
    return CHAR_PROFILE.get(char, _default_profile(char))


# ---------------------------------------------------------------------------
# STROKE GENERATOR — 6-DOF VERSION
# ---------------------------------------------------------------------------

def generate_word_data(word: str) -> pd.DataFrame:
    """
    Generate one synthetic 6-DOF stroke CSV for *word* with random augmentation.
    Returns a DataFrame with columns [time_ms, ax, ay, az, gx, gy, gz].

    Physics model:
    - We generate an idealized pen-tip position trajectory (x_mm, y_mm) using
      the same character profiles as before.
    - Accelerations (ax, ay) are derived from the 2nd derivative of position.
    - az simulates gravity + pen-paper contact pressure.
    - Gyroscope values (gx, gy, gz) are derived from angular velocity of the
      pen (wrist rotation), proportional to the rate of change of the stroke
      direction. The pen pivots around the wrist, so lateral movement creates
      angular velocity.
    """
    rng = random.Random()

    # Global augmentation
    speed_scale    = rng.uniform(0.7, 1.3)
    pressure_scale = rng.uniform(0.8, 1.2)
    gyro_scale     = rng.uniform(0.7, 1.3)

    # First: generate the position trajectory
    positions = []  # list of (x_mm, y_mm) tuples
    x, y = 0.0, 0.0

    for char in word:
        n_cycles, y_center, y_amp, x_width_mult, base_dur = get_profile(char)
        duration = max(8, int(base_dur * speed_scale * rng.uniform(0.85, 1.15)))
        x_advance = 2.0 * x_width_mult

        for step in range(duration):
            phase = (step / duration) * 2.0 * math.pi * n_cycles
            x += x_advance + rng.gauss(0.0, 0.3)
            y_raw = y_center + y_amp * math.sin(phase)
            y = y_raw + rng.gauss(0.0, 0.5)
            positions.append((x, y))

        # Hover between characters
        positions.append((x, y))

    # Add idle tail
    for _ in range(210):
        positions.append((x + rng.gauss(0.0, 0.05), y + rng.gauss(0.0, 0.05)))

    # Now derive 6-DOF sensor data from the trajectory
    data = []
    time_ms = 0

    # Compute velocities and accelerations via finite differences
    n_frames = len(positions)
    vx = [0.0] * n_frames
    vy = [0.0] * n_frames
    acc_x = [0.0] * n_frames
    acc_y = [0.0] * n_frames

    for i in range(1, n_frames):
        vx[i] = (positions[i][0] - positions[i-1][0]) / DT  # mm/s
        vy[i] = (positions[i][1] - positions[i-1][1]) / DT

    for i in range(1, n_frames):
        acc_x[i] = (vx[i] - vx[max(0, i-1)]) / DT  # mm/s²
        acc_y[i] = (vy[i] - vy[max(0, i-1)]) / DT

    # Determine which frames are "writing" (before idle tail starts)
    num_writing_frames = n_frames - 210  # approximate

    for i in range(n_frames):
        time_ms = i * SAMPLE_RATE_MS

        # Accelerometer (in g-force, 1g = 9810 mm/s²)
        # Scale pen accelerations to g-force range typical of MPU6050
        ax = acc_x[i] / 9810.0 + rng.gauss(0.0, 0.02)
        ay = acc_y[i] / 9810.0 + rng.gauss(0.0, 0.02)

        # az: gravity baseline + contact pressure during writing
        if i < num_writing_frames:
            base_pressure = rng.uniform(0.08, 0.25) * pressure_scale
            # Check if this is a hover frame (between characters)
            is_hover = False
            frame_in_word = i
            char_start = 0
            for char in word:
                _, _, _, _, base_dur = get_profile(char)
                dur = max(8, int(base_dur * speed_scale * rng.uniform(0.85, 1.15)))
                char_start += dur
                if i == char_start:
                    is_hover = True
                    break

            if is_hover:
                az = rng.gauss(0.0, 0.01)  # pen lifted
            else:
                az = 1.0 + base_pressure + rng.gauss(0.0, 0.03)
        else:
            az = rng.gauss(0.0, 0.01)  # idle: pen lifted

        # Wake impact — spike in total accel magnitude (~2.5g), spread
        # across axes to match pen held at ~50° writing angle.
        if i == 0:
            ax, ay, az = 0.0, 0.0, 0.0
        elif i == 1:
            ax = rng.gauss(0.0, 0.3)   # small lateral jitter
            ay = 1.8                     # along pen shaft (dominant impact axis)
            az = 1.5                     # perpendicular to board face

        # Gyroscope (deg/s) — angular velocity from wrist rotation
        # The pen acts as a lever arm; lateral velocity produces angular velocity
        # ω ≈ v_linear / lever_arm_mm (in rad/s), then convert to deg/s
        lever_arm = 150.0  # mm, matches pen::Defaults::leverArmMm
        gx = (vy[i] / lever_arm) * (180.0 / math.pi) * gyro_scale + rng.gauss(0.0, 2.0)
        gy = (-vx[i] / lever_arm) * (180.0 / math.pi) * gyro_scale + rng.gauss(0.0, 2.0)
        gz = rng.gauss(0.0, 1.0)  # small yaw-axis rotation noise

        data.append([time_ms, ax, ay, az, gx, gy, gz])

    return pd.DataFrame(data, columns=["time_ms", "ax", "ay", "az", "gx", "gy", "gz"])


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=== Smart Pen 6-DOF Synthetic Data Generator ===")
    print(f"Words: {WORDS}")
    print(f"Samples per word: {SAMPLES_PER_WORD}")
    print(f"Total CSVs: {len(WORDS) * SAMPLES_PER_WORD}\n")

    total = 0
    for word in WORDS:
        folder = os.path.join("data", word)
        os.makedirs(folder, exist_ok=True)
        print(f"Generating {SAMPLES_PER_WORD} samples for '{word}'...", end="", flush=True)

        for i in range(1, SAMPLES_PER_WORD + 1):
            filename = os.path.join(folder, f"sample_{i:03d}.csv")
            df = generate_word_data(word)
            df.to_csv(filename, index=False)

        total += SAMPLES_PER_WORD
        print(f"  done  ({total} total so far)")

    print(f"\n[SUCCESS] Generated {total} CSV files.")
    print("Next steps:")
    print("  1. python scripts/train_bilstm.py      # trains and exports pen_model.onnx")
    print("  2. cmake --build build                  # rebuild C++ decoder")
    print("  3. ./bin/decoder  (terminal 1)")
    print("  4. python scripts/mock_esp32.py  (terminal 2)")


if __name__ == "__main__":
    main()
