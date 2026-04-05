"""
generate_synthetic_data.py  —  High-quality synthetic stroke generator
=======================================================================
Design goals
------------
1. Every character has a UNIQUE, physically-motivated signature so that words
   with shared letters are still distinguishable (unlike the old ord()/20 hack
   that crammed all lowercase into 4.8-6.0 Hz).

2. Oscillation uses LOCAL phase (fraction of the character's own duration)
   so the waveform is the same shape regardless of where in the word the
   character appears.

3. Y-center bands are well-separated by category:
       numbers   : y_center ≈ -15 mm   (drops far below baseline)
       descenders: y_center ≈  -5 mm   (g, j, p, q, y)
       baseline  : y_center ≈   2 mm   (a, c, e, i, m, n, o, r, s, u, v, w, x, z)
       ascenders : y_center ≈  12 mm   (b, d, f, h, k, l, t)
       capitals  : y_center ≈  18 mm

4. Augmentation applied per-sample:
       • Horizontal drift   (global x-axis shift)
       • Vertical offset    (global y-axis shift)
       • Per-frame Gaussian position noise (σ=0.5 mm)
       • Per-frame Gaussian accel_z noise  (σ=0.03)
       • Speed jitter       (duration ± 30 % per character)
       • Pressure jitter    (accel_z scale ± 20 %)

5. 200 samples per word, 12 words → 2 400 total CSVs.

Output format (same as before, directly consumable by train_bilstm.py):
    time_ms, x, y, accel_z
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

SAMPLES_PER_WORD = 200   # 200 × 12 words = 2 400 samples
SAMPLE_RATE_MS   = 10    # 100 Hz, 10 ms per frame

# ---------------------------------------------------------------------------
# PER-CHARACTER PROFILES
# (n_cycles, y_center_mm, y_amp_mm, x_width_mult, base_duration_frames)
#
# n_cycles      – how many full oscillation cycles across the character
# y_center_mm   – vertical resting position of this glyph
# y_amp_mm      – half-amplitude of the oscillation
# x_width_mult  – width relative to the default 2 mm/frame advance
# base_duration – number of 10 ms frames at normal speed
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

# Fallback for any character not explicitly listed
def _default_profile(char):
    """Reasonable defaults for characters not in the table."""
    if char.isupper():
        return (1.5, 16.0, 8.0, 1.0, 22)
    if char.isdigit():
        return (1.5, -14.0, 8.0, 1.0, 20)
    return (1.0, 2.0, 5.0, 1.0, 20)


def get_profile(char):
    if char in CHAR_PROFILE:
        return CHAR_PROFILE[char]
    return _default_profile(char)


# ---------------------------------------------------------------------------
# STROKE GENERATOR
# ---------------------------------------------------------------------------

def generate_word_data(word: str) -> pd.DataFrame:
    """
    Generate one synthetic stroke CSV for *word* with random augmentation.
    Returns a DataFrame with columns [time_ms, x, y, accel_z].
    """
    rng = random.Random()          # fresh RNG per call (uses global seed)

    # --- Global augmentation parameters (vary per sample) ---
    x_drift      = rng.uniform(-5.0,  5.0)    # mm shift across whole stroke
    y_offset     = rng.uniform(-2.0,  2.0)    # mm shift across whole stroke
    speed_scale  = rng.uniform(0.7,  1.3)     # global speed multiplier
    pressure_scale = rng.uniform(0.8, 1.2)    # accel_z scale

    data     = []
    time_ms  = 0
    x        = 0.0
    y        = 0.0

    # ---- 1. WAKE IMPACT ----
    data.append([time_ms, x + x_drift, y + y_offset, 0.0])
    time_ms += SAMPLE_RATE_MS
    data.append([time_ms, x + x_drift, y + y_offset, 2.5])   # Z shockwave

    # ---- 2. WRITE EACH CHARACTER ----
    for char in word:
        n_cycles, y_center, y_amp, x_width_mult, base_dur = get_profile(char)

        # Speed jitter per character
        duration = max(8, int(base_dur * speed_scale * rng.uniform(0.85, 1.15)))

        # Pressure for this character (ballpoint scraping)
        base_pressure = rng.uniform(0.12, 0.35) * pressure_scale

        x_advance = 2.0 * x_width_mult   # mm per frame

        for step in range(duration):
            time_ms += SAMPLE_RATE_MS

            # Local phase [0, 2π * n_cycles] — shape is always the same
            phase = (step / duration) * 2.0 * math.pi * n_cycles

            # X: steady rightward advance + small lateral wobble
            x += x_advance + rng.gauss(0.0, 0.3)

            # Y: oscillation around y_center with Gaussian noise
            y_raw  = y_center + y_amp * math.sin(phase)
            y_noise = rng.gauss(0.0, 0.5)
            y       = y_raw + y_noise

            # accel_z: pressure with noise
            accel_z = abs(rng.gauss(base_pressure, 0.03))

            data.append([
                time_ms,
                x + x_drift,
                y + y_offset,
                accel_z,
            ])

        # Brief hover between characters (pen lifts slightly)
        time_ms += SAMPLE_RATE_MS
        data.append([time_ms, x + x_drift, y + y_offset, 0.0])

    # ---- 3. IDLE TAIL ----
    # Must be >= IDLE_TIMEOUT_MS / SAMPLE_RATE_MS = 200 frames.
    # We use 210 for a small margin of safety.
    final_x = x + x_drift
    final_y = y + y_offset
    for _ in range(210):
        time_ms += SAMPLE_RATE_MS
        # Add tiny noise so the model doesn't learn a perfectly flat line
        data.append([
            time_ms,
            final_x + rng.gauss(0.0, 0.05),
            final_y + rng.gauss(0.0, 0.05),
            0.0,
        ])

    return pd.DataFrame(data, columns=["time_ms", "x", "y", "accel_z"])


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=== Smart Pen High-Quality Synthetic Data Generator ===")
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
