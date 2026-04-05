import os
import math
import random
import pandas as pd

# The words we want to teach our AI for this simulation test
WORDS = ["hello", "world", "123", "pen"]
SAMPLES_PER_WORD = 50  # Generate 50 simulated strokes for each word


def generate_word_data(word, filename):
    data = []
    time_ms = 0
    x, y = 0.0, 0.0

    # 1. WAKE IMPACT (The pen taps the desk)
    data.append([time_ms, x, y, 0.0])
    time_ms += 10
    data.append([time_ms, x, y, 2.5])  # Massive Z-Axis shockwave!

    # 2. CONTINUOUS WRITING
    for char in word:
        # Give each character a unique mathematical signature based on its ASCII value
        char_val = ord(char)
        duration = random.randint(20, 30)  # Takes 20-30 frames to write a letter

        for _ in range(duration):
            time_ms += 10
            x += random.uniform(1.0, 3.0)  # Pen moves right across the page

            # Y oscillates to create a "squiggle". We add random noise so the AI doesn't overfit to perfection.
            freq = char_val / 20.0
            y = math.sin(time_ms / 50.0 * freq) * 15.0 + random.uniform(-1.0, 1.0)

            # Z vibrates because the ballpoint is scraping paper
            accel_z = random.uniform(0.1, 0.4)

            data.append([time_ms, x, y, accel_z])

        # Tiny hover/pause between letters
        time_ms += 10
        data.append([time_ms, x, y, 0.0])

    # 3. IDLE TIMEOUT — pen holds still at final position so the decoder's
    #    2-second (2000 ms) idle timeout fires correctly.  Must be at least
    #    IDLE_TIMEOUT_MS / 10ms_per_frame = 200 frames; use 210 for margin.
    for _ in range(210):
        time_ms += 10
        data.append([time_ms, x, y, 0.0])

    # Save to our exact Golden CSV Format
    df = pd.DataFrame(data, columns=["time_ms", "x", "y", "accel_z"])
    df.to_csv(filename, index=False)


def main():
    print("=== Smart Pen Synthetic Data Generator ===")
    for word in WORDS:
        folder = f"data/{word}"
        os.makedirs(folder, exist_ok=True)
        print(f"Generating {SAMPLES_PER_WORD} synthetic samples for '{word}'...")

        for i in range(1, SAMPLES_PER_WORD + 1):
            filename = f"{folder}/sample_{i:03d}.csv"
            generate_word_data(word, filename)

    print("\n[SUCCESS] Synthetic dataset generated! You can now run train_bilstm.py")


if __name__ == "__main__":
    main()
