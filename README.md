# Inertialink

> Motion-sensor handwriting recognition — no camera, no touchscreen, just physics.

Inertialink turns an ESP32 + MPU6050 IMU into a smart pen that streams angular
velocity over USB or WiFi. A BiLSTM + CTC neural network running on
your desktop decodes the motion into text in real time. A lazygit-style terminal
UI lets you launch, test, and monitor every component from one screen.

```
ESP32 / MPU6050                Desktop (C++ + Python)
──────────────────             ──────────────────────────────────────────
  pitch                WiFi    ┌──────────────┐    ONNX    ┌───────────┐
  roll    ──100Hz UDP──────▶   │   decoder    │  ───────▶  │  BiLSTM   │
  yaw                          │  (C++/ONNX)  │            │  + CTC    │
  accel_z              USB     └──────────────┘            └───────────┘
                  Serial──▶                                      │
                               ┌──────────────┐            "hello" (94%)
                               │  visualizer  │  ◀── stroke trail + HUD
                               │ (OpenGL 3.3) │
                               └──────────────┘
```

---

## Contents

- [Hardware](#hardware)
- [Quick Start](#quick-start)
- [Build](#build)
- [Workflow](#workflow)
  - [1. Connect](#1-connect)
  - [2. Collect training data](#2-collect-training-data)
  - [3. Generate synthetic data](#3-generate-synthetic-data)
  - [4. Train the model](#4-train-the-model)
  - [5. Run inference](#5-run-inference)
  - [6. Batch-test with the TUI](#6-batch-test-with-the-tui)
- [Terminal UI](#terminal-ui)
- [Binary Reference](#binary-reference)
- [Configuration](#configuration)
- [Docker / Intel Arc XPU](#docker--intel-arc-xpu)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Roadmap](#roadmap)

---

## Hardware

| Component | Spec |
|-----------|------|
| Microcontroller | ESP32-DEVKIT-C (Xtensa dual-core, 240 MHz) |
| IMU | MPU6050 (I²C, pins 21 SDA / 22 SCL) |
| Firmware | PlatformIO + Arduino, `MPU6050_light ^1.1.0` |
| Baud rate | 115200 (USB serial) |
| WiFi streaming | UDP → `127.0.0.1:5005` (decoder) + `:5006` (visualizer) |
| Sample rate | 100 Hz (10 ms per frame) |
| Lever arm | 150 mm wrist-pivot → pen tip |

The firmware supports **three modes** switchable at runtime via serial commands:

| Command | Mode |
|---------|------|
| `MODE:USB` | Stream over USB serial |
| `MODE:WIFI\|SSID\|PASS\|HOST_IP` | Connect to WiFi and stream UDP |
| (default) | IDLE — no streaming |

**Packet format** (all modes):
```
pitch_deg,roll_deg,yaw_deg,accel_z\n
12.34,0.0000,-5.67,0.12\n
```

---

## Quick Start

```bash
# 1. Install system deps, create .venv, set up PlatformIO and serial permissions
./bootstrap.sh

# Log out and back in if this is the first run (group membership change)

# 2. Build all C++ targets
cmake -B build -G Ninja
cmake --build build

# 3. Activate Python env
source .venv/bin/activate

# 4. Generate 2 400 synthetic training samples (200 × 12 words)
python3 scripts/augment_seed_data.py

# 5. Train (~10 min CPU, ~5 min GPU/XPU)
python3 scripts/train_bilstm.py

# 6. Launch the TUI — run everything from one screen
./bin/tui
```

No hardware required for steps 3–6. The mock ESP32 simulator replays CSV data
over the same UDP ports as real hardware.

---

## Build

### Requirements

- CMake ≥ 3.20, Ninja
- GCC / Clang with C++17
- Python 3.10+
- X11 or Wayland dev headers (for GLFW)

`bootstrap.sh` installs everything on Debian/Ubuntu, Arch, and Fedora.

### CMake targets

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build              # all targets
cmake --build build -t decoder   # single target
cmake --build build --parallel   # use all cores
```

| Target | Output |
|--------|--------|
| `data_collector` | `bin/data_collector` |
| `decoder` | `bin/decoder` |
| `visualizer` | `bin/visualizer` |
| `tui` | `bin/tui` |

Intermediate library (`lib/libPenCore.a`) bundles the hardware backend and
visualizer so all four binaries share the same I/O and rendering code.

### Third-party dependencies (FetchContent — no manual install)

| Library | Version | Used for |
|---------|---------|----------|
| GLFW | 3.3.8 | Window / GL context |
| GLM | 1.0.3 | Matrix math |
| GLAD | — | OpenGL 3.3 Core loader |
| ONNX Runtime | 1.17.1 | Model inference (pre-built `.so`) |
| FTXUI | 5.0.0 | Terminal UI |
| stb_easy_font | — | OpenGL HUD text |

---

## Workflow

### 1. Connect

```
./bin/tui
```

Press **`1`** to go to the Connection panel and choose your transport:

| Option | When to use |
|--------|-------------|
| **WiFi** | ESP32 on the same LAN; sends UDP to ports 5005/5006 |
| **USB** | Direct cable (`/dev/ttyUSB0`); lowest latency |
| **Simulation** | No hardware — use Mock ESP32 instead |

> **Serial permissions:** If you get a permission error on USB, run
> `sudo usermod -aG dialout $USER` and log out/in.  `bootstrap.sh` does this
> automatically, but only takes effect after a session restart.

---

### 2. Collect training data

Skip this step if you only want to use synthetic data (see step 3).

```bash
# Record "hello" over WiFi — writes to data/hello/sample_001.csv, _002, …
./bin/data_collector hello wifi

# Or launch from the TUI: Actions panel → Collector → Enter
```

**Collection loop:**
1. Hold the pen still — the collector waits for a Z-axis impact (> 0.5 m/s²)
2. Write the word in the air (150 mm lever arm, wrist as pivot)
3. Rest the pen — after 2 s of stillness the stroke is saved
4. Repeat; samples are numbered automatically

**Minimum:** 20 frames per stroke (short bumps are discarded automatically).
For the model to generalise, collect at least 10 samples per word.

**CSV format:**
```
time_ms,x,y,accel_z
0,0.0,0.0,0.12
10,1.2,0.8,0.10
```
`x` and `y` are lever-arm projected positions in mm.

---

### 3. Generate synthetic data

If you don't have hardware yet (or want to augment real data), generate 2 400
high-quality synthetic samples:

```bash
source .venv/bin/activate
python3 scripts/augment_seed_data.py
```

Creates `data/{word}/sample_001.csv … sample_200.csv` for all 12 trained words.
Each character has a unique, physically-motivated oscillation profile with
per-sample augmentation (speed jitter ±30%, position noise, pressure variation).

Output: ~300 MB of CSV files.

---

### 4. Train the model

```bash
python3 scripts/train_bilstm.py
```

**What happens:**
- Reads all CSVs (or loads `data/dataset_cache.pt` if already cached)
- Trains a 2-layer BiLSTM + CTC network for up to 300 epochs
- Saves `models/pen_model.onnx` (loaded by the C++ decoder)
- Saves `models/checkpoint.pt` every 2 epochs (safe to Ctrl+C and resume)
- Writes per-epoch metrics to `models/training_log.csv`

**Typical times:**

| Hardware | Time |
|----------|------|
| CPU (8-core) | ~10–15 min |
| NVIDIA GPU | ~5–8 min |
| Intel Arc XPU | ~5–8 min (see [Docker section](#docker--intel-arc-xpu)) |

**Resuming after interruption:**
```bash
# Just re-run — checkpoint.pt is detected automatically
python3 scripts/train_bilstm.py
```

**Evaluating accuracy after training:**
```bash
python3 scripts/eval_model.py
# Writes models/eval_results.csv and models/eval_summary.csv
# Typical accuracy: 95–99% on synthetic data
```

---

### 5. Run inference

Start the decoder and visualizer (from the TUI or directly):

```bash
# Terminal 1 — 3D visualizer listens on UDP 5006
./bin/visualizer wifi

# Terminal 2 — AI decoder listens on UDP 5005
./bin/decoder wifi

# Terminal 3 — mock hardware OR real ESP32
python3 scripts/mock_esp32.py hello
```

**Decoder output:**
```
====================================
>> PREDICTION : "hello"
>> RAW CTC    : "helo" (snapped to vocab)
>> CONFIDENCE : 94%
>> PER CHAR   : h=98%  e=91%  l=96%  l=95%  o=86%
====================================
```

**Visualizer HUD** shows the stroke being drawn, the current word, and the
decoder mode (green = reading, yellow = predicting).

The decoder pipeline:

```
UDP frame
  → impact detection (Z > 0.5 m/s²)
  → buffer stroke until 2 s idle
  → project angles → x/y mm  [x = -yaw×150, y = pitch×150]
  → ONNX inference  [BiLSTM logits]
  → CTC beam search (width=10)
  → Levenshtein snap to vocab (edit distance ≤ 2)
  → print prediction
```

---

### 6. Batch-test with the TUI

The TUI is the recommended way to run everything together:

```bash
./bin/tui
```

**Typical test session:**

1. **`1`** → Connection panel → pick WiFi
2. **`2`** → Actions panel → start Visualizer (`↑↓` then `Enter`)
3. **`2`** → start Decoder
4. **`3`** → Test panel → mark words with `Space`, press `A` to stream all marked
5. **`4`** → Output panel → watch per-process predictions scroll in

See the [Terminal UI](#terminal-ui) section for the full key reference.

---

## Terminal UI

```
 ★  Inertialink Smart Pen
────────────────────────────────────────────────────────────
 [1] Connection     │  [2] Actions   ↑↓ navigate   Space/Enter run·stop
  ● USB /dev/ttyUSB0│   1  RUN   Visualizer    3D cube + stroke canvas
  ○ WiFi 5005/5006  │   2  RUN   Decoder       Real-time AI recognition
  ○ Simulation      │   3  RUN   Collector     Record IMU data to CSV
                    │   4  STOP  Mock ESP32    word: hello
                    │   5  RUN   Train         BiLSTM CTC model training
 Status             │   6  RUN   Evaluate      Batch accuracy on all samples
  ● visualizer      │   Word: hello________________  [Tab] to type
  ● decoder         ├────────────────────────────────────────────────────
  ○ collector       │  [3] Test   ↑↓←→ navigate   Space mark   Enter stream
  ● Mock            │   ✓1 hello    2 world    3 pen      4 123
  ○ Train           │    5 write    6 note     7 data     8 code
  ○ Evaluate        │    9 test    10 abc     11 xyz     12 open
                    ├────────────────────────────────────────────────────
                    │  ↗ Now testing: hello  world
                    │  [4] Output
                    │  decoder
                    │  12:34:57  >> PREDICTION : "hello"  CONFIDENCE 94%
                    │  Mock
                    │  12:35:01  [Hardware] Target word: "world"
────────────────────────────────────────────────────────────
[1-4] panel  [Tab] cycle  [↑↓] nav  [Space] select  [Enter] run
[A] all  [K] kill  [C] clear  [PgUp/Dn] scroll  [?] help  [Q] quit
```

### Panels

| Key | Panel | Purpose |
|-----|-------|---------|
| `1` | Connection | Pick USB / WiFi / Simulation |
| `2` | Actions | Start / stop all 6 tools |
| `3` | Test | Navigate 12-word grid, mark + stream |
| `4` | Output | Live per-process output, scrollable |

### Key bindings

| Key | Action |
|-----|--------|
| `1`–`4` | Switch panel |
| `Tab` | Cycle panels forward |
| `↑` `↓` | Navigate list / word grid |
| `←` `→` | Move across word grid columns |
| `Space` | **Actions:** toggle run/stop · **Test:** mark/unmark word |
| `Enter` | **Actions:** toggle run/stop · **Test:** stream word immediately |
| `A` | Stream all marked words (or all 12 if none marked) |
| `K` | Kill every running process |
| `C` | Clear output log |
| `PgUp` / `PgDn` | Scroll output |
| `?` | Toggle keyboard reference overlay |
| `Q` | Quit (SIGTERM all processes) |
| `Tab` (in Word field) | Focus / unfocus the mock word text input |

---

## Binary Reference

### `bin/data_collector <label> [mode]`

Records IMU strokes to CSV for training.

```bash
./bin/data_collector hello usb    # USB serial (default)
./bin/data_collector hello wifi   # WiFi
```

Saves to `data/<label>/sample_001.csv`, incrementing automatically.

---

### `bin/decoder [mode]`

Real-time BiLSTM + CTC inference engine.

```bash
./bin/decoder          # USB serial (default)
./bin/decoder usb      # USB serial
./bin/decoder wifi     # WiFi, port 5005
```

Loads `models/pen_model.onnx`. Writes `/tmp/inertialink_mode` for the
visualizer HUD. Requires the model to be trained first.

---

### `bin/visualizer [mode]`

OpenGL 3.3 Core Profile — 3D cube + stroke trail.

```bash
./bin/visualizer           # USB serial (default)
./bin/visualizer usb
./bin/visualizer wifi      # WiFi, port 5006
./bin/visualizer sim       # socat virtual TTY at /tmp/vtty_laptop
```

**In-window keys:** `C` clear trail · `Esc` quit

Reads `/tmp/inertialink_word` and `/tmp/inertialink_mode` for the HUD overlays
(polled every 60 frames).

---

### `bin/tui`

Terminal UI — no arguments.

```bash
./bin/tui
```

---

### `scripts/mock_esp32.py`

Hardware simulator — replays training CSVs at 100 Hz over UDP.

```bash
python3 scripts/mock_esp32.py                 # hello, sample_001
python3 scripts/mock_esp32.py hello           # hello, random sample
python3 scripts/mock_esp32.py hello 3         # hello, sample_003.csv
python3 scripts/mock_esp32.py all             # cycle all 12 words
python3 scripts/mock_esp32.py hello world pen # multiple words in sequence
```

---

### `scripts/train_bilstm.py`

Train the BiLSTM + CTC model. Resumes automatically from `checkpoint.pt`.

```bash
python3 scripts/train_bilstm.py
```

---

### `scripts/eval_model.py`

Batch evaluation on all CSV samples.

```bash
python3 scripts/eval_model.py              # all 12 words
python3 scripts/eval_model.py hello world  # specific words only
```

Writes `models/eval_results.csv` and `models/eval_summary.csv`.

---

## Configuration

All physics constants are in one place: `include/pen/io.hpp`.

```cpp
namespace pen {
struct Defaults {
    static constexpr const char *usbPort        = "/dev/ttyUSB0";
    static constexpr int   wifiPort             = 5005;    // decoder UDP
    static constexpr int   wifiVizPort          = 5006;    // visualizer UDP
    static constexpr float leverArmMm           = 150.0f;  // wrist → pen tip
    static constexpr float wakeThresholdZ       = 0.5f;    // impact shock (m/s²)
    static constexpr float activityThreshold    = 0.02f;   // min angle delta (rad)
    static constexpr int   idleTimeoutMs        = 2000;    // pen-lift detection
};
}
```

> **Important:** `leverArmMm` and `idleTimeoutMs` are also hard-coded in the
> Python scripts. If you change them, update `scripts/mock_esp32.py`
> (`LEVER_ARM_MM`) and regenerate synthetic data before retraining.

---

## Docker / Intel Arc XPU

A Docker image is provided for training on Intel Arc GPUs.

### One-time host setup

```bash
sudo usermod -aG video,render $USER
# Log out and back in
```

### Training in Docker

```bash
# Train (auto-builds image on first run, ~5 min)
./docker/train_xpu.sh

# Interactive shell
./docker/train_xpu.sh bash

# Generate data inside container
./docker/train_xpu.sh python3 scripts/augment_seed_data.py
```

The container mounts the project root, so `data/` and `models/` are written to
your local filesystem. The image is based on
`intel/intel-extension-for-pytorch:2.8.10-xpu` (Ubuntu 22.04, PyTorch 2.8,
Level Zero drivers).

---

## Project Structure

```
inertialink/
├── include/pen/
│   ├── io.hpp              Hardware backend: Defaults, IMUData, PenBackend
│   └── viz.hpp             Visualizer class declaration
├── src/
│   ├── io.cpp              SerialReader, UDPReader, IMUFilter, PenBackend
│   └── viz.cpp             OpenGL geometry, shaders, stroke trail, HUD text
├── app/
│   ├── data_collector_main.cpp
│   ├── decoder_main.cpp    ONNX inference, CTC beam search, vocab snapping
│   ├── tui_main.cpp        FTXUI terminal launcher
│   └── visualizer_main.cpp
├── scripts/
│   ├── augment_seed_data.py   2 400 labelled CSV samples
│   ├── train_bilstm.py              BiLSTM + CTC, XPU/CUDA/CPU
│   ├── eval_model.py                Per-word accuracy report
│   └── mock_esp32.py                UDP hardware simulator
├── esp32_firmware/
│   ├── platformio.ini
│   └── src/main.cpp        MPU6050 → pitch/roll/yaw → USB/BT/WiFi UDP
├── docker/
│   ├── Dockerfile.xpu
│   └── train_xpu.sh
├── third_party/
│   ├── glad/
│   └── stb/stb_easy_font.h
├── data/                   Training CSVs (generated)
├── models/                 pen_model.onnx, checkpoint.pt, training_log.csv
├── bin/                    Compiled binaries
├── bootstrap.sh
├── requirements.txt
└── CMakeLists.txt
```

---

## Architecture

### Data flow

```
[ESP32 / mock_esp32.py]
        │  pitch, roll, yaw, accel_z  @100 Hz
        │  UDP 5005 ──────────────────────────▶ [decoder]
        │  UDP 5006 ──────────────────────────▶ [visualizer]
        │  Serial ────────────────────────────▶ [decoder / visualizer]
        │
[decoder]
  wait for Z-shock (> 0.5 m/s²)
  buffer frames until 2 s idle
  project: x_mm = −yaw_rad × 150
           y_mm =  pitch_rad × 150
  ONNX → [1, seq, 3] → BiLSTM → [1, seq, 64 logits]
  CTC beam search (width 10) → raw text
  Levenshtein snap → predicted word
  write /tmp/inertialink_mode, /tmp/inertialink_word

[visualizer]
  rotate cube by IMU angles
  append to stroke trail
  render HUD from /tmp files
```

### ML model

```
Input  (1, seq_len, 3)  — normalized [x_mm, y_mm, accel_z]
  ↓  Z-score normalization (baked into ONNX, no pre-processing needed)
  ↓  BiLSTM (128 units, 2 layers, bidirectional, dropout 0.3)
  ↓  Dense (256 → 64)
Output (1, seq_len, 64) — log-softmax over alphabet + CTC blank
```

**Alphabet (64 chars):**
`~` (blank) + space + `a-z` + `A-Z` + `0-9`

**Trained vocabulary (12 words):**
`hello world pen 123 write note data code test abc xyz open`

---

## Roadmap

- [ ] Full character set (62 chars: a–z, A–Z, 0–9) — profiles already drafted
- [ ] Remove vocab snapping — open-vocabulary CTC for any word
- [ ] Space character training → multi-word sentences
- [ ] Real-hardware fine-tuning with physical pen data
- [ ] Sentence buffer → print on pause / punctuation
- [ ] macOS / Windows port (GLFW and ONNX Runtime are cross-platform)

---

## License

MIT — see [LICENSE](LICENSE).
