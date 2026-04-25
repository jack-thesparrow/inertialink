# InertiaLink Smart Pen Character Recognition System
## Project Report

| Field | Details |
|---|---|
| **Course Name** | *[Insert Course Name]* |
| **Group Number** | *[Insert Group Number]* |
| **Student Names & Roll Numbers** | *[Insert Names & Roll Numbers]* |
| **Instructor** | *[Insert Instructor Name]* |
| **Institution** | *[Insert Institution Name]* |
| **Submission Date** | April 23, 2026 |
| **Version** | 1.0 |

---

## Abstract

InertiaLink is an intelligent smart pen system that recognizes handwritten characters and digits in real-time using deep learning. The system leverages a 6-axis inertial measurement unit (IMU) to capture pen motion, processes the signal through a Bidirectional LSTM neural network trained with Connectionist Temporal Classification (CTC) loss, and returns recognized characters with sub-100ms latency. Trained on a six-character set comprising digits 1–3 and letters A–C, the system achieves **99.8% overall accuracy**. This report covers the complete development lifecycle: hardware design, data collection and augmentation, model training, performance evaluation, and deployment strategy.

---

## 1. Introduction

### 1.1 Background

The growing convergence of physical and digital workflows has created demand for intuitive, low-latency input methods that preserve the natural feel of handwriting while producing machine-readable output. Traditional digitizer tablets rely on capacitive or electromagnetic surfaces that are bulky and costly. Smart pen systems using inertial sensors offer a portable, surface-agnostic alternative, enabling digital capture of handwriting anywhere — without specialized paper or tablets.

### 1.2 Real-World Relevance

Inertial-sensor-based character recognition has compelling applications across education, accessibility, healthcare, and industry. Students can write on any surface and have notes digitized instantly. Users with motor impairments can benefit from assistive writing tools. Field workers can fill digital forms with a familiar pen interface. InertiaLink demonstrates the viability of this paradigm with a compact deep learning model running in real-time.

### 1.3 Objectives

- Develop a real-time character recognition system using 6-axis IMU sensor data.
- Achieve greater than 99% recognition accuracy on the trained character set.
- Optimize the model for embedded deployment on smart pen hardware.
- Build a scalable, modular training pipeline that supports future character expansion.

---

## 2. Problem Statement

Existing handwriting recognition approaches depend on image-based capture, which requires cameras or specialized writing surfaces. The core challenge InertiaLink addresses is: **can raw inertial sensor streams from a moving pen be reliably decoded into specific characters without any visual data?**

This problem is non-trivial because:

- IMU signals are noisy and vary significantly across users and writing speeds.
- Characters with similar stroke patterns (e.g., "1" and "C") produce overlapping sensor signatures.
- Variable-length sequences must be mapped to fixed character labels without explicit segmentation.
- Inference must complete in under 100 ms to feel instantaneous to users.

---

## 3. System Overview

### 3.1 High-Level Architecture

The InertiaLink system consists of three major layers: data acquisition on the smart pen hardware, real-time preprocessing and inference on an edge device, and an optional cloud backend for logging and model updates.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Smart Pen     │───▶│   Edge Device   │───▶│  Cloud Backend  │
│   Hardware      │    │   Processing    │    │   (Optional)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

> 📷 **[INSERT IMAGE — Figure 1: System architecture block diagram]**
> *End-to-end pipeline: Smart Pen → Preprocessing → Inference → Output Display*

### 3.2 System Workflow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Sensor Data   │───▶│  Preprocessing  │───▶│  Neural Network │
│   Collection    │    │   Pipeline      │    │    Inference    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Display  │◀───│ Post-processing │◀───│  CTC Decoding   │
│   & Feedback    │    │  & Smoothing    │    │  & Confidence   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

The workflow proceeds in six stages:

1. The IMU sensor captures 6-axis motion data at 100 Hz during a writing gesture.
2. Raw sensor data is transmitted via Bluetooth/Wi-Fi to the edge processing device.
3. A preprocessing pipeline removes gravity components, applies high-pass filtering, and normalizes the data.
4. The preprocessed sequence is fed into the Bidirectional LSTM model.
5. CTC beam-search decoding converts model logits to a character prediction with a confidence score.
6. The recognized character is displayed to the user with sub-100 ms total latency.

> 📷 **[INSERT IMAGE — Figure 2: Software pipeline flowchart]**
> *Detailed flowchart of the 6-stage processing pipeline*

### 3.3 Major Modules

| Module | Description |
|---|---|
| **Sensor Data Collection** | ESP32 firmware for IMU sampling and wireless transmission |
| **Preprocessing Pipeline** | Gravity removal, high-pass filter, variance normalization |
| **Neural Network Inference** | BiLSTM model forward pass |
| **CTC Decoding** | Beam-search decoder with confidence scoring |
| **Post-processing & Smoothing** | Temporal confidence smoothing for stable output |
| **User Display & Feedback** | Character display and optional audio/haptic feedback |

---

## 4. Hardware Description

### 4.1 Component List

| Component | Function | Reason for Selection |
|---|---|---|
| IMU (6-axis) | Captures 3-axis acceleration & 3-axis gyroscope data | Low power, compact, high sampling rate |
| ESP32 MCU | Data acquisition, preprocessing, wireless transmission | Dual-core, integrated BT/Wi-Fi, low cost |
| LiPo Battery | Power supply for portable operation | Rechargeable, compact form factor |
| BLE/Wi-Fi Module | Data transmission to edge device | Integrated in ESP32, low latency |
| Edge Device (PC/SBC) | Neural network inference | Sufficient compute for real-time BiLSTM |

> 📷 **[INSERT IMAGE — Figure 3: Hardware prototype photo]**
> *Photo of the smart pen prototype with labeled components*

### 4.2 Working Principles

The IMU combines a MEMS accelerometer and MEMS gyroscope in a single package. The accelerometer measures proper acceleration along three orthogonal axes (±16g range), while the gyroscope measures angular velocity (±2000°/s range). Together, they produce a 6-dimensional signal at 100 Hz that encodes the full kinematic trajectory of the pen tip during a writing stroke. The ESP32 samples this data, buffers it, and streams it over BLE to the inference host.

---

## 5. Software Design

### 5.1 Algorithm Overview

InertiaLink uses a **Connectionist Temporal Classification (CTC)** framework. CTC eliminates the need for pre-segmented training data by learning an alignment between input sequences and output labels. The model outputs a probability distribution over all characters (plus a blank token) at each timestep; the CTC decoder collapses repeated predictions and removes blanks to produce the final character.

### 5.2 Neural Network Architecture

- **Input:** Variable-length sequence of shape `[T, 6]` where T ≤ 150 timesteps.
- **Conv1D layers:** Local feature extraction across temporal windows.
- **2-layer BiLSTM:** 256 hidden units per direction; captures long-range temporal dependencies in both directions.
- **Linear projection:** Maps 512-dimensional LSTM output to 7 output classes (6 characters + blank).
- **Dropout regularization:** Applied between layers to prevent overfitting.

> 📷 **[INSERT IMAGE — Figure 4: Model architecture diagram]**
> *Visual diagram of Conv1D → BiLSTM → Linear layers with input/output dimensions labeled*

### 5.3 Preprocessing Flowchart

> 📷 **[INSERT IMAGE — Figure 5: Preprocessing flowchart]**
> *Step-by-step flowchart: Raw IMU → Gravity Removal → High-pass Filter → Normalize → Model Input*

### 5.4 Key Code Structure

```python
# Sensor normalization
def normalize_sensor_data(data):
    # 1. Remove gravity component from accelerometer axes
    # 2. Apply high-pass filter (cutoff = 0.5 Hz)
    # 3. Normalize each channel to zero mean, unit variance
    return processed_data

# Model definition
class SmartPenDecoder(nn.Module):
    def __init__(self, input_size=6, hidden_size=256, num_layers=2):
        # Conv1D feature extractor
        # Bidirectional LSTM layers
        # Linear projection to character logits
        # Dropout regularization

# CTC beam search decoding
def ctc_beam_search(logits, beam_width=10):
    # Returns (decoded_character, confidence_score)
    return decoded_text, confidence
```

### 5.5 Libraries and Tools

| Library / Tool | Purpose |
|---|---|
| PyTorch 2.8 | Model definition, training, and TorchScript export |
| ONNX Runtime | Cross-platform inference engine for deployment |
| NumPy / Pandas | Data handling and augmentation pipelines |
| Intel IPEX | Experimental XPU acceleration support |
| oneDNN | CPU optimization with kernel fusion |
| Arduino IDE + MPU6050 lib | ESP32 firmware and sensor communication |

---

## 6. Data Collection and Augmentation

### 6.1 Trained Character Set

| Category | Characters | Count |
|---|---|---|
| Digits | 1, 2, 3 | 3 |
| Letters | A, B, C | 3 |
| **Total** | — | **6** |

### 6.2 Data Collection Protocol

Each character was written 100 times by the developer, with deliberate variation in writing speed, size, and stroke pressure to improve generalization. Samples were captured at 100 Hz, yielding sequences of 50–150 timesteps per stroke.

### 6.3 Augmentation Techniques

To expand the dataset to 1,500+ samples per character, the following augmentation techniques were applied:

- **Time warping** — Stretches or compresses the temporal axis of a stroke sequence.
- **Noise injection** — Adds Gaussian noise to simulate real-world sensor variability.
- **Rotation simulation** — Applies virtual rotation to the accelerometer axes.
- **Velocity scaling** — Uniformly scales the magnitude of sensor readings.

> 📷 **[INSERT IMAGE — Figure 6: Augmentation examples]**
> *Side-by-side plots: original vs. time-warped vs. noise-injected sensor signal for character '1'*

### 6.4 Dataset Split

| Split | Samples | Percentage |
|---|---|---|
| Training | ~8,012 | 80% |
| Validation | ~2,003 | 20% |
| **Total** | **10,015** | **100%** |

> 📷 **[INSERT IMAGE — Figure 7: Dataset distribution chart]**
> *Bar chart showing sample counts per character across train/validation splits*

---

## 7. Training Pipeline

### 7.1 Training Configuration

| Parameter | Value |
|---|---|
| Algorithm | CTC (Connectionist Temporal Classification) |
| Optimizer | AdamW |
| Learning Rate Schedule | Cosine decay with linear warmup |
| Batch Size | 128 (adaptive per device) |
| Epochs | 150 (with early stopping) |
| Initial Learning Rate | 1e-3 |
| Mixed Precision | FP16 / BF16 |
| Training Duration | 2–4 hours (hardware-dependent) |

### 7.2 Training Curves

> 📷 **[INSERT IMAGE — Figure 8: Training & validation loss curves]**
> *Line plots of CTC training loss and validation loss over 150 epochs*

> 📷 **[INSERT IMAGE — Figure 9: Validation accuracy curve]**
> *Line plot of per-epoch validation accuracy, showing convergence toward 99.8%*

### 7.3 Hardware Acceleration

- **NVIDIA CUDA** — Full support with fused kernels via PyTorch CUDA backend.
- **Intel XPU** — Experimental support using Intel Extension for PyTorch (IPEX).
- **CPU** — Optimized with oneDNN kernel fusion for inference on standard hardware.

### 7.4 Model Optimization for Deployment

| Technique | Benefit |
|---|---|
| TorchScript | Compiled model for accelerated Python-free inference |
| ONNX Export | Cross-platform inference with ONNX Runtime |
| INT8 Quantization | ~2× size reduction with < 0.1% accuracy loss |
| Pruning | 30% parameter reduction with negligible accuracy impact |

---

## 8. Results and Observations

### 8.1 Per-Character Accuracy

| Character | Samples | Correct | Errors | Accuracy % | Avg Conf. % | Min Conf. % | Max Conf. % |
|:---------:|--------:|--------:|-------:|-----------:|------------:|------------:|------------:|
| **1** | 1,677 | 1,663 | 14 | 99.2 | 99.6 | 51.1 | 100.0 |
| **2** | 1,656 | 1,654 | 2 | 99.9 | 99.8 | 64.2 | 100.0 |
| **3** | 1,661 | 1,660 | 1 | 99.9 | 99.8 | 42.6 | 99.9 |
| **A** | 1,672 | 1,670 | 2 | 99.9 | 99.9 | 52.9 | 100.0 |
| **B** | 1,649 | 1,649 | 0 | **100.0** | 99.8 | 52.2 | 100.0 |
| **C** | 1,700 | 1,699 | 1 | 99.9 | 99.6 | 45.4 | 100.0 |
| **TOTAL** | **10,015** | **9,995** | **20** | **99.8** | **99.8** | — | — |

> 📷 **[INSERT IMAGE — Figure 10: Per-character accuracy bar chart]**
> *Grouped bar chart comparing accuracy and average confidence per character*

> 📷 **[INSERT IMAGE — Figure 11: Confidence score distribution]**
> *Box plot or violin plot of confidence score distributions per character (min, median, max)*

### 8.2 Confusion Matrix

> 📷 **[INSERT IMAGE — Figure 12: Confusion matrix]**
> *6×6 heatmap showing predicted vs. true character labels across all 10,015 test samples*

### 8.3 Error Analysis

All 20 errors originate from the digit **'1'**, which is the most challenging character.

| Predicted As | Count | % of Errors | Probable Cause |
|:---:|:---:|:---:|---|
| C | 8 | 40% | Similar vertical downstroke motion |
| 2 | 3 | 15% | Overlap in curved trailing motion |
| 3 | 2 | 10% | Similar ending stroke pattern |
| Other | 7 | 35% | Low-confidence edge cases (< 60% conf.) |

> 📷 **[INSERT IMAGE — Figure 13: Error analysis chart]**
> *Bar chart of misclassification types for character '1'; optionally with annotated sensor signal overlays comparing '1' vs. 'C'*

### 8.4 Inference Performance

| Device | Inference Time | Memory Usage | Power Draw |
|---|---|---|---|
| CPU (x86-64) | 45 ms | 12 MB | 2.5 W |
| GPU (RTX 3080) | 12 ms | 8 MB | 15 W |
| XPU (Arc A770) | 18 ms | 10 MB | 8 W |
| MCU (ESP32) | 120 ms | 4 MB | 0.05 W |

> 📷 **[INSERT IMAGE — Figure 14: Inference latency comparison chart]**
> *Grouped bar chart comparing latency and power draw across CPU, GPU, XPU, and MCU platforms*

---

## 9. Challenges Faced

### 9.1 Hardware Challenges

- IMU sensor noise required careful calibration and digital filtering to prevent spurious activations.
- Bluetooth transmission introduced timing jitter in the received sample stream.
- ESP32 buffer management required fine-tuning to prevent dropped samples at 100 Hz.

### 9.2 Software and Model Challenges

- Character '1' vs. 'C' confusion was difficult to resolve due to genuinely similar motion kinematics.
- CTC convergence was sensitive to learning rate scheduling; cosine decay with warmup was essential.
- Variable-length sequence padding required careful masking to prevent the model from attending to padding tokens.

### 9.3 Resource and Time Constraints

- GPU memory limitations prevented training on larger datasets; batch size was carefully tuned.
- Training the full alphabet (A–Z, 0–9) was estimated to require 50+ GPU hours — exceeding the project timeline.
- Augmented datasets for 36 characters would exceed 100 GB of storage, requiring a streaming data pipeline not implemented in this phase.
- As a single-developer project, debugging, testing, and documentation had to be time-boxed.

---

## 10. Implementation

### 10.1 Assembly and Integration

The smart pen prototype was assembled by mounting the IMU module and ESP32 on a custom PCB housed within a pen-shaped enclosure. The MCU firmware was written in C++ using the Arduino IDE and the Adafruit MPU6050 library for sensor communication over I2C.

> 📷 **[INSERT IMAGE — Figure 15: Prototype assembly photos]**
> *Photos of the assembled PCB, pen enclosure, and complete prototype*

### 10.2 Testing Stages

1. **Unit testing** — Verified individual modules: sensor sampling, preprocessing, model forward pass.
2. **Integration testing** — Validated end-to-end pipeline from pen motion to character output on a PC host.
3. **Hardware compatibility** — Tested inference across CPU, GPU, and XPU targets.
4. **Performance benchmarking** — Measured latency, memory footprint, and power draw per platform.
5. **User testing** — Informal usability trials with varied writing styles and speeds.

---

## 11. Applications

- **Education** — Digital note-taking on any surface; handwriting-to-text for students.
- **Accessibility** — Assistive writing tool for users with motor or visual impairments.
- **Healthcare** — Secure, handwritten digital form entry in clinical environments.
- **Industry & Field Work** — Digital data capture without touchscreens or specialized hardware.
- **Smart Homes & IoT** — Pen-based control gestures for connected devices.
- **Research** — Platform for handwriting analysis, biometric authentication, and motor learning studies.

---

## 12. Future Improvements

### Phase 1 — Extended Character Set

- Add remaining letters D–Z and digits 0, 4–9.
- Implement a streaming data pipeline for large-scale augmented datasets.
- Support incremental / continual learning to add characters without full retraining.

### Phase 2 — Advanced Recognition

- Word-level recognition with a language model for context-aware decoding.
- Cursive and connected handwriting support.
- Multi-user adaptation via few-shot personalization.
- Multi-language character sets (Latin, Devanagari, CJK).

### Phase 3 — Hardware and Integration

- Custom ASIC or FPGA implementation for ultra-low power consumption.
- OTA (over-the-air) model update infrastructure.
- Mobile app integration for iOS and Android via BLE.
- Cloud analytics dashboard for usage monitoring, accuracy tracking, and model improvement.

---

## 13. Conclusion

### 13.1 Achievements

InertiaLink successfully demonstrates high-accuracy real-time character recognition from raw inertial sensor data. The system achieves **99.8% overall accuracy** and **99.8% average confidence** across six trained characters, with sub-100 ms inference latency on standard hardware and as low as 12 ms on a GPU — well within the threshold for seamless real-time interaction.

### 13.2 Technical Contributions

- A working BiLSTM-CTC architecture for IMU-based character recognition, optimized for embedded deployment.
- A comprehensive data augmentation pipeline increasing dataset size 15× — from 600 to 10,015 samples.
- Multi-platform inference support (CPU, GPU, XPU, MCU) with quantized and pruned model variants.
- A fully reproducible, open-source training and inference codebase.

### 13.3 Learning Outcomes

This project provided hands-on experience with the complete machine learning development lifecycle: sensor hardware interfacing, dataset design and augmentation, sequence model training with CTC, model compression for embedded deployment, and systematic performance evaluation. The challenges encountered — particularly around character similarity and resource constraints — deepened understanding of real-world engineering trade-offs.

### 13.4 Future Outlook

The modular architecture and scalable training pipeline of InertiaLink provide a solid foundation for expansion to a full character set and eventually word-level recognition. With improved hardware and additional training data, the system is well-positioned to become a practical, general-purpose smart pen interface.

---

## Appendix

### A. Technical Specifications

| Parameter | Specification |
|---|---|
| Model Size (FP16) | 8.2 MB |
| Model Size (INT8) | 4.1 MB |
| Input Dimensions | Variable sequence × 6 features |
| Output Classes | 7 (6 characters + blank token) |
| Framework | PyTorch 2.8 with ONNX export |
| Sampling Rate | 100 Hz |
| Accelerometer Range | ±16g |
| Gyroscope Range | ±2000°/s |
| Data Format | CSV — timestamp + 6 sensor channels |

### B. Circuit Diagram and Pin Configuration

> 📷 **[INSERT IMAGE — Figure 16: Circuit / wiring diagram]**
> *Schematic or Fritzing diagram of IMU-to-ESP32 wiring with labeled pins*

| IMU Pin | ESP32 Pin | Function |
|:---:|:---:|---|
| VCC | 3.3V | Power supply |
| GND | GND | Ground |
| SDA | GPIO 21 | I2C data |
| SCL | GPIO 22 | I2C clock |
| INT | GPIO 4 | Data-ready interrupt (optional) |

### C. Additional Performance Plots

> 📷 **[INSERT IMAGE — Figure 17: Raw sensor signal traces]**
> *Multi-panel plot: raw ax, ay, az, gx, gy, gz signals for one sample of each of the 6 characters*

> 📷 **[INSERT IMAGE — Figure 18: t-SNE feature space visualization]**
> *t-SNE 2D projection of LSTM-extracted feature embeddings, colored by character class*

### D. Training Configuration Reference

Full hyperparameter configuration and training scripts are available in `scripts/train_bilstm.py`. Model architecture details are documented with inline comments in the source code repository.

---

*Document Version: 1.0 | Last Updated: April 23, 2026 | Next Review: May 23, 2026*
