#include "pen/io.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// ML Input Format
struct DataPoint {
  float x;
  float y;
  float accel_z;
};

// Must match train_bilstm.py exactly.  Index 0 ('~') is the CTC blank — it is
// always skipped by the decoder.  Real characters start at index 1.
const std::string ALPHABET =
    "~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

// ---------------------------------------------------------
// ONNX INFERENCE & CTC DECODER
// ---------------------------------------------------------
void runAIInference(Ort::Session &session,
                    const std::vector<DataPoint> &strokeBuffer) {
  if (strokeBuffer.empty())
    return;

  // 1. Flatten our structured data into a raw 1D float array for the Neural
  // Network The AI expects shape: [1 (batch), Sequence_Length, 3 (features)]
  std::vector<float> input_tensor_values;
  input_tensor_values.reserve(strokeBuffer.size() * 3);
  for (const auto &pt : strokeBuffer) {
    input_tensor_values.push_back(pt.x);
    input_tensor_values.push_back(pt.y);
    input_tensor_values.push_back(pt.accel_z);
  }

  // 2. Define the Tensor Shapes
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> input_shape = {
      1, static_cast<int64_t>(strokeBuffer.size()), 3};

  // 3. Create the ONNX Input Tensor
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_tensor_values.data(), input_tensor_values.size(),
      input_shape.data(), input_shape.size());

  const char *input_names[] = {"input_stroke"};
  const char *output_names[] = {"predicted_logits"};

  try {
    // 4. RUN THE AI
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names,
                                      &input_tensor, 1, output_names, 1);

    // 5. EXTRACT THE MATH (Shape: [1, Sequence_Length, Num_Classes])
    float *floatarr = output_tensors.front().GetTensorMutableData<float>();
    auto output_type_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    auto output_shape = output_type_info.GetShape();

    int64_t seq_len = output_shape[1];
    int64_t num_classes = output_shape[2];

    // 6. CTC GREEDY DECODING
    std::string predicted_text = "";
    int last_char_index = 0; // Starts at <BLANK>

    for (int64_t t = 0; t < seq_len; ++t) {
      int best_class = 0;
      float max_prob = -1e9; // Start with a very low number

      // Find the most likely character for this specific millisecond frame
      for (int c = 0; c < num_classes; ++c) {
        float prob = floatarr[t * num_classes + c];
        if (prob > max_prob) {
          max_prob = prob;
          best_class = c;
        }
      }

      // CTC Rule: Ignore <BLANK> (0) and ignore consecutive duplicates!
      if (best_class != 0 && best_class != last_char_index) {
        predicted_text += ALPHABET[best_class];
      }

      last_char_index = best_class;
    }

    std::cout << "\n====================================\n";
    std::cout << ">> AI PREDICTION: \"" << predicted_text << "\"\n";
    std::cout << "====================================\n\n";

  } catch (const Ort::Exception &e) {
    std::cerr << "[ONNX Error] " << e.what() << "\n";
  }
}

// ---------------------------------------------------------
// MAIN APPLICATION
// ---------------------------------------------------------
int main(int argc, char *argv[]) {
  // Default to "wifi" so `./bin/decoder` works out-of-the-box with mock_esp32.py.
  // Use "usb" or "bt" for physical hardware.
  std::string mode = (argc > 1) ? argv[1] : "wifi";

  // --- 1. INITIALIZE AI ENGINE ---
  std::cout << "[System] Booting ONNX Machine Learning Engine...\n";
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SmartPenDecoder");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(
      1); // Run efficiently on a single CPU core

  // We wrap this in a try-catch to get EXACT error messages
  std::unique_ptr<Ort::Session> session;
  try {
    session = std::make_unique<Ort::Session>(env, "models/pen_model.onnx",
                                             session_options);
    std::cout << "[System] AI Model 'pen_model.onnx' loaded successfully.\n";
  } catch (const Ort::Exception &e) {
    // This will print the actual mathematical or version error if it crashes!
    std::cerr << "\n[FATAL ONNX ERROR] " << e.what() << "\n";
  } catch (...) {
    std::cerr << "\n[Warning] 'models/pen_model.onnx' not found!\n";
  }

  // --- 2. INITIALIZE HARDWARE BACKEND ---
  // Port/address defaults live in pen::Defaults so the UI can reference them
  // too — nothing is hardcoded here.
  pen::PenBackend backend;
  if (mode == "usb")
    backend.connectUSB(pen::Defaults::usbPort);
  else if (mode == "bt")
    backend.connectBluetooth(pen::Defaults::bluetoothPort);
  else // "wifi" (default) — matches mock_esp32.py
    backend.connectWiFi(pen::Defaults::wifiPort);

  std::cout << "Mode: " << backend.getStatus() << "\n";
  std::cout << "--------------------------------\n";

  // --- PHYSICAL TUNING PARAMETERS ---
  const float LEVER_ARM_MM = 150.0f;
  const float WAKE_THRESHOLD_Z = 0.5f;
  const float ACTIVITY_THRESHOLD = 0.02f;
  const int IDLE_TIMEOUT_MS = 2000;

  std::vector<DataPoint> strokeBuffer;
  strokeBuffer.reserve(5000);
  pen::IMUData currentData, prevData, anchor;

  while (true) {
    std::cout << "\n[AI IDLE] Waiting for pen impact...\n";

    // 1. WAKE-ON-IMPACT LOOP
    bool isWriting = false;
    while (!isWriting) {
      if (backend.getLatestData(currentData)) {
        if (std::abs(currentData.accel_z - prevData.accel_z) >
            WAKE_THRESHOLD_Z) {
          std::cout << "[AI ACTIVE] Impact detected. Reading stroke...\n";
          anchor = currentData;
          isWriting = true;
        }
        prevData = currentData;
      }
    }

    // 2. CONTINUOUS WRITING LOOP
    strokeBuffer.clear();
    auto startTime = std::chrono::steady_clock::now();
    long long lastActiveTime = 0;

    while (isWriting) {
      if (backend.getLatestData(currentData)) {
        auto now = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             now - startTime)
                             .count();

        float dPitch = std::abs(currentData.pitch - prevData.pitch);
        float dYaw = std::abs(currentData.yaw - prevData.yaw);
        float z_shock = std::abs(currentData.accel_z - prevData.accel_z);

        if (dPitch > ACTIVITY_THRESHOLD || dYaw > ACTIVITY_THRESHOLD ||
            z_shock > WAKE_THRESHOLD_Z) {
          lastActiveTime = elapsedMs;
        }

        float x_mm = -(currentData.yaw - anchor.yaw) * LEVER_ARM_MM;
        float y_mm = (currentData.pitch - anchor.pitch) * LEVER_ARM_MM;
        strokeBuffer.push_back({x_mm, y_mm, currentData.accel_z});

        // 3. IDLE TIMEOUT -> TRIGGER AI INFERENCE!
        if ((elapsedMs - lastActiveTime) > IDLE_TIMEOUT_MS) {
          std::cout << "[AI PROCESSING] Idle timeout reached. Analyzing "
                    << strokeBuffer.size() << " frames...\n";
          isWriting = false;
        }

        prevData = currentData;
      }
    }

    // 4. FEED THE BRAIN
    if (strokeBuffer.size() > 20 && session) {
      runAIInference(*session, strokeBuffer);
    }
  }
  return 0;
}
