#include "pen/io.hpp"
#include <chrono>
#include <cmath>
#include <fcntl.h>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <termios.h>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------
// HELPER STRUCTS
// ---------------------------------------------------------
struct DataPoint {
  float pitch, roll, yaw;
};

struct EMAFilter {
  float alpha = 0.15f;
  float fPitch = 0.0f, fRoll = 0.0f, fYaw = 0.0f;
  float sPitch = 0.0f, sRoll = 0.0f, sYaw = 0.0f;
  bool isFirstFrame = true;

  void process(pen::IMUData &raw) {
    if (isFirstFrame) {
      sPitch = raw.pitch;
      sRoll = raw.roll;
      sYaw = raw.yaw;
      fPitch = 0;
      fRoll = 0;
      fYaw = 0;
      isFirstFrame = false;
    }
    fPitch = (alpha * (raw.pitch - sPitch)) + ((1.0f - alpha) * fPitch);
    fRoll = (alpha * (raw.roll - sRoll)) + ((1.0f - alpha) * fRoll);
    fYaw = (alpha * (raw.yaw - sYaw)) + ((1.0f - alpha) * fYaw);

    raw.pitch = fPitch;
    raw.roll = fRoll;
    raw.yaw = fYaw;
  }
  void reset() { isFirstFrame = true; }
};

// Linux Terminal non-blocking keypress
bool isSpacebarPressed() {
  struct termios oldt, newt;
  int ch;
  int oldf;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
  ch = getchar();
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, oldf);
  if (ch != EOF) {
    ungetc(ch, stdin);
    return (ch == ' ');
  }
  return false;
}

// ---------------------------------------------------------
// C++ 1D INTERPOLATION (Stretches/Squishes stroke to 100 frames)
// ---------------------------------------------------------
std::vector<float> resampleStroke(const std::vector<DataPoint> &raw_stroke,
                                  int target_len = 100) {
  std::vector<float> flat_tensor(target_len * 3, 0.0f);
  if (raw_stroke.empty())
    return flat_tensor;

  int n = raw_stroke.size();
  for (int i = 0; i < target_len; ++i) {
    // Find where this point maps onto the original array
    float original_idx = (float)i * (n - 1) / (target_len - 1);
    int idx1 = std::floor(original_idx);
    int idx2 = std::min((int)std::ceil(original_idx), n - 1);
    float weight = original_idx - idx1;

    // Linear interpolation for Pitch, Roll, Yaw
    float p = raw_stroke[idx1].pitch +
              weight * (raw_stroke[idx2].pitch - raw_stroke[idx1].pitch);
    float r = raw_stroke[idx1].roll +
              weight * (raw_stroke[idx2].roll - raw_stroke[idx1].roll);
    float y = raw_stroke[idx1].yaw +
              weight * (raw_stroke[idx2].yaw - raw_stroke[idx1].yaw);

    // Flatten into [100 * 3] 1D array for ONNX
    flat_tensor[(i * 3) + 0] = p;
    flat_tensor[(i * 3) + 1] = r;
    flat_tensor[(i * 3) + 2] = y;
  }
  return flat_tensor;
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main() {
  std::cout << "\n=== Smart Pen ML Decoder Booting ===\n";

  // 1. Initialize ONNX Runtime
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SmartPen");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1); // Optimize for single CPU core

  const char *model_path = "data/smart_pen.onnx";
  Ort::Session session(env, model_path, session_options);
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::cout << "[+] ONNX Neural Network loaded: " << model_path << "\n";

  // 2. Connect to Hardware (Using your virtual simulator for now)
  pen::SerialReader imu("/tmp/vtty_laptop");
  if (!imu.isOpen()) {
    std::cerr << "[-] Error: Simulator cable not found. Start socat first!\n";
    return -1;
  }

  std::vector<DataPoint> strokeBuffer;
  strokeBuffer.reserve(1000);
  EMAFilter filter;

  // Define our two possible outputs (Since we trained on A and B)
  std::vector<std::string> classes = {"A", "B"};

  while (true) {
    std::cout << "\n[READY] Press SPACEBAR to start writing...";
    std::cout.flush();
    while (!isSpacebarPressed()) {
      usleep(10000);
    }
    getchar();

    std::cout << "\n[DRAWING] Move the pen... Press SPACEBAR to finish.";
    std::cout.flush();

    strokeBuffer.clear();
    filter.reset();
    pen::IMUData currentData;

    // High-Speed Recording Loop
    while (true) {
      if (isSpacebarPressed()) {
        getchar();
        break;
      }
      if (imu.readData(currentData)) {
        filter.process(currentData);
        strokeBuffer.push_back(
            {currentData.pitch, currentData.roll, currentData.yaw});
      }
    }

    if (strokeBuffer.empty())
      continue;

    // 3. Process the Data (Interpolate to 100 frames)
    std::vector<float> input_tensor_values = resampleStroke(strokeBuffer, 100);

    // 4. Run ONNX Inference
    std::vector<int64_t> input_shape = {
        1, 100, 3}; // Batch: 1, Frames: 100, Features: 3

    // These strings must exactly match the names generated by Keras/tf2onnx
    const char *input_names[] = {"input_layer"};
    const char *output_names[] = {"classification"};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names,
                                      &input_tensor, 1, output_names, 1);

    // 5. Decode the Result
    float *floatarr = output_tensors.front().GetTensorMutableData<float>();

    // Find the index with the highest probability (argmax)
    int best_match_idx = (floatarr[0] > floatarr[1]) ? 0 : 1;
    float confidence = floatarr[best_match_idx] * 100.0f;

    std::cout << "\n----------------------------------------";
    std::cout << "\n[PREDICTION]  =>  " << classes[best_match_idx]
              << "  (Confidence: " << confidence << "%)";
    std::cout << "\n----------------------------------------\n";
  }

  return 0;
}
