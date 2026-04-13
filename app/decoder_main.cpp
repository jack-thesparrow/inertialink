#include "pen/io.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>
#include <vector>

// Write a short string to /tmp/inertialink_mode so the visualizer can show it.
static void writeMode(const char *mode) {
  if (std::FILE *f = std::fopen("/tmp/inertialink_mode", "w")) {
    std::fputs(mode, f);
    std::fclose(f);
  }
}

// ML Input Format — 6-DOF raw sensor data
static constexpr int NUM_FEATURES = 6;

struct DataPoint {
  float ax, ay, az;   // Accelerometer (g-force)
  float gx, gy, gz;   // Gyroscope (deg/s)
};

// Must match train_bilstm.py ALPHABET exactly.
// Index 0 ('~') is the CTC blank — always skipped by the decoder.
// Real characters start at index 1.
static const std::string ALPHABET =
    "~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
static constexpr const char *MODEL_PATH = "models/pen_model.onnx";

// Words the model was trained on — must match mock_esp32.py TRAINED_WORDS.
// After beam-search decoding, the raw prediction is snapped to the nearest
// vocabulary entry (Levenshtein ≤ 2) so that partial drops like "nte"→"note"
// or "wrld"→"world" are automatically corrected.
static const std::vector<std::string> VOCABULARY = {
    "hello", "world", "pen", "123", "write",
    "note",  "data",  "code","test", "abc", "xyz", "open"};

// ---------------------------------------------------------
// HELPERS
// ---------------------------------------------------------
static int editDistance(const std::string &a, const std::string &b) {
  int m = static_cast<int>(a.size()), n = static_cast<int>(b.size());
  std::vector<int> prev(n + 1), curr(n + 1);
  for (int j = 0; j <= n; ++j) prev[j] = j;
  for (int i = 1; i <= m; ++i) {
    curr[0] = i;
    for (int j = 1; j <= n; ++j)
      curr[j] = (a[i-1] == b[j-1]) ? prev[j-1]
                : 1 + std::min({prev[j], curr[j-1], prev[j-1]});
    std::swap(prev, curr);
  }
  return prev[n];
}

// Snap raw CTC output to the nearest vocabulary word if within edit distance 2.
// Returns {corrected_word, was_corrected}.
static std::pair<std::string, bool> snapToVocab(const std::string &raw) {
  if (raw.empty()) return {raw, false};
  int   best_dist = 3; // only snap if dist <= 2
  std::string best = raw;
  for (const auto &w : VOCABULARY) {
    int d = editDistance(raw, w);
    if (d < best_dist) { best_dist = d; best = w; }
  }
  return {best, best != raw};
}

static std::vector<float> softmax(const float *logits, int64_t n) {
  float max_val = *std::max_element(logits, logits + n);
  std::vector<float> probs(n);
  float sum = 0.0f;
  for (int64_t i = 0; i < n; ++i) { probs[i] = std::exp(logits[i] - max_val); sum += probs[i]; }
  for (auto &p : probs) p /= sum;
  return probs;
}

// Numerically stable log(exp(a) + exp(b)).
static constexpr float LOG_ZERO = -1e30f;
static float log_sum_exp(float a, float b) {
  if (a <= LOG_ZERO) return b;
  if (b <= LOG_ZERO) return a;
  float hi = std::max(a, b), lo = std::min(a, b);
  return hi + std::log1p(std::exp(lo - hi));
}

// ---------------------------------------------------------
// CTC BEAM SEARCH
// ---------------------------------------------------------
struct Beam {
  float log_pb  = LOG_ZERO;
  float log_pnb = LOG_ZERO;
  float log_total() const { return log_sum_exp(log_pb, log_pnb); }
};

static constexpr int BEAM_WIDTH = 10;

static std::string ctc_beam_search(
    const float *logits_ptr, int64_t seq_len, int64_t num_classes,
    std::vector<std::pair<char, float>> &per_char_conf)
{
  std::unordered_map<std::string, Beam> beams;
  beams[""].log_pb = 0.0f;

  for (int64_t t = 0; t < seq_len; ++t) {
    const float *frame = logits_ptr + t * num_classes;
    std::vector<float> lp(num_classes);
    {
      auto probs = softmax(frame, num_classes);
      for (int c = 0; c < num_classes; ++c)
        lp[c] = (probs[c] > 1e-30f) ? std::log(probs[c]) : LOG_ZERO;
    }

    std::unordered_map<std::string, Beam> next;
    next.reserve(beams.size() * 4);

    for (auto &[prefix, bm] : beams) {
      char last = prefix.empty() ? '\0' : prefix.back();

      next[prefix].log_pb =
          log_sum_exp(next[prefix].log_pb,
                      log_sum_exp(bm.log_pb, bm.log_pnb) + lp[0]);

      for (int c = 1; c < static_cast<int>(num_classes); ++c) {
        char ch = ALPHABET[c];
        if (ch == last) {
          std::string ext = prefix + ch;
          next[ext].log_pnb =
              log_sum_exp(next[ext].log_pnb, bm.log_pb + lp[c]);
          next[prefix].log_pnb =
              log_sum_exp(next[prefix].log_pnb, bm.log_pnb + lp[c]);
        } else {
          std::string ext = prefix + ch;
          next[ext].log_pnb =
              log_sum_exp(next[ext].log_pnb,
                          log_sum_exp(bm.log_pb, bm.log_pnb) + lp[c]);
        }
      }
    }

    if (static_cast<int>(next.size()) > BEAM_WIDTH) {
      std::vector<std::pair<float, std::string>> scores;
      scores.reserve(next.size());
      for (auto &[p, b] : next) scores.emplace_back(b.log_total(), p);
      std::partial_sort(scores.begin(), scores.begin() + BEAM_WIDTH,
                        scores.end(), std::greater<>{});
      std::unordered_map<std::string, Beam> pruned;
      pruned.reserve(BEAM_WIDTH);
      for (int i = 0; i < BEAM_WIDTH; ++i) pruned[scores[i].second] = next[scores[i].second];
      beams = std::move(pruned);
    } else {
      beams = std::move(next);
    }
  }

  std::string best_text;
  float best_score = LOG_ZERO;
  for (auto &[p, b] : beams) {
    if (b.log_total() > best_score) { best_score = b.log_total(); best_text = p; }
  }

  for (char ch : best_text) {
    auto pos = ALPHABET.find(ch);
    if (pos == std::string::npos) continue;
    float peak = 0.0f;
    for (int64_t t = 0; t < seq_len; ++t) {
      auto probs = softmax(logits_ptr + t * num_classes, num_classes);
      peak = std::max(peak, probs[static_cast<int>(pos)]);
    }
    per_char_conf.push_back({ch, peak});
  }

  return best_text;
}

// ---------------------------------------------------------
// ONNX INFERENCE & CTC DECODER
// ---------------------------------------------------------
void runAIInference(Ort::Session &session,
                    const std::vector<DataPoint> &strokeBuffer) {
  if (strokeBuffer.empty())
    return;

  // 1. Flatten stroke into [1, seq_len, NUM_FEATURES] float array
  std::vector<float> input_tensor_values;
  input_tensor_values.reserve(strokeBuffer.size() * NUM_FEATURES);
  for (const auto &pt : strokeBuffer) {
    input_tensor_values.push_back(pt.ax);
    input_tensor_values.push_back(pt.ay);
    input_tensor_values.push_back(pt.az);
    input_tensor_values.push_back(pt.gx);
    input_tensor_values.push_back(pt.gy);
    input_tensor_values.push_back(pt.gz);
  }

  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> input_shape = {
      1, static_cast<int64_t>(strokeBuffer.size()), NUM_FEATURES};

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_tensor_values.data(), input_tensor_values.size(),
      input_shape.data(), input_shape.size());

  const char *input_names[]  = {"input_stroke"};
  const char *output_names[] = {"predicted_logits"};

  try {
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names,
                                      &input_tensor, 1, output_names, 1);

    float *logits_ptr   = output_tensors.front().GetTensorMutableData<float>();
    auto   output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
    int64_t seq_len     = output_shape[1];
    int64_t num_classes = output_shape[2];

    std::vector<std::pair<char, float>> char_confs;
    std::string predicted_text =
        ctc_beam_search(logits_ptr, seq_len, num_classes, char_confs);

    float overall_confidence = 0.0f;
    if (!char_confs.empty()) {
      for (auto &[ch, p] : char_confs) overall_confidence += p;
      overall_confidence = overall_confidence / char_confs.size() * 100.0f;
    }

    std::string per_char;
    for (auto &[ch, p] : char_confs) {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "%c=%d%%  ", ch, static_cast<int>(p * 100.0f));
      per_char += buf;
    }

    auto [corrected, was_corrected] = snapToVocab(predicted_text);

    std::cout << "\n====================================\n";
    if (corrected.empty()) {
      std::cout << ">> PREDICTION : (nothing recognised)\n";
    } else {
      std::cout << ">> PREDICTION : \"" << corrected << "\"\n";
      if (was_corrected)
        std::cout << ">> RAW CTC    : \"" << predicted_text << "\" (snapped to vocab)\n";
      std::cout << ">> CONFIDENCE : " << static_cast<int>(overall_confidence)
                << "%\n";
      std::cout << ">> PER CHAR   : " << per_char << "\n";
    }
    std::cout << "====================================\n\n";

  } catch (const Ort::Exception &e) {
    std::cerr << "[ONNX Error] " << e.what() << "\n";
  }
}

// ---------------------------------------------------------
// MAIN APPLICATION
// ---------------------------------------------------------
int main(int argc, char *argv[]) {
  std::cout.setf(std::ios::unitbuf);

  std::string mode = (argc > 1) ? argv[1] : "usb";

  writeMode("idle");

  // --- 1. INITIALIZE AI ENGINE ---
  std::cout << "[System] Booting ONNX Machine Learning Engine...\n";
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SmartPenDecoder");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

  std::unique_ptr<Ort::Session> session;
  try {
    session = std::make_unique<Ort::Session>(env, MODEL_PATH, session_options);
    std::cout << "[System] AI Model '" << MODEL_PATH << "' loaded successfully.\n";
  } catch (const Ort::Exception &e) {
    std::cerr << "\n[FATAL ONNX ERROR] " << e.what() << "\n";
  } catch (...) {
    std::cerr << "\n[FATAL] '" << MODEL_PATH << "' not found — run train_bilstm.py first.\n";
  }

  if (!session) {
    std::cerr << "[FATAL] Cannot run without a model. Exiting.\n";
    return 1;
  }

  // --- 2. INITIALIZE HARDWARE BACKEND ---
  pen::PenBackend backend;
  if (mode == "usb")
    backend.connectUSB(pen::Defaults::usbPort);
  else if (mode == "bt")
    backend.connectBluetooth(pen::Defaults::btPort);
  else if (mode == "sim")
    backend.connectUSB("/tmp/vtty_laptop");
  else // "wifi"
    backend.connectWiFi(pen::Defaults::wifiPort);

  // Alpha=1.0 bypasses the low-pass filter (passthrough).
  backend.setSmoothing(1.0f);

  std::cout << "Mode: " << backend.getStatus() << "\n";
  std::cout << "--------------------------------\n";

  constexpr float WAKE_THRESHOLD_Z   = pen::Defaults::wakeThresholdZ;
  constexpr float ACTIVITY_THRESHOLD = pen::Defaults::activityThreshold;
  constexpr int   IDLE_TIMEOUT_MS    = pen::Defaults::idleTimeoutMs;

  std::vector<DataPoint> strokeBuffer;
  strokeBuffer.reserve(5000);
  pen::IMUData currentData, prevData;

  while (true) {
    std::cout << "\n[AI IDLE] Waiting for pen impact...\n";
    writeMode("idle");

    // 1. WAKE-ON-IMPACT LOOP — az spike triggers recording
    bool isWriting = false;
    while (!isWriting) {
      if (backend.getLatestData(currentData)) {
        if (std::abs(currentData.az - prevData.az) >
            WAKE_THRESHOLD_Z) {
          std::cout << "[AI ACTIVE] Impact detected. Reading stroke...\n";
          writeMode("Reading stroke...");
          isWriting = true;
        }
        prevData = currentData;
      }
    }

    // 2. CONTINUOUS WRITING LOOP — record raw 6-DOF sensor data
    strokeBuffer.clear();
    auto startTime = std::chrono::steady_clock::now();
    long long lastActiveTime = 0;

    while (isWriting) {
      if (backend.getLatestData(currentData)) {
        auto now = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             now - startTime)
                             .count();

        // Activity from gyro magnitude + az shock
        float gyroMag = std::sqrt(currentData.gx * currentData.gx +
                                  currentData.gy * currentData.gy +
                                  currentData.gz * currentData.gz);
        float z_shock = std::abs(currentData.az - prevData.az);

        if (gyroMag > (ACTIVITY_THRESHOLD * 180.0f / M_PI) ||
            z_shock > WAKE_THRESHOLD_Z) {
          lastActiveTime = elapsedMs;
        }

        // Record raw sensor values — the ML model learns from these directly
        strokeBuffer.push_back({currentData.ax, currentData.ay, currentData.az,
                                currentData.gx, currentData.gy, currentData.gz});

        // 3. IDLE TIMEOUT -> TRIGGER AI INFERENCE!
        if ((elapsedMs - lastActiveTime) > IDLE_TIMEOUT_MS) {
          std::cout << "[AI PROCESSING] Idle timeout reached. Analyzing "
                    << strokeBuffer.size() << " frames...\n";
          writeMode("Predicting...");
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
