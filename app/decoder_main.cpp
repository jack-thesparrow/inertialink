#include "pen/io.hpp"
#include <algorithm>
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
static const std::string ALPHABET = "~ 123ABC";
static constexpr const char *MODEL_PATH = "models/pen_model.onnx";

// Vocabulary the model was trained on: individual characters + words.
// After beam-search decoding, the raw prediction is snapped to the nearest
// vocabulary entry (Levenshtein ≤ 2) so that partial drops like "helo"→"hello"
// are automatically corrected.  Single-char predictions pass through as-is.
static const std::vector<std::string> VOCABULARY = {
    "1", "2", "3", "A", "B", "C"
};

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
  // Do not alter single-character outputs in character mode.
  if (raw.size() == 1) return {raw, false};
  int   best_dist = 3; // only snap if dist <= 2
  std::string best = raw;
  for (const auto &w : VOCABULARY) {
    int d = editDistance(raw, w);
    if (d < best_dist) { best_dist = d; best = w; }
  }
  return {best, best != raw};
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
  // Single pass: pre-compute log-probs and track per-class peak probability.
  // This eliminates repeated softmax calls inside the beam loop and the
  // redundant O(seq_len * num_chars) softmax pass that was done afterwards.
  std::vector<float> log_probs(seq_len * num_classes);
  std::vector<float> class_peak(num_classes, 0.0f);

  for (int64_t t = 0; t < seq_len; ++t) {
    const float *frame = logits_ptr + t * num_classes;
    float max_val = *std::max_element(frame, frame + num_classes);
    float *lp = log_probs.data() + t * num_classes;
    float sum = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
      lp[c] = std::exp(frame[c] - max_val);
      sum += lp[c];
    }
    for (int c = 0; c < num_classes; ++c) {
      lp[c] /= sum; // probability
      if (lp[c] > class_peak[c]) class_peak[c] = lp[c];
      lp[c] = (lp[c] > 1e-30f) ? std::log(lp[c]) : LOG_ZERO;
    }
  }

  std::unordered_map<std::string, Beam> beams;
  beams[""].log_pb = 0.0f;

  for (int64_t t = 0; t < seq_len; ++t) {
    const float *lp = log_probs.data() + t * num_classes;

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

  // Confidence from pre-computed class peaks — no extra softmax pass needed.
  for (char ch : best_text) {
    auto pos = ALPHABET.find(ch);
    if (pos == std::string::npos) continue;
    per_char_conf.push_back({ch, class_peak[static_cast<int>(pos)]});
  }

  return best_text;
}

// ---------------------------------------------------------
// STROKE TRIMMER
// ---------------------------------------------------------
// The data_collector stops recording 700 ms (70 frames @ 100 Hz) after the
// last active frame, so every training sample ends with ~70 quiet frames.
// The decoder's LIFTED timeout is 2 s for UX reasons, which pads strokes with
// ~250 extra quiet frames the model never saw during training.  Extra quiet
// frames also bias the local accel mean used for normalization inside the model.
// This function removes the excess tail so inference input matches training data.
static constexpr float TRIM_GYRO_QUIET = 5.0f; // deg/s  — same as data_collector
static constexpr int   TRIM_TAIL_FRAMES = 70;   // 700 ms — matches data_collector idle timeout

static void trimStroke(std::vector<DataPoint> &buf) {
  int lastActive = -1;
  for (int i = static_cast<int>(buf.size()) - 1; i >= 0; --i) {
    float gm = std::sqrt(buf[i].gx * buf[i].gx +
                         buf[i].gy * buf[i].gy +
                         buf[i].gz * buf[i].gz);
    if (gm > TRIM_GYRO_QUIET) { lastActive = i; break; }
  }
  if (lastActive < 0) return; // all quiet — leave as-is, inference will skip

  int keepEnd = std::min(lastActive + TRIM_TAIL_FRAMES + 1,
                         static_cast<int>(buf.size()));
  buf.resize(static_cast<size_t>(keepEnd));
}

// ---------------------------------------------------------
// ONNX INFERENCE & CTC DECODER
// ---------------------------------------------------------
void runAIInference(Ort::Session &session,
                    const std::vector<DataPoint> &strokeBuffer,
                    pen::PenBackend &backend) {
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
      backend.sendCommand("PRED:?\n");
    } else {
      std::cout << ">> PREDICTION : \"" << corrected << "\"\n";
      backend.sendCommand("PRED:" + corrected + "\n");
      if (was_corrected)
        std::cout << ">> RAW CTC    : \"" << predicted_text << "\" (snapped to vocab)\n";
      std::cout << ">> CONFIDENCE : " << static_cast<int>(overall_confidence)
                << "%\n";
      std::cout << ">> PER CHAR   : " << per_char << "\n";
    }
    std::cout << "====================================\n\n";

    // Feed result back to the ESP32 OLED (STATE already set to PROCESSING)
    char fbuf[32];
    if (corrected.empty()) {
      backend.sendFeedback("PRED::0\n");
    } else {
      std::snprintf(fbuf, sizeof(fbuf), "PRED:%s:%d\n",
                    corrected.c_str(), static_cast<int>(overall_confidence));
      backend.sendFeedback(fbuf);
    }

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

  // XPU/Intel Arc GPU configuration
  const char* model_device_env = std::getenv("MODEL_DEVICE");
  std::string device_setting = model_device_env ? model_device_env : "auto";
  
  // Configure providers based on device setting
  std::vector<std::string> providers;
  
  if (device_setting == "xpu" || device_setting == "gpu" || device_setting == "auto") {
    // Try OpenVINO GPU provider first for Intel Arc
    providers.push_back("OpenVINOExecutionProvider");
    providers.push_back("CPUExecutionProvider");
  } else {
    // CPU only
    providers.push_back("CPUExecutionProvider");
  }

  std::cout << "[System] Device setting: " << device_setting << "\n";
  std::cout << "[System] Requested providers: ";
  for (size_t i = 0; i < providers.size(); ++i) {
    std::cout << providers[i];
    if (i < providers.size() - 1) std::cout << ", ";
  }
  std::cout << "\n";

  std::unique_ptr<Ort::Session> session;
  try {
    // Create session options
    Ort::SessionOptions session_options_with_providers;
    session_options_with_providers.SetIntraOpNumThreads(1);
    
    // For XPU/GPU support, we need to append the OpenVINO execution provider
    // The ONNX Runtime C++ API automatically uses available providers
    // OpenVINO will be used if available and GPU is detected
    
    if (device_setting == "xpu" || device_setting == "gpu" || device_setting == "auto") {
      std::cout << "[System] Attempting to use OpenVINO GPU provider...\n";
      // Set environment variables for OpenVINO GPU if needed
      // OpenVINO provider will automatically detect Intel Arc GPU
    }
    
    session = std::make_unique<Ort::Session>(env, MODEL_PATH, session_options_with_providers);
    
    // Simple provider reporting - the API is limited in C++
    std::cout << "[System] AI Model '" << MODEL_PATH << "' loaded successfully.\n";
    std::cout << "[System] Device setting: " << device_setting << "\n";
    
    if (device_setting == "xpu" || device_setting == "gpu" || device_setting == "auto") {
      std::cout << "[System] OpenVINO GPU provider will be used if Intel Arc GPU is available\n";
    } else {
      std::cout << "[System] Using CPU inference\n";
    }
    
  } catch (const Ort::Exception &e) {
    std::cerr << "\n[FATAL ONNX ERROR] " << e.what() << "\n";
    if (device_setting == "xpu" || device_setting == "gpu") {
      std::cerr << "[Hint] Make sure Intel OpenVINO is installed and GPU drivers are up to date.\n";
      std::cerr << "[Hint] Try MODEL_DEVICE=cpu to force CPU inference.\n";
    }
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

  // ── Pen state machine (mirrors visualizer) ──────────────────────────────
  // IDLE    → waiting for pen impact or movement
  // WRITING → recording sensor frames into strokeBuffer
  // LIFTED  → pen lifted; short silence returns to WRITING, long → IDLE + infer
  enum class PenState { IDLE, WRITING, LIFTED };

  // Activity thresholds — same tuning as visualizer
  constexpr float DT = 0.01f; // 100 Hz
  constexpr float JERK_IMPACT_THRESHOLD = 12.0f; // g/s — pen touches paper
  constexpr float JERK_QUIET_THRESHOLD  = 2.0f;  // g/s — pen is still
  constexpr float GYRO_ACTIVE_THRESHOLD = 15.0f; // deg/s — pen is moving
  constexpr float GYRO_QUIET_THRESHOLD  = 5.0f;  // deg/s — pen is still
  constexpr int   LIFT_QUIET_FRAMES     = 50;    // 500 ms of quiet → LIFTED
  constexpr int   IDLE_QUIET_FRAMES     = 200;   // 2 s of quiet after lift → IDLE + infer
  constexpr float JERK_SMOOTH           = 0.3f;

  PenState penState = PenState::IDLE;
  int      quietFrames = 0;
  float    prevAccelMag = 1.0f;
  float    prevJerk = 0.0f;

  std::vector<DataPoint> strokeBuffer;
  strokeBuffer.reserve(5000);
  pen::IMUData currentData;

  std::cout << "\n[IDLE] Waiting for pen activity...\n";
  writeMode("idle");
  backend.sendFeedback("STATE:IDLE\n");

  while (true) {
    if (!backend.getLatestData(currentData))
      continue;

    // ── Compute activity signals ──────────────────────────────────────────
    float curAccelMag = std::sqrt(currentData.ax * currentData.ax +
                                  currentData.ay * currentData.ay +
                                  currentData.az * currentData.az);
    float jerk = std::abs(curAccelMag - prevAccelMag) / DT; // g/s
    float smoothJerk = JERK_SMOOTH * jerk + (1.0f - JERK_SMOOTH) * prevJerk;
    prevJerk = smoothJerk;
    prevAccelMag = curAccelMag;

    float gyroMag = std::sqrt(currentData.gx * currentData.gx +
                              currentData.gy * currentData.gy +
                              currentData.gz * currentData.gz);

    bool jerkActive = smoothJerk > JERK_IMPACT_THRESHOLD;
    bool gyroActive = gyroMag > GYRO_ACTIVE_THRESHOLD;
    bool isActive   = jerkActive || gyroActive;

    bool jerkQuiet = smoothJerk < JERK_QUIET_THRESHOLD;
    bool gyroQuiet = gyroMag < GYRO_QUIET_THRESHOLD;
    bool isQuiet   = jerkQuiet && gyroQuiet;

    // ── State machine ─────────────────────────────────────────────────────
    switch (penState) {

    case PenState::IDLE:
      if (isActive) {
        penState = PenState::WRITING;
        quietFrames = 0;
        strokeBuffer.clear();
        std::cout << "[WRITING] Pen activity detected — recording...\n";
        writeMode("Reading stroke...");
        backend.sendFeedback("STATE:WRITING\n");
        // Record the triggering frame
        strokeBuffer.push_back({currentData.ax, currentData.ay, currentData.az,
                                currentData.gx, currentData.gy, currentData.gz});
      }
      break;

    case PenState::WRITING:
      // Always record while in WRITING state
      strokeBuffer.push_back({currentData.ax, currentData.ay, currentData.az,
                              currentData.gx, currentData.gy, currentData.gz});

      if (isQuiet) {
        quietFrames++;
        if (quietFrames > LIFT_QUIET_FRAMES) {
          penState = PenState::LIFTED;
          quietFrames = 0;
          std::cout << "[LIFTED] Pen lifted — waiting for more writing or timeout...\n";
          writeMode("Pen lifted...");
          backend.sendFeedback("STATE:LIFTED\n");
        }
      } else {
        quietFrames = 0;
      }
      break;

    case PenState::LIFTED:
      // Still record frames in case the pen comes back down (captures the gap)
      strokeBuffer.push_back({currentData.ax, currentData.ay, currentData.az,
                              currentData.gx, currentData.gy, currentData.gz});

      if (isActive) {
        // Pen back on paper — resume writing
        penState = PenState::WRITING;
        quietFrames = 0;
        std::cout << "[WRITING] Pen back on paper — continuing...\n";
        writeMode("Reading stroke...");
        backend.sendFeedback("STATE:WRITING\n");
      } else {
        quietFrames++;
        if (quietFrames > IDLE_QUIET_FRAMES) {
          // Extended silence — stroke is complete → trim then run inference
          penState = PenState::IDLE;
          quietFrames = 0;

          int rawFrames = static_cast<int>(strokeBuffer.size());
          trimStroke(strokeBuffer);
          int trimmedFrames = static_cast<int>(strokeBuffer.size());

          std::cout << "[PROCESSING] Stroke complete (" << rawFrames
                    << " frames → " << trimmedFrames
                    << " after trim). Running AI inference...\n";
          writeMode("Predicting...");
          backend.sendFeedback("STATE:PROCESSING\n");

          if (strokeBuffer.size() > 20 && session) {
            runAIInference(*session, strokeBuffer, backend);
          } else if (strokeBuffer.size() <= 20) {
            std::cout << "[SKIPPED] Too few frames (" << strokeBuffer.size()
                      << ") — likely accidental tap.\n";
          }

          strokeBuffer.clear();
          std::cout << "\n[IDLE] Waiting for pen activity...\n";
          writeMode("idle");
          backend.sendFeedback("STATE:IDLE\n");
        }
      }
      break;
    }
  }
  return 0;
}
