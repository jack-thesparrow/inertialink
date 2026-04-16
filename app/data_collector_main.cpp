#include "pen/io.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// Non-blocking key detection on Linux (for 'd' = discard, ENTER = next)
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

namespace fs = std::filesystem;

// ML Input Tensor Format — 6-DOF raw sensor data
struct DataPoint {
  long long timestamp;
  float ax, ay, az; // Accelerometer (g-force)
  float gx, gy, gz; // Gyroscope (deg/s)
};

// ---------------------------------------------------------
// TERMINAL HELPERS (non-blocking keyboard + raw mode)
// ---------------------------------------------------------
static struct termios savedTermios;
static bool termiosModified = false;

static void enableRawMode() {
  struct termios raw;
  tcgetattr(STDIN_FILENO, &savedTermios);
  raw = savedTermios;
  raw.c_lflag &= ~(ICANON | ECHO); // no line buffering, no echo
  raw.c_cc[VMIN] = 0;
  raw.c_cc[VTIME] = 0;
  tcsetattr(STDIN_FILENO, TCSANOW, &raw);
  termiosModified = true;
}

static void disableRawMode() {
  if (termiosModified) {
    tcsetattr(STDIN_FILENO, TCSANOW, &savedTermios);
    termiosModified = false;
  }
}

// Returns 0 if no key pressed, otherwise the character code
static int readKeyNonBlocking() {
  char c = 0;
  if (read(STDIN_FILENO, &c, 1) == 1)
    return c;
  return 0;
}

// Wait for a specific key, consuming all other input. Returns the key pressed.
// Also monitors pen impact (returns -1 if pen tapped before key).
static int waitForKeyOrImpact(pen::PenBackend &backend, int targetKey) {
  pen::IMUData data;
  float prevMag = 0.0f;
  while (true) {
    int key = readKeyNonBlocking();
    if (key == targetKey || key == '\n' || key == '\r')
      return key;
    if (key == 'q' || key == 'Q' || key == 27) // ESC or q = quit
      return 'q';

    // Also check for pen impact while waiting
    if (backend.getLatestData(data)) {
      float curMag =
          std::sqrt(data.ax * data.ax + data.ay * data.ay + data.az * data.az);
      float shock = std::abs(curMag - prevMag);
      prevMag = curMag;
      if (shock > pen::Defaults::wakeThresholdZ)
        return -1; // pen tapped
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

// ---------------------------------------------------------
// ANSI COLOR HELPERS
// ---------------------------------------------------------
namespace color {
constexpr const char *reset = "\033[0m";
constexpr const char *bold = "\033[1m";
constexpr const char *dim = "\033[2m";
constexpr const char *green = "\033[32m";
constexpr const char *yellow = "\033[33m";
constexpr const char *cyan = "\033[36m";
constexpr const char *red = "\033[31m";
constexpr const char *magenta = "\033[35m";
constexpr const char *bg_green = "\033[42m";
constexpr const char *bg_red = "\033[41m";
} // namespace color

// ---------------------------------------------------------
// CSV EXPORT HELPER
// ---------------------------------------------------------
static std::string saveStrokeToCSV(const std::string &baseDir,
                                   const std::vector<DataPoint> &buffer,
                                   int sampleNum) {
  if (buffer.empty())
    return {};

  std::ostringstream oss;
  oss << baseDir << "/sample_" << std::setw(3) << std::setfill('0')
      << sampleNum << ".csv";
  std::string filename = oss.str();

  std::ofstream outFile(filename);
  outFile << "time_ms,ax,ay,az,gx,gy,gz\n";
  for (const auto &pt : buffer) {
    outFile << pt.timestamp << "," << pt.ax << "," << pt.ay << "," << pt.az
            << "," << pt.gx << "," << pt.gy << "," << pt.gz << "\n";
  }
  outFile.close();
  return filename;
}

// Count existing sample files in a directory
static int countExistingSamples(const std::string &dir) {
  int count = 0;
  if (!fs::exists(dir))
    return 0;
  for (const auto &entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() == ".csv")
      count++;
  }
  return count;
}

// Find the next available sample number
static int findNextSampleNum(const std::string &dir) {
  int maxNum = 0;
  if (!fs::exists(dir))
    return 1;
  for (const auto &entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() == ".csv") {
      std::string stem = entry.path().stem().string();
      // Parse "sample_NNN"
      if (stem.rfind("sample_", 0) == 0) {
        try {
          int num = std::stoi(stem.substr(7));
          if (num > maxNum)
            maxNum = num;
        } catch (...) {
        }
      }
    }
  }
  return maxNum + 1;
}

// ---------------------------------------------------------
// STROKE QUALITY VALIDATION
// ---------------------------------------------------------
struct StrokeQuality {
  bool valid;
  std::string reason;
  int frameCount;
  float durationSec;
  float maxGyroMag;
  float avgGyroMag;
};

static StrokeQuality validateStroke(const std::vector<DataPoint> &buffer) {
  StrokeQuality q;
  q.frameCount = static_cast<int>(buffer.size());
  q.durationSec = buffer.empty()
                      ? 0.0f
                      : (buffer.back().timestamp - buffer.front().timestamp) /
                            1000.0f;
  q.maxGyroMag = 0.0f;
  q.avgGyroMag = 0.0f;

  float gyroSum = 0.0f;
  for (const auto &pt : buffer) {
    float mag =
        std::sqrt(pt.gx * pt.gx + pt.gy * pt.gy + pt.gz * pt.gz);
    if (mag > q.maxGyroMag)
      q.maxGyroMag = mag;
    gyroSum += mag;
  }
  q.avgGyroMag = buffer.empty() ? 0.0f : gyroSum / buffer.size();

  // Validation checks
  if (q.frameCount < 30) {
    q.valid = false;
    q.reason = "Too short (" + std::to_string(q.frameCount) +
               " frames, min 30)";
    return q;
  }
  if (q.durationSec > 10.0f) {
    q.valid = false;
    q.reason = "Too long (" + std::to_string(static_cast<int>(q.durationSec)) +
               "s, max 10s)";
    return q;
  }
  if (q.maxGyroMag < 5.0f) {
    q.valid = false;
    q.reason = "No pen movement detected (max gyro " +
               std::to_string(static_cast<int>(q.maxGyroMag)) + "°/s)";
    return q;
  }

  q.valid = true;
  q.reason = "OK";
  return q;
}

// ---------------------------------------------------------
// MAIN APPLICATION
// ---------------------------------------------------------
int main(int argc, char *argv[]) {
  std::cout.setf(std::ios::unitbuf);

  if (argc < 2) {
    std::cerr << color::bold
              << "Usage: ./data_collector <word> [count] [mode]\n"
              << color::reset
              << "\n"
              << "  word   Target word/character to collect samples for\n"
              << "  count  Number of samples to collect (default: 50)\n"
              << "  mode   Connection mode: usb, wifi, sim (default: usb)\n"
              << "\nExamples:\n"
              << "  ./data_collector hello           # 50 samples via USB\n"
              << "  ./data_collector pen 30 wifi     # 30 samples via WiFi\n"
              << "\nControls during session:\n"
              << "  ENTER  Start next recording / ready prompt\n"
              << "  d      Discard the last saved sample\n"
              << "  q/ESC  Quit and show summary\n";
    return 1;
  }

  std::string label = argv[1];
  int targetCount = (argc > 2) ? std::atoi(argv[2]) : 50;
  std::string mode = (argc > 3) ? argv[3] : "usb";

  if (targetCount <= 0)
    targetCount = 50;

  // Save to data/seed/ for the augmentation pipeline
  std::string baseDir = "data/seed/" + label;
  if (!fs::exists(baseDir))
    fs::create_directories(baseDir);

  // ── Connect to hardware ──────────────────────────────────
  pen::PenBackend backend;
  if (mode == "usb")
    backend.connectUSB(pen::Defaults::usbPort); // auto-detects port
  else if (mode == "bt")
    backend.connectBluetooth(pen::Defaults::btPort);
  else if (mode == "sim")
    backend.connectUSB("/tmp/vtty_laptop");
  else
    backend.connectWiFi(pen::Defaults::wifiPort);

  // Disable input smoothing — we want raw sensor data for ML training
  backend.setSmoothing(1.0f);

  // Count existing samples
  int existingSamples = countExistingSamples(baseDir);
  int nextSampleNum = findNextSampleNum(baseDir);

  // ── Session banner ───────────────────────────────────────
  std::cout << "\n"
            << color::bold << color::cyan
            << "╔══════════════════════════════════════════════╗\n"
            << "║       Smart Pen Data Collector v2.0          ║\n"
            << "╚══════════════════════════════════════════════╝\n"
            << color::reset << "\n"
            << "  Target word : " << color::bold << color::green << label
            << color::reset << "\n"
            << "  Samples     : " << targetCount << " to collect"
            << (existingSamples > 0
                    ? " (" + std::to_string(existingSamples) + " already saved)"
                    : "")
            << "\n"
            << "  Save dir    : " << color::dim << baseDir << "/" << color::reset
            << "\n"
            << "  Connection  : " << backend.getStatus() << "\n"
            << "  Controls    : ENTER=record  SPACE=pause  d=discard  q=quit\n"
            << "\n";

  // ── Setup ────────────────────────────────────────────────
  enableRawMode();
  // Ensure terminal is restored on exit
  std::atexit(disableRawMode);

  constexpr float WAKE_THRESHOLD = pen::Defaults::wakeThresholdZ;
  constexpr float GYRO_ACTIVE_THRESHOLD = 5.0f; // deg/s — direct comparison
  constexpr int IDLE_TIMEOUT_MS = pen::Defaults::idleTimeoutMs;

  std::vector<DataPoint> strokeBuffer;
  strokeBuffer.reserve(5000);

  pen::IMUData currentData, prevData;
  int collectedCount = 0;
  std::string lastSavedFile;
  std::vector<float> sessionDurations; // track durations for summary

  // ── Collection loop ──────────────────────────────────────
  while (collectedCount < targetCount) {
    int sampleIdx = collectedCount + 1;

    // ── Ready prompt ────────────────────────────────────────
    std::cout << color::bold << color::yellow << "\n[" << sampleIdx << "/"
              << targetCount << "]" << color::reset
              << " Write \"" << color::bold << label << color::reset
              << "\" — press " << color::bold << "ENTER" << color::reset
              << " or " << color::bold << "tap pen" << color::reset
              << " when ready";
    if (!lastSavedFile.empty()) {
      std::cout << "  " << color::dim << "(d=discard last)" << color::reset;
    }
    std::cout << "\n";

    // Wait for ENTER, pen tap, or 'd' to discard
    bool readyToRecord = false;
    bool paused = false;
    float prevMag = 0.0f;
    while (!readyToRecord) {
      int key = readKeyNonBlocking();
      
      if (key == ' ') {
        paused = !paused;
        if (paused) {
          std::cout << color::magenta << "  [PAUSED] " << color::reset 
                    << "Adjust pen/paper. Press SPACE again to RESUME.\n";
        } else {
          std::cout << color::magenta << "  [RESUMED] " << color::reset 
                    << "Tap pen or press ENTER to record.\n";
        }
        continue;
      }

      if (paused) {
         // consume IMU data so it doesn't build up, but ignore shocks
         if (backend.getLatestData(currentData)) {
            prevMag = std::sqrt(currentData.ax * currentData.ax +
                                currentData.ay * currentData.ay +
                                currentData.az * currentData.az);
         }
         std::this_thread::sleep_for(std::chrono::milliseconds(5));
         continue;
      }

      if (key == '\n' || key == '\r') {
        readyToRecord = true;
        break;
      }
      if (key == 'q' || key == 'Q' || key == 27) {
        goto session_end;
      }
      if ((key == 'd' || key == 'D') && !lastSavedFile.empty()) {
        // Discard last saved sample
        if (fs::exists(lastSavedFile)) {
          fs::remove(lastSavedFile);
          collectedCount--;
          if (!sessionDurations.empty())
            sessionDurations.pop_back();
          std::cout << color::red << "  ✗ Discarded: " << color::reset
                    << fs::path(lastSavedFile).filename().string() << "\n";
          lastSavedFile.clear();
          nextSampleNum--;
          // Update the prompt
          sampleIdx = collectedCount + 1;
          std::cout << color::bold << color::yellow << "[" << sampleIdx
                    << "/" << targetCount << "]" << color::reset
                    << " Write \"" << color::bold << label << color::reset
                    << "\" — press " << color::bold << "ENTER" << color::reset
                    << " or " << color::bold << "tap pen" << color::reset
                    << " when ready\n";
        }
        continue;
      }

      // Check for pen impact
      if (backend.getLatestData(currentData)) {
        float curMag = std::sqrt(currentData.ax * currentData.ax +
                                 currentData.ay * currentData.ay +
                                 currentData.az * currentData.az);
        float shock = std::abs(curMag - prevMag);
        prevMag = curMag;
        if (shock > WAKE_THRESHOLD) {
          readyToRecord = true;
          break;
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // ── Recording ───────────────────────────────────────────
    std::cout << color::green << "  ● Recording..." << color::reset;

    strokeBuffer.clear();
    auto startTime = std::chrono::steady_clock::now();
    long long lastActiveTime = 0;
    bool isRecording = true;
    int progressTick = 0;

    while (isRecording) {
      if (backend.getLatestData(currentData)) {
        auto now = std::chrono::steady_clock::now();
        auto elapsedMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(now -
                                                                  startTime)
                .count();

        // Activity detection
        float gyroMag = std::sqrt(currentData.gx * currentData.gx +
                                  currentData.gy * currentData.gy +
                                  currentData.gz * currentData.gz);
        float curMag = std::sqrt(currentData.ax * currentData.ax +
                                 currentData.ay * currentData.ay +
                                 currentData.az * currentData.az);
        float pMag = std::sqrt(prevData.ax * prevData.ax +
                               prevData.ay * prevData.ay +
                               prevData.az * prevData.az);
        float shock = std::abs(curMag - pMag);

        if (gyroMag > GYRO_ACTIVE_THRESHOLD || shock > WAKE_THRESHOLD) {
          lastActiveTime = elapsedMs;
        }

        // Record raw sensor data
        strokeBuffer.push_back({elapsedMs, currentData.ax, currentData.ay,
                                currentData.az, currentData.gx, currentData.gy,
                                currentData.gz});

        // Live progress (update every 10 frames to avoid terminal spam)
        progressTick++;
        if (progressTick % 10 == 0) {
          float durSec = elapsedMs / 1000.0f;
          std::cout << "\r" << color::green << "  ● Recording..." << color::reset
                    << "  " << strokeBuffer.size() << " frames ("
                    << std::fixed << std::setprecision(1) << durSec
                    << "s)  gyro: " << static_cast<int>(gyroMag)
                    << "°/s   ";
          std::cout.flush();
        }

        // Idle timeout → stop recording
        if ((elapsedMs - lastActiveTime) > IDLE_TIMEOUT_MS) {
          isRecording = false;
        }

        prevData = currentData;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::cout << "\r                                                        \r";

    // ── Validate stroke quality ────────────────────────────
    auto quality = validateStroke(strokeBuffer);

    if (!quality.valid) {
      std::cout << color::red << "  ✗ REJECTED: " << color::reset
                << quality.reason << "\n";
      continue; // don't count, don't save — retry this sample
    }

    // ── Save ────────────────────────────────────────────────
    lastSavedFile = saveStrokeToCSV(baseDir, strokeBuffer, nextSampleNum);
    collectedCount++;
    sessionDurations.push_back(quality.durationSec);

    std::cout << color::green << "  ✓ Saved " << color::reset
              << color::bold << fs::path(lastSavedFile).filename().string()
              << color::reset << color::dim << "  (" << quality.frameCount
              << " frames, " << std::fixed << std::setprecision(1)
              << quality.durationSec << "s, peak gyro "
              << static_cast<int>(quality.maxGyroMag) << "°/s)"
              << color::reset << "\n";

    nextSampleNum++;
  }

session_end:
  disableRawMode();

  // ── Session summary ──────────────────────────────────────
  int finalSampleCount = countExistingSamples(baseDir);
  float avgDuration = 0.0f;
  if (!sessionDurations.empty()) {
    float sum = 0.0f;
    for (float d : sessionDurations)
      sum += d;
    avgDuration = sum / sessionDurations.size();
  }

  std::cout << "\n"
            << color::bold << color::cyan
            << "╔══════════════════════════════════════════════╗\n"
            << "║              Session Summary                 ║\n"
            << "╚══════════════════════════════════════════════╝\n"
            << color::reset
            << "  Word          : " << color::bold << label << color::reset << "\n"
            << "  Collected     : " << color::bold << color::green
            << collectedCount << color::reset << " samples this session\n"
            << "  Total on disk : " << finalSampleCount << " samples in "
            << baseDir << "/\n";
  if (!sessionDurations.empty()) {
    std::cout << "  Avg duration  : " << std::fixed << std::setprecision(1)
              << avgDuration << "s per stroke\n";
  }
  if (collectedCount >= targetCount) {
    std::cout << color::bold << color::green
              << "  Status        : ✓ Target reached!\n"
              << color::reset;
  } else {
    std::cout << color::yellow
              << "  Status        : Session ended early ("
              << (targetCount - collectedCount) << " remaining)\n"
              << color::reset;
  }
  std::cout << "\n"
            << "Next steps:\n"
            << "  1. Collect more words: ./bin/data_collector <another_word>\n"
            << "  2. Augment data:       python scripts/augment_seed_data.py\n"
            << "  3. Train the model:    python scripts/train_bilstm.py\n"
            << "\n";

  return 0;
}
