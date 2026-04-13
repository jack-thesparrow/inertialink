#include "pen/io.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

// ML Input Tensor Format — 6-DOF raw sensor data
struct DataPoint {
  long long timestamp;
  float ax, ay, az;   // Accelerometer (g-force)
  float gx, gy, gz;   // Gyroscope (deg/s)
};

// ---------------------------------------------------------
// CSV EXPORT HELPER
// ---------------------------------------------------------
void saveStrokeToCSV(const std::string &baseDir,
                     const std::vector<DataPoint> &buffer, int &sampleCount) {
  if (buffer.empty())
    return;

  std::string filename;
  while (true) {
    std::ostringstream oss;
    oss << baseDir << "/sample_" << std::setw(3) << std::setfill('0')
        << sampleCount << ".csv";
    filename = oss.str();
    if (!fs::exists(filename))
      break;
    sampleCount++;
  }

  std::ofstream outFile(filename);
  outFile << "time_ms,ax,ay,az,gx,gy,gz\n";
  for (const auto &pt : buffer) {
    outFile << pt.timestamp << ","
            << pt.ax << "," << pt.ay << "," << pt.az << ","
            << pt.gx << "," << pt.gy << "," << pt.gz << "\n";
  }
  outFile.close();
  std::cout << "[SAVED] Captured " << buffer.size() << " data points to "
            << filename << "\n";
}

// ---------------------------------------------------------
// MAIN APPLICATION
// ---------------------------------------------------------
int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: ./data_collector <character_or_word> [mode]\n";
    return -1;
  }

  std::string label = argv[1];
  std::string mode = (argc > 2) ? argv[2] : "usb";
  std::string baseDir = "data/" + label;
  if (!fs::exists(baseDir))
    fs::create_directories(baseDir);

  pen::PenBackend backend;
  if (mode == "usb")
    backend.connectUSB(pen::Defaults::usbPort);
  else if (mode == "bt")
    backend.connectBluetooth(pen::Defaults::btPort);
  else if (mode == "sim")
    backend.connectUSB("/tmp/vtty_laptop");
  else
    backend.connectWiFi(pen::Defaults::wifiPort);

  std::cout << "\n=== Smart Pen Autonomous Collector ===\n";
  std::cout << "Target: " << label << " | Mode: " << backend.getStatus()
            << "\n";

  // Physics constants live in pen::Defaults (io.hpp) so they stay in sync
  // with decoder_main.cpp and the synthetic data generator.
  constexpr float WAKE_THRESHOLD_Z  = pen::Defaults::wakeThresholdZ;
  constexpr float ACTIVITY_THRESHOLD= pen::Defaults::activityThreshold;
  constexpr int   IDLE_TIMEOUT_MS   = pen::Defaults::idleTimeoutMs;

  int sampleCount = 1;
  std::vector<DataPoint> strokeBuffer;
  strokeBuffer.reserve(5000);

  pen::IMUData currentData, prevData;

  while (true) {
    std::cout << "\n[IDLE] Waiting for pen impact on paper...\n";

    // 1. WAKE-ON-IMPACT LOOP — az spike triggers recording
    bool isWriting = false;
    float prevMag = 0.0f;
    while (!isWriting) {
      if (backend.getLatestData(currentData)) {
        // Mounting-independent: total acceleration magnitude spike
        float curMag = std::sqrt(currentData.ax * currentData.ax +
                                 currentData.ay * currentData.ay +
                                 currentData.az * currentData.az);
        float shock  = std::abs(curMag - prevMag);

        if (shock > WAKE_THRESHOLD_Z) {
          std::cout << "[RECORDING] Impact detected! Writing...\n";
          isWriting = true;
        }
        prevMag  = curMag;
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

        // Calculate activity from gyroscope angular velocity magnitude
        float gyroMag = std::sqrt(currentData.gx * currentData.gx +
                                  currentData.gy * currentData.gy +
                                  currentData.gz * currentData.gz);
        // Mounting-independent impact: total acceleration magnitude
        float curMag = std::sqrt(currentData.ax * currentData.ax +
                                 currentData.ay * currentData.ay +
                                 currentData.az * currentData.az);
        float pMag   = std::sqrt(prevData.ax * prevData.ax +
                                 prevData.ay * prevData.ay +
                                 prevData.az * prevData.az);
        float shock  = std::abs(curMag - pMag);

        // If the pen is moving (gyro activity) or tapping, reset the sleep timer
        if (gyroMag > (ACTIVITY_THRESHOLD * 180.0f / M_PI) ||
            shock > WAKE_THRESHOLD_Z) {
          lastActiveTime = elapsedMs;
        }

        // Record raw sensor values — the ML model learns from these directly
        strokeBuffer.push_back({elapsedMs,
                                currentData.ax, currentData.ay, currentData.az,
                                currentData.gx, currentData.gy, currentData.gz});

        // 3. IDLE TIMEOUT LOGIC
        if ((elapsedMs - lastActiveTime) > IDLE_TIMEOUT_MS) {
          std::cout << "[STOP] Pen idle for " << (IDLE_TIMEOUT_MS / 1000)
                    << " seconds. Halting read.\n";
          isWriting = false;
        }

        prevData = currentData;
      }
    }

    // Save only if it wasn't a tiny accidental bump
    if (strokeBuffer.size() > 20) {
      saveStrokeToCSV(baseDir, strokeBuffer, sampleCount);
    }
  }
  return 0;
}
