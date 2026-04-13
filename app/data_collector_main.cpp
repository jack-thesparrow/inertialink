#include "pen/io.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

// ML Input Tensor Format
struct DataPoint {
  long long timestamp;
  float x;       // Lever-Arm Projected X
  float y;       // Lever-Arm Projected Y
  float accel_z; // Raw Z-Axis Impact
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
  outFile << "time_ms,x,y,accel_z\n";
  for (const auto &pt : buffer) {
    outFile << pt.timestamp << "," << pt.x << "," << pt.y << "," << pt.accel_z
            << "\n";
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
  constexpr float LEVER_ARM_MM      = pen::Defaults::leverArmMm;
  constexpr float WAKE_THRESHOLD_Z  = pen::Defaults::wakeThresholdZ;
  constexpr float ACTIVITY_THRESHOLD= pen::Defaults::activityThreshold;
  constexpr int   IDLE_TIMEOUT_MS   = pen::Defaults::idleTimeoutMs;

  int sampleCount = 1;
  std::vector<DataPoint> strokeBuffer;
  strokeBuffer.reserve(5000);

  pen::IMUData currentData, prevData, anchor;

  while (true) {
    std::cout << "\n[IDLE] Waiting for pen impact on paper...\n";

    // 1. WAKE-ON-IMPACT LOOP
    bool isWriting = false;
    while (!isWriting) {
      if (backend.getLatestData(currentData)) {
        // Calculate the Z-axis shockwave
        float z_shock = std::abs(currentData.accel_z - prevData.accel_z);

        if (z_shock > WAKE_THRESHOLD_Z) {
          std::cout << "[RECORDING] Impact detected! Writing...\n";
          anchor = currentData; // Tare the grip angle instantly upon impact
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

        // Calculate deltas to see if the pen is currently moving
        float dPitch = std::abs(currentData.pitch - prevData.pitch);
        float dYaw = std::abs(currentData.yaw - prevData.yaw);
        float z_shock = std::abs(currentData.accel_z - prevData.accel_z);

        // If moving or tapping, reset the sleep timer
        if (dPitch > ACTIVITY_THRESHOLD || dYaw > ACTIVITY_THRESHOLD ||
            z_shock > WAKE_THRESHOLD_Z) {
          lastActiveTime = elapsedMs;
        }

        // Project the relative angles into 2D canvas coordinates
        float relYaw = currentData.yaw - anchor.yaw;
        float relPitch = currentData.pitch - anchor.pitch;
        float x_mm = -relYaw * LEVER_ARM_MM;
        float y_mm = relPitch * LEVER_ARM_MM;

        strokeBuffer.push_back({elapsedMs, x_mm, y_mm, currentData.accel_z});

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
