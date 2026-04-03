#include "pen/io.hpp"
#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <termios.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

// A single row of our CSV
struct DataPoint {
  long long timestamp;
  float pitch;
  float roll;
  float yaw;
};

// ---------------------------------------------------------
// DATA FILTERING & ZERO-ANCHORING
// ---------------------------------------------------------
struct EMAFilter {
  float alpha = 0.15f; // Tune this: 0.1 is smooth, 0.9 is raw
  float filteredPitch = 0.0f;
  float filteredRoll = 0.0f;
  float filteredYaw = 0.0f;
  bool isFirstFrame = true;

  float startPitch = 0.0f;
  float startRoll = 0.0f;
  float startYaw = 0.0f;

  void process(pen::IMUData &rawData) {
    if (isFirstFrame) {
      // Zero-Anchoring: Capture the starting orientation
      startPitch = rawData.pitch;
      startRoll = rawData.roll;
      startYaw = rawData.yaw;

      filteredPitch = 0.0f;
      filteredRoll = 0.0f;
      filteredYaw = 0.0f;
      isFirstFrame = false;
    }

    // Subtract the anchor to make the stroke relative to (0,0,0)
    float relativePitch = rawData.pitch - startPitch;
    float relativeRoll = rawData.roll - startRoll;
    float relativeYaw = rawData.yaw - startYaw;

    // Apply the Exponential Moving Average (EMA) Low-Pass Filter
    filteredPitch = (alpha * relativePitch) + ((1.0f - alpha) * filteredPitch);
    filteredRoll = (alpha * relativeRoll) + ((1.0f - alpha) * filteredRoll);
    filteredYaw = (alpha * relativeYaw) + ((1.0f - alpha) * filteredYaw);

    // Overwrite the raw data with the clean data
    rawData.pitch = filteredPitch;
    rawData.roll = filteredRoll;
    rawData.yaw = filteredYaw;
  }

  void reset() { isFirstFrame = true; }
};

// ---------------------------------------------------------
// LINUX TERMINAL MAGIC: Non-blocking keyboard reads
// ---------------------------------------------------------
bool isSpacebarPressed() {
  struct termios oldt, newt;
  int ch;
  int oldf;

  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO); // Disable buffering and echo
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

  ch = getchar();

  // Restore original terminal settings
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, oldf);

  if (ch != EOF) {
    ungetc(ch, stdin);
    return (ch == ' ');
  }
  return false;
}

// ---------------------------------------------------------
// MAIN APPLICATION
// ---------------------------------------------------------
int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: ./data_collector <character_label>\n";
    std::cerr << "Example: ./data_collector A\n";
    return -1;
  }

  std::string label = argv[1];
  std::string baseDir = "data/" + label;

  // Create the directory if it doesn't exist
  if (!fs::exists(baseDir)) {
    fs::create_directories(baseDir);
  }

  pen::SerialReader imu("/dev/ttyUSB0");
  // pen::SerialReader imu("/tmp/vtty_laptop");
  if (!imu.isOpen()) {
    std::cerr << "Cannot proceed without hardware connection.\n";
    return -1;
  }

  int sampleCount = 1;
  std::vector<DataPoint> strokeBuffer;
  strokeBuffer.reserve(1000);

  EMAFilter strokeFilter;

  std::cout << "\n=== Smart Pen Data Collector ===\n";
  std::cout << "Target Character: '" << label << "'\n";
  std::cout << "Saving to: " << baseDir << "\n";
  std::cout << "--------------------------------\n";

  while (true) {
    std::cout
        << "\n[READY] Press SPACEBAR to START recording (or CTRL+C to quit)...";
    std::cout.flush();

    // 1. Wait for Start Signal
    while (!isSpacebarPressed()) {
      usleep(10000); // 10ms sleep
    }
    getchar(); // Consume the spacebar character

    std::cout << "\n[RECORDING] Move the pen! Press SPACEBAR to STOP...";
    std::cout.flush();

    strokeBuffer.clear();
    strokeFilter.reset(); // CRITICAL: Reset the zero-anchor for the new stroke

    pen::IMUData currentData;
    auto startTime = std::chrono::steady_clock::now();

    // 2. High-Speed Recording Loop
    while (true) {
      // Check for Stop Signal
      if (isSpacebarPressed()) {
        getchar(); // Consume the spacebar
        break;
      }

      // If we have a fresh IMU packet, process it and save it to RAM
      if (imu.readData(currentData)) {

        // --- PASS RAW DATA THROUGH OUR FILTER ---
        strokeFilter.process(currentData);

        auto now = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             now - startTime)
                             .count();

        strokeBuffer.push_back(
            {elapsedMs, currentData.pitch, currentData.roll, currentData.yaw});
      }
    }

    // 3. Save to Disk (After the stroke is done)
    if (strokeBuffer.empty()) {
      std::cout << "\n[WARNING] No data recorded. Try again.\n";
      continue;
    }

    // Auto-increment filename (e.g., sample_001.csv)
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
    outFile << "time_ms,pitch,roll,yaw\n";
    for (const auto &pt : strokeBuffer) {
      outFile << pt.timestamp << "," << pt.pitch << "," << pt.roll << ","
              << pt.yaw << "\n";
    }
    outFile.close();

    std::cout << "\n[SAVED] Captured " << strokeBuffer.size()
              << " data points to " << filename << "\n";
  }

  return 0;
}
