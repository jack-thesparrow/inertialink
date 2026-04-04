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

struct DataPoint {
  long long timestamp;
  float pitch, roll, yaw;
};

// ---------------------------------------------------------
// TERMINAL I/O HELPER
// ---------------------------------------------------------
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
// CSV EXPORT HELPER
// ---------------------------------------------------------
void saveStrokeToCSV(const std::string &baseDir,
                     const std::vector<DataPoint> &buffer, int &sampleCount) {
  if (buffer.empty()) {
    std::cout << "\n[WARNING] No data recorded. Try again.\n";
    return;
  }

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
  for (const auto &pt : buffer) {
    outFile << pt.timestamp << "," << pt.pitch << "," << pt.roll << ","
            << pt.yaw << "\n";
  }
  outFile.close();

  std::cout << "\n[SAVED] Captured " << buffer.size() << " data points to "
            << filename << "\n";
}

// ---------------------------------------------------------
// MAIN APPLICATION
// ---------------------------------------------------------
int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: ./data_collector <character_label> [mode]\n";
    std::cerr << "Modes: sim, usb, bt, wifi\n";
    std::cerr << "Example: ./data_collector A wifi\n";
    return -1;
  }

  std::string label = argv[1];
  std::string mode = (argc > 2) ? argv[2] : "sim";
  std::string baseDir = "data/" + label;

  if (!fs::exists(baseDir)) {
    fs::create_directories(baseDir);
  }

  // 1. Initialize the shared Pen Backend
  pen::PenBackend backend;
  if (mode == "usb")
    backend.connectUSB("/dev/ttyUSB0");
  else if (mode == "bt")
    backend.connectBluetooth("/dev/rfcomm0");
  else if (mode == "wifi")
    backend.connectWiFi(5005);
  else
    backend.connectUSB("/tmp/vtty_laptop");

  std::cout << "\n=== Smart Pen Data Collector ===\n";
  std::cout << "Target Character: '" << label << "'\n";
  std::cout << "Connection:       " << backend.getStatus() << "\n";
  std::cout << "Saving to:        " << baseDir << "\n";
  std::cout << "--------------------------------\n";

  int sampleCount = 1;
  std::vector<DataPoint> strokeBuffer;
  strokeBuffer.reserve(1000);

  while (true) {
    std::cout
        << "\n[READY] Press SPACEBAR to START recording (or CTRL+C to quit)...";
    std::cout.flush();

    while (!isSpacebarPressed()) {
      usleep(10000);
    }
    getchar(); // Consume spacebar

    std::cout << "\n[RECORDING] Move the pen! Press SPACEBAR to STOP...";
    std::cout.flush();

    strokeBuffer.clear();
    pen::IMUData currentData;
    pen::IMUData anchor;
    bool isFirstFrame = true;
    auto startTime = std::chrono::steady_clock::now();

    // 2. High-Speed Recording Loop
    while (true) {
      if (isSpacebarPressed()) {
        getchar();
        break;
      }

      // Backend handles I/O and EMA smoothing automatically!
      if (backend.getLatestData(currentData)) {

        // Zero-Anchoring: Tare the pen to (0,0,0) at the start of the stroke
        if (isFirstFrame) {
          anchor = currentData;
          isFirstFrame = false;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             now - startTime)
                             .count();

        // Save relative, smoothed data
        strokeBuffer.push_back({elapsedMs, currentData.pitch - anchor.pitch,
                                currentData.roll - anchor.roll,
                                currentData.yaw - anchor.yaw});
      }
    }

    // 3. Delegate to CSV helper
    saveStrokeToCSV(baseDir, strokeBuffer, sampleCount);
  }

  return 0;
}
