#include "pen/io.hpp"
#include <iostream>

int main() {
  pen::SerialReader imu("/dev/ttyUSB0");
  pen::IMUData currentData;

  std::cout << "Starting Headless Smart Pen Decoder...\n";

  while (true) {
    if (imu.readData(currentData)) {
      std::cout << "Pitch: " << currentData.pitch
                << " | Roll: " << currentData.roll
                << " | Yaw: " << currentData.yaw << "      \r" << std::flush;
    }
  }
  return 0;
}
