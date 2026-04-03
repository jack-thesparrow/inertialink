#include "pen/io.hpp"
#include "pen/viz.hpp"

int main() {
  pen::SerialReader imu("/dev/ttyUSB0");
  pen::Visualizer window(800, 800, "Smart Pen IMU Debugger");

  pen::IMUData currentData;

  while (window.isOpen()) {
    imu.readData(currentData);
    window.drawCube(currentData);
    window.update();
  }
  return 0;
}
