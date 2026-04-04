#include "pen/io.hpp"
#include "pen/viz.hpp"
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  // 1. Initialize our new dynamic backend!
  pen::PenBackend backend;

  // 2. Parse command line arguments (Defaults to "sim" if nothing is typed)
  std::string mode = (argc > 1) ? argv[1] : "sim";

  if (mode == "usb") {
    backend.connectUSB("/dev/ttyUSB0");
  } else if (mode == "bt") {
    backend.connectBluetooth("/dev/rfcomm0");
  } else if (mode == "wifi") {
    backend.connectWiFi(5005);
  } else {
    std::cout << "[Visualizer] Defaulting to Simulator mode...\n";
    // Connects to the virtual cable created by socat!
    backend.connectUSB("/tmp/vtty_laptop");
  }

  std::cout << "[Visualizer] Status: " << backend.getStatus() << "\n";

  // 3. Initialize the OpenGL Window
  pen::Visualizer window(800, 800, ("Smart Pen IMU - " + mode).c_str());
  pen::IMUData currentData;

  // 4. Render Loop
  while (window.isOpen()) {
    // getLatestData() automatically reads from whichever connection is active
    backend.getLatestData(currentData);

    window.drawCube(currentData);
    window.update();
  }

  return 0;
}
