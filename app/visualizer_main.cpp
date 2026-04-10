#include "pen/io.hpp"
#include "pen/viz.hpp"
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  // Default to "wifi" — works out-of-the-box with mock_esp32.py and real
  // WiFi ESP32.  Other modes match the decoder's argument names exactly so
  // both programs are invoked the same way.
  //
  //   ./bin/visualizer            → WiFi UDP port 5006  (dedicated viz port)
  //   ./bin/visualizer wifi       → WiFi UDP port 5006
  //   ./bin/visualizer usb        → /dev/ttyUSB0
  //   ./bin/visualizer bt         → /dev/rfcomm0
  //   ./bin/visualizer sim        → /tmp/vtty_laptop  (socat virtual cable)
  //
  // Port split: decoder=5005, visualizer=5006.  mock_esp32.py sends to both
  // so decoder + visualizer can run simultaneously without packet loss.
  //
  // Controls:
  //   C      — clear stroke canvas
  //   ESC    — quit

  std::string mode = (argc > 1) ? argv[1] : "wifi";

  pen::PenBackend backend;
  if (mode == "usb")
    backend.connectUSB(pen::Defaults::usbPort);
  else if (mode == "bt")
    backend.connectBluetooth(pen::Defaults::bluetoothPort);
  else if (mode == "sim")
    backend.connectUSB("/tmp/vtty_laptop");
  else // "wifi" (default)
    backend.connectWiFi(pen::Defaults::wifiVizPort);

  std::cout << "=== Smart Pen Visualizer ===\n";
  std::cout << "Mode   : " << backend.getStatus() << "\n";
  std::cout << "C = clear canvas   ESC = quit\n";
  std::cout << "----------------------------\n";

  pen::Visualizer window(800, 800, ("Smart Pen Visualizer [" + mode + "]").c_str());
  pen::IMUData currentData;

  while (window.isOpen()) {
    backend.getLatestData(currentData);
    window.drawCube(currentData);
    window.update();
  }

  return 0;
}
