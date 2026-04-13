#include "pen/io.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

static bool expect(bool cond, const char* msg) {
  if (!cond) {
    std::cerr << "[FAIL] " << msg << "\n";
    return false;
  }
  return true;
}

int main() {
  bool ok = true;

  ok &= expect(!pen::device::serialDeviceExists("/this/path/does/not/exist"),
               "serialDeviceExists should be false for non-existent path");

  const fs::path base = fs::temp_directory_path() / "inertialink_io_test_dev";
  fs::remove_all(base);
  fs::create_directories(base / "serial" / "by-id");

  const fs::path tty = base / "ttyUSB7";
  {
    std::ofstream f(tty.string());
    f << "fake tty";
  }

  const fs::path byId = base / "serial" / "by-id" / "usb-ESP32_CP210x";
  fs::create_symlink(tty, byId);

  ::setenv("INERTIALINK_DEV_ROOT", base.c_str(), 1);

  ok &= expect(pen::device::serialDeviceExists(tty.string()),
               "serialDeviceExists should detect mock tty device");
  ok &= expect(pen::device::esp32DeviceFound("/nonexistent"),
               "esp32DeviceFound should discover ESP32-like symlink in by-id");
  ok &= expect(pen::device::resolveEsp32Port("/nonexistent") == tty.string(),
               "resolveEsp32Port should resolve to discovered ttyUSB device");

  fs::remove_all(base);
  return ok ? 0 : 1;
}
