#pragma once
#include <memory>
#include <string>

namespace pen {

// Connection mode — use this in the UI to drive PenBackend::connect*().
enum class ConnectionMode { None, USB, Bluetooth, WiFi };

// Default port/address values and physics constants.
// All app-layer code references these — nothing is hardcoded in the apps.
// Keep leverArmMm / wakeThresholdZ / activityThreshold / idleTimeoutMs in sync
// with generate_synthetic_data.py (IDLE_TIMEOUT_MS) and the ESP32 firmware.
struct Defaults {
  // Connection
  static constexpr const char *usbPort       = "/dev/ttyUSB0";
  static constexpr const char *bluetoothPort = "/dev/rfcomm0";
  static constexpr int         wifiPort      = 5005;

  // Physics — distance (mm) from wrist pivot to pen tip used by both the
  // data collector and the ML decoder to project IMU angles onto 2-D canvas.
  static constexpr float leverArmMm        = 150.0f;
  // Z-axis shock magnitude that wakes the system from idle.
  static constexpr float wakeThresholdZ    = 0.5f;
  // Minimum angular delta (radians) that counts as pen movement.
  static constexpr float activityThreshold = 0.02f;
  // Milliseconds of stillness before a stroke is considered complete.
  static constexpr int   idleTimeoutMs     = 2000;
};

struct IMUData {
  float pitch = 0.0f;
  float roll = 0.0f;
  float yaw = 0.0f;
  float accel_z = 0.0f;
};

// --- SIGNAL PROCESSING: LOW-PASS FILTER ---
class IMUFilter {
public:
  IMUFilter(float alpha = 0.2f);
  IMUData process(const IMUData &raw);
  void reset();
  void setAlpha(float newAlpha);

private:
  float alpha;
  IMUData previous;
  bool isFirstRun;
};

// --- ABSTRACT BASE CLASS ---
class IMUReader {
public:
  virtual ~IMUReader() = default;
  [[nodiscard]] virtual bool isOpen() const = 0;
  virtual bool readData(IMUData &data) = 0;
};

// --- WIRED & BLUETOOTH READER ---
class SerialReader : public IMUReader {
public:
  SerialReader(const std::string &portName);
  ~SerialReader() override;

  bool isOpen() const override;
  bool readData(IMUData &data) override;

private:
  int fd;
  char buffer[256];
  int bufPos;
};

// --- WI-FI UDP READER ---
class UDPReader : public IMUReader {
public:
  UDPReader(int port);
  ~UDPReader() override;

  bool isOpen() const override;
  bool readData(IMUData &data) override;

private:
  int sockfd;
  bool active;
};

// --- BACKEND CONTROLLER ---
// The UI calls connect*() to switch modes at runtime; getMode() lets the UI
// reflect the current state without keeping its own copy.
class PenBackend {
public:
  bool getLatestData(IMUData &data);
  std::string getStatus() const;
  ConnectionMode getMode() const { return currentMode; }

  void connectUSB(const std::string &port = Defaults::usbPort);
  void connectBluetooth(const std::string &port = Defaults::bluetoothPort);
  void connectWiFi(int listenPort = Defaults::wifiPort);
  void disconnect();

  void setSmoothing(float alpha) { filter.setAlpha(alpha); }

private:
  std::unique_ptr<IMUReader> activeReader;
  IMUFilter filter;
  ConnectionMode currentMode = ConnectionMode::None;
  std::string currentStatus  = "Disconnected";
};

} // namespace pen
