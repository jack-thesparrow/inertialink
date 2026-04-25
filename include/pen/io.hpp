#pragma once
#include <memory>
#include <string>

namespace pen {

// Connection mode — use this in the UI to drive PenBackend::connect*().
enum class ConnectionMode { None, USB, Bluetooth, WiFi };

// Default port/address values and physics constants.
// All app-layer code references these — nothing is hardcoded in the apps.
// Keep leverArmMm / wakeThresholdZ / activityThreshold / idleTimeoutMs in sync
// with augment_seed_data.py (IDLE_TIMEOUT_MS) and the ESP32 firmware.
struct Defaults {
  // Connection
  static constexpr const char *usbPort = "/dev/ttyUSB0";
  // Backward-compat alias for older app objects that still reference btPort.
  static constexpr const char *btPort = "/dev/rfcomm0";
  // Decoder and visualizer listen on separate ports so both can run
  // simultaneously — SO_REUSEPORT load-balances (one receiver per packet),
  // two ports guarantee each process sees every packet independently.
  static constexpr int wifiPort = 5005;    // decoder
  static constexpr int wifiVizPort = 5006; // visualizer
  // Feedback channel: host→pen (STATE/PRED commands for the OLED display)
  static constexpr const char *wifiFeedbackTarget = "192.168.4.1";
  static constexpr int wifiFeedbackPort = 5007;

  // Physics — distance (mm) from wrist pivot to pen tip used by both the
  // data collector and the ML decoder to project IMU angles onto 2-D canvas.
  static constexpr float leverArmMm = 150.0f;
  // Z-axis shock magnitude that wakes the system from idle.
  static constexpr float wakeThresholdZ = 0.5f;
  // Minimum angular delta (radians) that counts as pen movement.
  static constexpr float activityThreshold = 0.02f;
  // Milliseconds of stillness before a stroke is considered complete.
  static constexpr int idleTimeoutMs = 700;
  // Mounting tilt: angle (degrees) between pen shaft and the writing surface.
  // Adjust to match your natural grip (most people 30-45°).
  static constexpr float tiltAngleDeg = 45.0f;
};

struct IMUData {
  // Accelerometer (g-force)
  float ax = 0.0f;
  float ay = 0.0f;
  float az = 0.0f;
  // Gyroscope (deg/s)
  float gx = 0.0f;
  float gy = 0.0f;
  float gz = 0.0f;
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

// --- WIRED (USB) SERIAL READER ---
// Also used for the socat simulation virtual tty.
// sendCommand() writes a raw string to the serial port so the firmware
// can be poked into the correct mode at connect-time.
class SerialReader : public IMUReader {
public:
  explicit SerialReader(const std::string &portName);
  ~SerialReader() override;

  bool isOpen() const override;
  bool readData(IMUData &data) override;
  void sendCommand(const std::string &cmd);

private:
  int fd;
  char buffer[256];
  int bufPos;
};

// --- WI-FI UDP READER ---
class UDPReader : public IMUReader {
public:
  explicit UDPReader(int port);
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
  void sendCommand(const std::string &cmd);

  // connectUSB: opens the port and immediately sends "MODE:USB\n" so the
  // firmware starts streaming regardless of its current state.
  void connectUSB(const std::string &port = Defaults::usbPort);
  void connectBluetooth(const std::string &port = Defaults::btPort);
  void connectWiFi(int listenPort = Defaults::wifiPort);
  void disconnect();

  // Send a STATE:xxx or PRED:char:conf command back to the ESP32 OLED.
  // USB mode  → writes to the serial port.
  // WiFi mode → sends a UDP packet to 192.168.4.1:5007.
  void sendFeedback(const std::string &msg);

  void setSmoothing(float alpha) { filter.setAlpha(alpha); }

private:
  std::unique_ptr<IMUReader> activeReader;
  IMUFilter filter;
  ConnectionMode currentMode = ConnectionMode::None;
  std::string currentStatus = "Disconnected";
};

namespace device {
bool serialDeviceExists(const std::string &port);
bool esp32DeviceFound(const std::string &preferredPort = Defaults::usbPort);
std::string
resolveEsp32Port(const std::string &preferredPort = Defaults::usbPort);
} // namespace device

} // namespace pen
