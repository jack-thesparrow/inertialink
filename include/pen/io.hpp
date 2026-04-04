#pragma once
#include <memory>
#include <string>

namespace pen {

struct IMUData {
  float pitch = 0.0f;
  float roll = 0.0f;
  float yaw = 0.0f;
  float accel_z = 0.0f; // NEW: The shockwave sensor for pen-lift!
};

// --- SIGNAL PROCESSING: LOW-PASS FILTER ---
class IMUFilter {
public:
  IMUFilter(float alpha = 0.2f); // 0.2 is the sweet spot for IMU smoothing
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
  virtual bool isOpen() const = 0;
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

// --- THE TUI CONTROLLER ---
class PenBackend {
public:
  bool getLatestData(IMUData &data);
  std::string getStatus() const;

  void connectUSB(const std::string &port = "/dev/ttyUSB0");
  void connectBluetooth(const std::string &port = "/dev/rfcomm0");
  void connectWiFi(int listenPort = 5005);
  void disconnect();

  // Allow the future UI to adjust smoothing on the fly!
  void setSmoothing(float alpha) { filter.setAlpha(alpha); }

private:
  std::unique_ptr<IMUReader> activeReader;
  IMUFilter filter; // The backend now owns a filter
  std::string currentStatus = "Disconnected";
};

} // namespace pen
