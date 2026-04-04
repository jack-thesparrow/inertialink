#pragma once
#include <memory>
#include <string>

namespace pen {

struct IMUData {
  float pitch = 0.0f;
  float roll = 0.0f;
  float yaw = 0.0f;
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
  char buffer[256]; // Fixed: Now a char array to match your loop!
  int bufPos;       // Fixed: Added bufPos to track the index!
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

private:
  std::unique_ptr<IMUReader> activeReader;
  std::string currentStatus = "Disconnected";
};

} // namespace pen
