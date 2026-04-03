#pragma once
#include <string>

namespace pen {
struct IMUData {
  float pitch = 0.0f;
  float roll = 0.0f;
  float yaw = 0.0f;
};

class SerialReader {
public:
  SerialReader(const std::string &portName);
  ~SerialReader();

  bool isOpen() const;
  bool readData(IMUData &data);

private:
  int fd;
  char buffer[256];
  int bufPos;
};
} // namespace pen
