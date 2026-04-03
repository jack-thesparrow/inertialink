#include "pen/io.hpp"
#include <cmath>
#include <fcntl.h>
#include <iostream>
#include <termios.h>
#include <unistd.h>

namespace pen {
SerialReader::SerialReader(const std::string &portName) : fd(-1), bufPos(0) {
  fd = open(portName.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
  if (fd == -1) {
    std::cerr << "[Serial] Failed to open " << portName
              << " (Check permissions!)\n";
    return;
  }

  struct termios options;
  tcgetattr(fd, &options);
  cfsetispeed(&options, B115200);
  cfsetospeed(&options, B115200);

  options.c_cflag |= (CLOCAL | CREAD | CS8);
  options.c_cflag &= ~(PARENB | CSTOPB | CSIZE);
  options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);

  tcsetattr(fd, TCSANOW, &options);
  fcntl(fd, F_SETFL, FNDELAY);
  std::cout << "[Serial] Connected to " << portName << "\n";
}

SerialReader::~SerialReader() {
  if (fd >= 0) {
    close(fd);
    std::cout << "[Serial] Port closed.\n";
  }
}

bool SerialReader::isOpen() const { return fd >= 0; }

bool SerialReader::readData(IMUData &data) {
  if (!isOpen())
    return false;
  char c;
  bool newDataReady = false;

  while (read(fd, &c, 1) > 0) {
    if (c == '\n') {
      buffer[bufPos] = '\0';
      float p, r, y;
      if (sscanf(buffer, "%f,%f,%f", &p, &r, &y) == 3) {
        data.pitch = p * (M_PI / 180.0f);
        data.roll = r * (M_PI / 180.0f);
        data.yaw = y * (M_PI / 180.0f);
        newDataReady = true;
      }
      bufPos = 0;
    } else if (bufPos < 255) {
      buffer[bufPos++] = c;
    }
  }
  return newDataReady;
}
} // namespace pen
