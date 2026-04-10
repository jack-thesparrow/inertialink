#include "pen/io.hpp"
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <termios.h>
#include <unistd.h>

namespace pen {

static constexpr float DEG_TO_RAD = static_cast<float>(M_PI / 180.0);

// ==========================================
// LOW-PASS FILTER MATH
// ==========================================
IMUFilter::IMUFilter(float alpha) : alpha(alpha), isFirstRun(true) {}

void IMUFilter::reset() { isFirstRun = true; }

void IMUFilter::setAlpha(float newAlpha) {
  // Clamp alpha between 0.01 (max smoothing) and 1.0 (no smoothing)
  alpha = std::fmax(0.01f, std::fmin(newAlpha, 1.0f));
}

IMUData IMUFilter::process(const IMUData &raw) {
  if (isFirstRun) {
    previous = raw;
    isFirstRun = false;
    return raw;
  }

  // The Magic Equation: Output = (Alpha * New) + ((1 - Alpha) * Old)
  IMUData filtered;
  filtered.pitch = (alpha * raw.pitch) + ((1.0f - alpha) * previous.pitch);
  filtered.roll = (alpha * raw.roll) + ((1.0f - alpha) * previous.roll);
  filtered.yaw = (alpha * raw.yaw) + ((1.0f - alpha) * previous.yaw);

  // NEVER filter the accelerometer! We want the raw, sharp impact spikes for
  // the ML model.
  filtered.accel_z = raw.accel_z;

  previous = filtered;
  return filtered;
}

// ==========================================
// SERIAL READER (USB / BLUETOOTH)
// ==========================================
SerialReader::SerialReader(const std::string &portName) : fd(-1), bufPos(0) {
  fd = open(portName.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
  if (fd == -1) {
    std::cerr << "[Serial] Failed to open " << portName << "\n";
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
}

SerialReader::~SerialReader() {
  if (fd >= 0)
    close(fd);
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
      // Look for 4 values, but accept 3 so the simulator doesn't crash
      float p, r, y, az = 0.0f;
      int parsed = sscanf(buffer, "%f,%f,%f,%f", &p, &r, &y, &az);
      if (parsed >= 3) {
        data.pitch   = p * DEG_TO_RAD;
        data.roll    = r * DEG_TO_RAD;
        data.yaw     = y * DEG_TO_RAD;
        data.accel_z = az;
        newDataReady = true;
      }
      bufPos = 0;
    } else if (bufPos < 255) {
      buffer[bufPos++] = c;
    }
  }
  return newDataReady;
}

// ==========================================
// UDP READER (WI-FI)
// ==========================================
UDPReader::UDPReader(int port) : sockfd(-1), active(false) {
  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0)
    return;

  // SO_REUSEPORT lets multiple processes (e.g. decoder + visualizer) bind the
  // same UDP port and each receive every packet independently.
  int reuse = 1;
  setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse));

  struct sockaddr_in servaddr;
  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_addr.s_addr = INADDR_ANY;
  servaddr.sin_port = htons(port);

  if (bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) >= 0) {
    active = true;
    fcntl(sockfd, F_SETFL, O_NONBLOCK);
  } else {
    close(sockfd);
    sockfd = -1;
  }
}

UDPReader::~UDPReader() {
  if (sockfd >= 0)
    close(sockfd);
}

bool UDPReader::isOpen() const { return active; }

bool UDPReader::readData(IMUData &data) {
  if (!active)
    return false;
  char buf[1025]; // +1 so buf[n] = '\0' is always in bounds
  int n = recvfrom(sockfd, (char *)buf, 1024, MSG_DONTWAIT, NULL, NULL);
  if (n > 0) {
    buf[n] = '\0';
    float p, r, y, az = 0.0f;
    int parsed = sscanf(buf, "%f,%f,%f,%f", &p, &r, &y, &az);
    if (parsed >= 3) {
      data.pitch   = p * DEG_TO_RAD;
      data.roll    = r * DEG_TO_RAD;
      data.yaw     = y * DEG_TO_RAD;
      data.accel_z = az;
      return true;
    }
  }
  return false;
}

// ==========================================
// PEN BACKEND CONTROLLER
// ==========================================
bool PenBackend::getLatestData(IMUData &data) {
  IMUData rawData;
  // 1. Fetch the noisy data from the hardware
  if (activeReader && activeReader->isOpen() &&
      activeReader->readData(rawData)) {
    // 2. Pass it through the Low-Pass Filter before giving it to the
    // Visualizer/ML
    data = filter.process(rawData);
    return true;
  }
  return false;
}

std::string PenBackend::getStatus() const { return currentStatus; }

void PenBackend::connectUSB(const std::string &port) {
  activeReader = std::make_unique<SerialReader>(port);
  filter.reset();
  if (activeReader->isOpen()) {
    currentMode   = ConnectionMode::USB;
    currentStatus = "Connected via USB (" + port + ")";
  } else {
    currentMode   = ConnectionMode::None;
    currentStatus = "USB Failed (" + port + ")";
  }
}

void PenBackend::connectBluetooth(const std::string &port) {
  activeReader = std::make_unique<SerialReader>(port);
  filter.reset();
  if (activeReader->isOpen()) {
    currentMode   = ConnectionMode::Bluetooth;
    currentStatus = "Connected via Bluetooth (" + port + ")";
  } else {
    currentMode   = ConnectionMode::None;
    currentStatus = "Bluetooth Failed (" + port + ")";
  }
}

void PenBackend::connectWiFi(int listenPort) {
  activeReader = std::make_unique<UDPReader>(listenPort);
  filter.reset();
  if (activeReader->isOpen()) {
    currentMode   = ConnectionMode::WiFi;
    currentStatus = "Listening on WiFi (port " + std::to_string(listenPort) + ")";
  } else {
    currentMode   = ConnectionMode::None;
    currentStatus = "WiFi Failed (port " + std::to_string(listenPort) + ")";
  }
}

void PenBackend::disconnect() {
  activeReader.reset();
  currentMode   = ConnectionMode::None;
  currentStatus = "Disconnected";
}

} // namespace pen
