#include "pen/io.hpp"
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <termios.h>
#include <unistd.h>
#include <vector>

namespace pen {

static constexpr float DEG_TO_RAD = static_cast<float>(M_PI / 180.0);
namespace fs = std::filesystem;

namespace {
std::string devRoot() {
  const char *root = std::getenv("INERTIALINK_DEV_ROOT");
  return (root && *root) ? root : "/dev";
}

bool containsAnyToken(const std::string &value,
                      const std::vector<std::string> &tokens) {
  std::string lower = value;
  for (char &c : lower)
    c = static_cast<char>(std::tolower(c));

  for (const auto &t : tokens) {
    if (lower.find(t) != std::string::npos)
      return true;
  }
  return false;
}

std::vector<std::string> scanCandidatePorts() {
  std::vector<std::string> ports;
  const std::string root = devRoot();
  const std::vector<std::string> idTokens = {"esp32", "cp210", "ch340", "wch", "silicon_labs"};
  const fs::path byId = fs::path(root) / "serial" / "by-id";

  if (fs::exists(byId) && fs::is_directory(byId)) {
    for (const auto &entry : fs::directory_iterator(byId)) {
      const std::string name = entry.path().filename().string();
      if (containsAnyToken(name, idTokens)) {
        std::error_code ec;
        fs::path resolved = fs::weakly_canonical(entry.path(), ec);
        if (!ec)
          ports.push_back(resolved.string());
      }
    }
  }

  for (const auto &entry : fs::directory_iterator(root)) {
    const std::string name = entry.path().filename().string();
    if (name.rfind("ttyUSB", 0) == 0 || name.rfind("ttyACM", 0) == 0) {
      ports.push_back(entry.path().string());
    }
  }
  return ports;
}
} // namespace

// ==========================================
// LOW-PASS FILTER
// ==========================================
IMUFilter::IMUFilter(float alpha) : alpha(alpha), isFirstRun(true) {}

void IMUFilter::reset() { isFirstRun = true; }

void IMUFilter::setAlpha(float newAlpha) {
  alpha = std::fmax(0.01f, std::fmin(newAlpha, 1.0f));
}

IMUData IMUFilter::process(const IMUData &raw) {
  if (isFirstRun) {
    previous   = raw;
    isFirstRun = false;
    return raw;
  }

  IMUData filtered;
  filtered.pitch = (alpha * raw.pitch) + ((1.0f - alpha) * previous.pitch);
  filtered.roll  = (alpha * raw.roll)  + ((1.0f - alpha) * previous.roll);
  filtered.yaw   = (alpha * raw.yaw)   + ((1.0f - alpha) * previous.yaw);

  // Never filter the accelerometer — we need the raw impact spikes for ML.
  filtered.accel_z = raw.accel_z;

  previous = filtered;
  return filtered;
}

// ==========================================
// SERIAL READER  (USB / socat sim)
// ==========================================
SerialReader::SerialReader(const std::string &portName) : fd(-1), bufPos(0) {
  fd = open(portName.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
  if (fd == -1) {
    std::cerr << "[Serial] Failed to open " << portName << "\n";
    return;
  }

  struct termios options;
  tcgetattr(fd, &options);
  cfmakeraw(&options);
  cfsetispeed(&options, B115200);
  cfsetospeed(&options, B115200);
  options.c_cflag |= (CLOCAL | CREAD);
  options.c_cflag &= ~(CRTSCTS | PARENB | CSTOPB);
  options.c_cflag &= ~CSIZE;
  options.c_cflag |= CS8;
  options.c_cc[VMIN]  = 0;
  options.c_cc[VTIME] = 1;
  tcflush(fd, TCIOFLUSH);
  tcsetattr(fd, TCSANOW, &options);
  fcntl(fd, F_SETFL, FNDELAY);
}

SerialReader::~SerialReader() {
  if (fd >= 0)
    close(fd);
}

bool SerialReader::isOpen() const { return fd >= 0; }

void SerialReader::sendCommand(const std::string &cmd) {
  if (fd >= 0)
    ::write(fd, cmd.c_str(), cmd.size());
}

bool SerialReader::readData(IMUData &data) {
  if (!isOpen())
    return false;

  char c;
  bool newDataReady = false;

  while (read(fd, &c, 1) > 0) {
    if (c == '\n') {
      buffer[bufPos] = '\0';
      // Accept 3 values (pitch,roll,yaw) or 4 (with accel_z).
      float p, r, y, az = 0.0f;
      if (sscanf(buffer, "%f,%f,%f,%f", &p, &r, &y, &az) >= 3) {
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
// UDP READER  (Wi-Fi)
// ==========================================
UDPReader::UDPReader(int port) : sockfd(-1), active(false) {
  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0)
    return;

  // SO_REUSEPORT lets decoder (port 5005) and visualizer (port 5006) each
  // receive every packet independently without dropping either stream.
  int reuse = 1;
  setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse));

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family      = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port        = htons(port);

  if (bind(sockfd, reinterpret_cast<const struct sockaddr *>(&addr),
           sizeof(addr)) >= 0) {
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

  char buf[1025]; // +1 so buf[n]='\0' is always in bounds
  int n = recvfrom(sockfd, buf, 1024, MSG_DONTWAIT, nullptr, nullptr);
  if (n > 0) {
    buf[n] = '\0';
    float p, r, y, az = 0.0f;
    if (sscanf(buf, "%f,%f,%f,%f", &p, &r, &y, &az) >= 3) {
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
  IMUData raw;
  if (activeReader && activeReader->isOpen() && activeReader->readData(raw)) {
    data = filter.process(raw);
    return true;
  }
  return false;
}

std::string PenBackend::getStatus() const { return currentStatus; }

void PenBackend::connectUSB(const std::string &port) {
  const std::string resolvedPort = device::resolveEsp32Port(port);
  auto reader = std::make_unique<SerialReader>(resolvedPort.empty() ? port : resolvedPort);
  filter.reset();

  if (reader->isOpen()) {
    // ── Key fix for wired mode ───────────────────────────────────────────────
    // Opening the port toggles DTR which resets the ESP32 (~200 ms).
    // We wait just long enough for the DTR reset to complete, then write
    // MODE:USB into the UART buffer.  The firmware reads it from the buffer
    // when loop() starts (~1.5–2 s later) and confirms WIRED mode.
    // Apps don't need to wait further — getLatestData() returns false until
    // the ESP32 finishes calibration and starts streaming, then data flows.
    usleep(300000); // 300 ms: let USB-serial reset settle
    reader->sendCommand("MODE:USB\n");
    // Some boards drop the first command while booting/calibrating.
    usleep(1200000);
    reader->sendCommand("MODE:USB\n");
    currentMode   = ConnectionMode::USB;
    currentStatus = "Connected via USB (" + (resolvedPort.empty() ? port : resolvedPort) + ")";
  } else {
    currentMode   = ConnectionMode::None;
    currentStatus = "USB Failed (" + port + ") - ESP32 not found / not accessible";
  }

  activeReader = std::move(reader);
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

namespace device {
bool serialDeviceExists(const std::string &port) {
  std::error_code ec;
  return !port.empty() && fs::exists(port, ec);
}

std::string resolveEsp32Port(const std::string &preferredPort) {
  if (serialDeviceExists(preferredPort))
    return preferredPort;

  for (const auto &candidate : scanCandidatePorts()) {
    if (serialDeviceExists(candidate))
      return candidate;
  }
  return {};
}

bool esp32DeviceFound(const std::string &preferredPort) {
  return !resolveEsp32Port(preferredPort).empty();
}
} // namespace device

} // namespace pen
