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

// ==========================================
// SERIAL READER (USB / BLUETOOTH)
// ==========================================
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
        // Preserved your Radian conversion!
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

// ==========================================
// UDP READER (WI-FI)
// ==========================================
UDPReader::UDPReader(int port) : sockfd(-1), active(false) {
  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0) {
    std::cerr << "[UDP] Failed to create socket\n";
    return;
  }

  struct sockaddr_in servaddr;
  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_addr.s_addr = INADDR_ANY;
  servaddr.sin_port = htons(port);

  if (bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) >= 0) {
    active = true;
    fcntl(sockfd, F_SETFL, O_NONBLOCK);
    std::cout << "[UDP] Listening on port " << port << "\n";
  } else {
    std::cerr << "[UDP] Bind failed on port " << port << "\n";
    close(sockfd);
    sockfd = -1;
  }
}

UDPReader::~UDPReader() {
  if (sockfd >= 0) {
    close(sockfd);
    std::cout << "[UDP] Socket closed.\n";
  }
}

bool UDPReader::isOpen() const { return active; }

bool UDPReader::readData(IMUData &data) {
  if (!active)
    return false;
  char buf[1024];

  int n = recvfrom(sockfd, (char *)buf, 1024, MSG_DONTWAIT, NULL, NULL);
  if (n > 0) {
    buf[n] = '\0';
    float p, r, y;
    if (sscanf(buf, "%f,%f,%f", &p, &r, &y) == 3) {
      // Applied your Radian conversion to Wi-Fi data as well
      data.pitch = p * (M_PI / 180.0f);
      data.roll = r * (M_PI / 180.0f);
      data.yaw = y * (M_PI / 180.0f);
      return true;
    }
  }
  return false;
}

// ==========================================
// PEN BACKEND CONTROLLER (FOR FTXUI)
// ==========================================
bool PenBackend::getLatestData(IMUData &data) {
  if (activeReader && activeReader->isOpen()) {
    return activeReader->readData(data);
  }
  return false;
}

std::string PenBackend::getStatus() const { return currentStatus; }

void PenBackend::connectUSB(const std::string &port) {
  activeReader = std::make_unique<SerialReader>(port);
  currentStatus = activeReader->isOpen() ? "Connected to USB: " + port
                                         : "USB Failed: " + port;
}

void PenBackend::connectBluetooth(const std::string &port) {
  activeReader = std::make_unique<SerialReader>(port);
  currentStatus = activeReader->isOpen() ? "Connected to BT: " + port
                                         : "BT Failed: " + port;
}

void PenBackend::connectWiFi(int listenPort) {
  activeReader = std::make_unique<UDPReader>(listenPort);
  currentStatus = activeReader->isOpen()
                      ? "Listening on UDP " + std::to_string(listenPort)
                      : "WiFi Bind Failed";
}

void PenBackend::disconnect() {
  activeReader.reset();
  currentStatus = "Disconnected";
}

} // namespace pen
