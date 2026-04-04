#include <Arduino.h>
#include <BluetoothSerial.h>
#include <MPU6050_light.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

// --- DYNAMIC STATE ---
enum Mode { IDLE, WIRED, BLUETOOTH, WIFI };
Mode currentMode = IDLE;

String targetIP = "";
int targetPort = 5005;

MPU6050 mpu(Wire);
BluetoothSerial SerialBT;
WiFiUDP udp;

void parseCommand(String cmd) {
  cmd.trim();
  if (cmd == "MODE:USB") {
    currentMode = WIRED;
    Serial.println("[ESP] Switched to USB Streaming");
  } else if (cmd == "MODE:BT") {
    currentMode = BLUETOOTH;
    Serial.println("[ESP] Switched to BT Streaming");
  } else if (cmd.startsWith("MODE:WIFI")) {
    // Expected format: MODE:WIFI|SSID|PASSWORD|HOST_IP
    int split1 = cmd.indexOf('|');
    int split2 = cmd.indexOf('|', split1 + 1);
    int split3 = cmd.indexOf('|', split2 + 1);

    if (split1 > 0 && split2 > 0 && split3 > 0) {
      String ssid = cmd.substring(split1 + 1, split2);
      String pass = cmd.substring(split2 + 1, split3);
      targetIP = cmd.substring(split3 + 1);

      Serial.println("[ESP] Connecting to WiFi: " + ssid);
      WiFi.begin(ssid.c_str(), pass.c_str());

      int retries = 0;
      while (WiFi.status() != WL_CONNECTED && retries < 20) {
        delay(500);
        retries++;
      }

      if (WiFi.status() == WL_CONNECTED) {
        currentMode = WIFI;
        Serial.println("[ESP] WiFi Connected! Streaming to " + targetIP);
      } else {
        Serial.println("[ESP] WiFi Failed. Falling back to IDLE.");
        currentMode = IDLE;
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  SerialBT.begin("SmartPen_Config"); // Always active so TUI can send commands
  Wire.begin(21, 22);

  byte status = mpu.begin();
  if (status != 0) {
    while (1)
      delay(10);
  }
  delay(1000);
  mpu.calcOffsets();
  Serial.println("[ESP] Boot Complete. Waiting for TUI commands...");
}

void loop() {
  // 1. Check for commands from USB
  if (Serial.available()) {
    parseCommand(Serial.readStringUntil('\n'));
  }
  // 2. Check for commands from Bluetooth
  if (SerialBT.available()) {
    parseCommand(SerialBT.readStringUntil('\n'));
  }

  // 3. Stream Data based on current mode
  if (currentMode != IDLE) {
    mpu.update();
    String payload = String(mpu.getAngleX()) + "," + String(mpu.getAngleY()) +
                     "," + String(mpu.getAngleZ()) + "\n";

    if (currentMode == WIRED) {
      Serial.print(payload);
    } else if (currentMode == BLUETOOTH) {
      SerialBT.print(payload);
    } else if (currentMode == WIFI) {
      udp.beginPacket(targetIP.c_str(), targetPort);
      udp.print(payload);
      udp.endPacket();
    }
  }
  delay(10); // 100Hz
}
