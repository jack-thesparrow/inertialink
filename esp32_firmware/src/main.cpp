#include <Arduino.h>
#include <Wire.h>
#include <MPU6050_light.h>
#include <BluetoothSerial.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ── OLED ──────────────────────────────────────────────────────────────────────
// Robocraze 0.96" 4-pin I2C yellow-blue OLED (SSD1306, 128×64)
// Shares the I2C bus with MPU6050 (SDA=21, SCL=22). No address conflict:
//   MPU6050 = 0x68, SSD1306 = 0x3C
// Physical colour split: top ~16 px → yellow, remaining 48 px → blue.
#define SCREEN_W   128
#define SCREEN_H    64
#define OLED_ADDR  0x3C
static Adafruit_SSD1306 display(SCREEN_W, SCREEN_H, &Wire, -1);
static bool oledOk = false;

// ── Push buttons (active-LOW, internal pull-up) ───────────────────────────────
// Btn1 (GPIO 14): cycle modes  IDLE → USB → BT → IDLE
// Btn2 (GPIO 27): reset to IDLE from any mode
#define BTN_CYCLE_PIN  14
#define BTN_RESET_PIN  27
#define DEBOUNCE_MS   200
static unsigned long lastBtn1Ms = 0;
static unsigned long lastBtn2Ms = 0;

// ── IMU / radio state ─────────────────────────────────────────────────────────
enum Mode { IDLE, WIRED, BLUETOOTH, WIFI };
static Mode   currentMode = IDLE;
static String targetIP    = "";
static int    targetPort  = 5005;

static MPU6050         mpu(Wire);
static BluetoothSerial SerialBT;
static WiFiUDP         udp;

static float pitch = 0.0f, roll = 0.0f, yaw = 0.0f;

// ── Helpers ───────────────────────────────────────────────────────────────────
static const char* modeName(Mode m) {
  switch (m) {
    case WIRED:     return "USB";
    case BLUETOOTH: return "BLUETOOTH";
    case WIFI:      return "WiFi";
    default:        return "IDLE";
  }
}

// ── OLED drawing ──────────────────────────────────────────────────────────────
// drawHeader: fills the yellow band (y 0-15) with the app title in black.
static void drawHeader() {
  display.fillRect(0, 0, SCREEN_W, 16, SSD1306_WHITE);
  display.setTextColor(SSD1306_BLACK);
  display.setTextSize(1);
  display.setCursor(19, 4);        // roughly centred on 128 px
  display.print("* InertiaLink *");
}

// splashOLED: two-line status message used during boot / WiFi connect.
static void splashOLED(const char* line1, const char* line2 = nullptr) {
  if (!oledOk) return;
  display.clearDisplay();
  drawHeader();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 20);
  display.print(line1);
  if (line2) {
    display.setCursor(0, 32);
    display.print(line2);
  }
  display.display();
}

// drawDisplay: main HUD – called on mode change and periodically while streaming.
static void drawDisplay() {
  if (!oledOk) return;
  display.clearDisplay();
  drawHeader();

  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  // ── Mode row (blue zone starts at y=16) ──────────────────────────────────
  display.setCursor(0, 18);
  display.print("Mode : ");
  display.print(modeName(currentMode));

  if (currentMode == IDLE) {
    // Help text
    display.setCursor(0, 30);
    display.print("Awaiting command");
    display.setCursor(0, 42);
    display.print("[Btn1] Cycle mode");
    display.setCursor(0, 54);
    display.print("[Btn2] -> IDLE");
  } else {
    // Live IMU values
    char buf[22];

    // Pitch  (P: ±xxx.xx deg)
    snprintf(buf, sizeof(buf), "P: %+7.2f deg", pitch);
    display.setCursor(0, 30);
    display.print(buf);

    // Roll
    snprintf(buf, sizeof(buf), "R: %+7.2f deg", roll);
    display.setCursor(0, 40);
    display.print(buf);

    // Yaw
    snprintf(buf, sizeof(buf), "Y: %+7.2f deg", yaw);
    display.setCursor(0, 50);
    display.print(buf);

    // Small "live" dot in the bottom-right corner
    display.fillCircle(124, 61, 2, SSD1306_WHITE);
  }

  display.display();
}

// ── Command parser ────────────────────────────────────────────────────────────
static void parseCommand(const String& raw) {
  String cmd = raw;
  cmd.trim();

  if (cmd == "MODE:USB") {
    currentMode = WIRED;
    Serial.println("[ESP] Mode -> USB");

  } else if (cmd == "MODE:BT") {
    currentMode = BLUETOOTH;
    Serial.println("[ESP] Mode -> BT");

  } else if (cmd == "MODE:IDLE") {
    currentMode = IDLE;
    Serial.println("[ESP] Mode -> IDLE");

  } else if (cmd.startsWith("MODE:WIFI")) {
    // Expected format: MODE:WIFI|SSID|PASSWORD|HOST_IP
    int a = cmd.indexOf('|');
    int b = cmd.indexOf('|', a + 1);
    int c = cmd.indexOf('|', b + 1);

    if (a > 0 && b > 0 && c > 0) {
      String ssid = cmd.substring(a + 1, b);
      String pass = cmd.substring(b + 1, c);
      targetIP    = cmd.substring(c + 1);

      Serial.println("[ESP] Connecting WiFi: " + ssid);
      WiFi.begin(ssid.c_str(), pass.c_str());

      for (int i = 0; i < 20 && WiFi.status() != WL_CONNECTED; i++) {
        char buf[22];
        snprintf(buf, sizeof(buf), "Attempt %d/20 ...", i + 1);
        splashOLED("Connecting WiFi...", buf);
        delay(500);
      }

      if (WiFi.status() == WL_CONNECTED) {
        currentMode = WIFI;
        Serial.println("[ESP] WiFi OK -> " + targetIP);
      } else {
        currentMode = IDLE;
        Serial.println("[ESP] WiFi failed -> IDLE");
        splashOLED("WiFi failed!", "Falling back: IDLE");
        delay(1000);
      }
    }
  }

  drawDisplay();
}

// ── Button handling ───────────────────────────────────────────────────────────
static void handleButtons() {
  unsigned long now = millis();

  // BTN_CYCLE: IDLE → USB → BT → IDLE  (WiFi is serial-only due to credentials)
  if (digitalRead(BTN_CYCLE_PIN) == LOW && (now - lastBtn1Ms) > DEBOUNCE_MS) {
    lastBtn1Ms = now;
    switch (currentMode) {
      case IDLE:      currentMode = WIRED;     break;
      case WIRED:     currentMode = BLUETOOTH; break;
      default:        currentMode = IDLE;      break; // BT or WiFi → IDLE
    }
    Serial.printf("[ESP] Btn1 -> %s\n", modeName(currentMode));
    drawDisplay();
  }

  // BTN_RESET: any → IDLE
  if (digitalRead(BTN_RESET_PIN) == LOW && (now - lastBtn2Ms) > DEBOUNCE_MS) {
    lastBtn2Ms = now;
    currentMode = IDLE;
    Serial.println("[ESP] Btn2 -> IDLE");
    drawDisplay();
  }
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  pinMode(BTN_CYCLE_PIN, INPUT_PULLUP);
  pinMode(BTN_RESET_PIN, INPUT_PULLUP);

  // I2C bus shared by MPU6050 (0x68) and SSD1306 (0x3C)
  Wire.begin(21, 22);

  // OLED – non-fatal if absent
  oledOk = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  if (!oledOk) {
    Serial.println("[ESP] OLED not found - continuing without display");
  }

  splashOLED("Booting...", "Starting BT + IMU");

  // Bluetooth always active so the TUI can send mode commands
  SerialBT.begin("SmartPen_Config");

  // MPU6050 – fatal if absent
  byte imuStatus = mpu.begin();
  if (imuStatus != 0) {
    splashOLED("MPU6050 Error!", "Check I2C wiring");
    Serial.println("[ESP] MPU6050 init failed, halting.");
    while (true) delay(10);
  }

  splashOLED("Calibrating IMU...", "Keep device still");
  delay(1000);
  mpu.calcOffsets();

  Serial.println("[ESP] Boot complete. Waiting for commands.");
  drawDisplay();
}

// ── Main loop ─────────────────────────────────────────────────────────────────
static unsigned long lastDispMs = 0;
static const unsigned long DISP_INTERVAL_MS = 100; // Refresh OLED at 10 Hz

void loop() {
  handleButtons();

  // Inbound serial commands
  if (Serial.available())   parseCommand(Serial.readStringUntil('\n'));
  if (SerialBT.available()) parseCommand(SerialBT.readStringUntil('\n'));

  // Stream IMU data in active modes
  if (currentMode != IDLE) {
    mpu.update();
    pitch = mpu.getAngleX();
    roll  = mpu.getAngleY();
    yaw   = mpu.getAngleZ();

    // CSV: pitch,roll,yaw
    char payload[32];
    snprintf(payload, sizeof(payload), "%.2f,%.2f,%.2f\n", pitch, roll, yaw);

    switch (currentMode) {
      case WIRED:
        Serial.print(payload);
        break;
      case BLUETOOTH:
        SerialBT.print(payload);
        break;
      case WIFI:
        udp.beginPacket(targetIP.c_str(), targetPort);
        udp.print(payload);
        udp.endPacket();
        break;
      default:
        break;
    }

    // Throttle OLED refresh to ~10 Hz to avoid stealing time from 100 Hz IMU
    unsigned long now = millis();
    if (now - lastDispMs >= DISP_INTERVAL_MS) {
      lastDispMs = now;
      drawDisplay();
    }
  }

  delay(10); // 100 Hz cap
}
