// InertiaLink ESP32 Firmware  —  USB + WiFi only (no Bluetooth)
//
// Default boot mode: WIRED (USB streaming starts immediately).
// This matters because the desktop apps open the serial port and expect
// data to flow without sending any mode-switch command first.
//
// Wiring:
//   MPU6050   SDA → GPIO 21   SCL → GPIO 22   (I2C address 0x68)
//   SSD1306   SDA → GPIO 21   SCL → GPIO 22   (I2C address 0x3C)
//   Btn1      → GPIO 14   (other pin → GND)   toggles WIRED ↔ WIFI
//   Btn2      → GPIO 27   (other pin → GND)   force back to WIRED
//
// Serial commands (sent by the TUI or any serial terminal at 115200):
//   MODE:USB                         – stream pitch/roll/yaw over USB
//   MODE:WIFI|<ssid>|<pass>|<host>   – connect WiFi, stream to host:5005
//   MODE:IDLE                        – pause streaming

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Arduino.h>
#include <MPU6050_light.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

// ── OLED ──────────────────────────────────────────────────────────────────────
// Robocraze 0.96" 4-pin yellow-blue OLED (SSD1306, 128×64).
// Hardware colour split (fixed by the panel, not the driver):
//   top 16 px  → yellow    ← we use this as a dynamic status bar
//   bottom 48 px → blue    ← we use this for sensor values / hints
//
// Both devices share Wire (SDA=21, SCL=22).
// MPU6050 is at 0x68, SSD1306 is at 0x3C — no address conflict.
#define SCREEN_W  128
#define SCREEN_H   64
#define OLED_ADDR 0x3C
static Adafruit_SSD1306 display(SCREEN_W, SCREEN_H, &Wire, /*reset=*/-1);
static bool oledOk = false;

// ── Push buttons (active-LOW with internal pull-up) ───────────────────────────
#define BTN_CYCLE_PIN  14   // GPIO 14: toggle WIRED ↔ WIFI
#define BTN_RESET_PIN  27   // GPIO 27: force WIRED
#define DEBOUNCE_MS   200
static unsigned long lastBtn1Ms = 0;
static unsigned long lastBtn2Ms = 0;

// ── Mode & radio state ────────────────────────────────────────────────────────
enum Mode { WIRED, WIFI, IDLE };

static Mode   currentMode    = WIRED;  // boot in WIRED — data flows immediately
static bool   wifiReady      = false;  // true once WiFi creds received & connected
static String wifiTargetIP   = "";
static int    wifiTargetPort = 5005;

static MPU6050 mpu(Wire);
static WiFiUDP udp;

static float pitch = 0.0f, roll = 0.0f, yaw = 0.0f;

// ── OLED helpers ──────────────────────────────────────────────────────────────
static const char *modeName(Mode m) {
  switch (m) {
  case WIRED: return "WIRED";
  case WIFI:  return "WIFI ";
  default:    return "IDLE ";
  }
}

// ── Yellow zone ───────────────────────────────────────────────────────────────
// fillRect (0,0,128,16,WHITE) inverts the top strip so it renders yellow on the
// physical panel. We draw BLACK text on it.
//
// Line 0 (y=0): mode label + live/idle badge
// Line 1 (y=8): pitch & roll (the two most useful real-time values)
static void drawYellowZone(bool streaming) {
  display.fillRect(0, 0, SCREEN_W, 16, SSD1306_WHITE);
  display.setTextColor(SSD1306_BLACK);
  display.setTextSize(1);

  if (currentMode == IDLE) {
    display.setCursor(12, 1);
    display.print("* InertiaLink v2 *");
    display.setCursor(16, 9);
    display.print("USB + WiFi  no BT");
    return;
  }

  // Row 0: "WIRED  [LIVE]" or "WIFI   [LIVE]"
  display.setCursor(0, 1);
  display.print(modeName(currentMode));
  display.setCursor(66, 1);
  display.print(streaming ? "  [LIVE]" : "  [----]");

  // Row 1: P and R tightly packed so both fit on one yellow line
  char buf[22];
  snprintf(buf, sizeof(buf), "P:%+5.1f  R:%+5.1f", pitch, roll);
  display.setCursor(0, 9);
  display.print(buf);
}

// ── Blue zone ─────────────────────────────────────────────────────────────────
// Normal white-on-black text occupying y=16..63 (6 rows at textSize=1).
// The physical panel renders this in blue.
static void drawBlueZone() {
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  if (currentMode == IDLE) {
    display.setCursor(0, 18);
    display.print("Awaiting command...");
    display.setCursor(0, 30);
    display.print("[B1] WIRED / WIFI");
    display.setCursor(0, 40);
    display.print("[B2] Force WIRED");
    display.setCursor(0, 52);
    display.print("TUI: MODE:WIFI|...");
    return;
  }

  // Yaw value (row 1 of blue zone)
  char buf[22];
  snprintf(buf, sizeof(buf), "Y: %+6.1f deg", yaw);
  display.setCursor(0, 18);
  display.print(buf);

  // WiFi target IP (row 2, only in WIFI mode)
  if (currentMode == WIFI && wifiTargetIP.length() > 0) {
    display.setCursor(0, 28);
    display.print("->");
    display.print(wifiTargetIP.substring(0, 17).c_str()); // truncate if long
  }

  // Thin separator
  display.drawFastHLine(0, 40, SCREEN_W, SSD1306_WHITE);

  // Button hints (rows 5-6 of blue zone)
  display.setCursor(0, 44);
  display.print(currentMode == WIRED ? "[B1] -> WIFI" : "[B1] -> WIRED");
  display.setCursor(0, 54);
  display.print("[B2] Force WIRED");
}

// Full display refresh — call on mode changes and ~10 Hz while streaming.
static void drawDisplay(bool streaming = true) {
  if (!oledOk) return;
  display.clearDisplay();
  drawYellowZone(streaming);
  drawBlueZone();
  display.display();
}

// Status/progress splash — used during boot and WiFi connecting.
// Yellow zone = title (inverted), blue zone = two info lines.
static void splashOLED(const char *title, const char *line1 = nullptr,
                       const char *line2 = nullptr) {
  if (!oledOk) return;
  display.clearDisplay();

  // Yellow zone: bold title
  display.fillRect(0, 0, SCREEN_W, 16, SSD1306_WHITE);
  display.setTextColor(SSD1306_BLACK);
  display.setTextSize(1);
  display.setCursor(2, 4);
  display.print(title);

  // Blue zone: info lines
  display.setTextColor(SSD1306_WHITE);
  if (line1) { display.setCursor(0, 20); display.print(line1); }
  if (line2) { display.setCursor(0, 32); display.print(line2); }

  display.display();
}

// ── Serial command parser ─────────────────────────────────────────────────────
static void parseCommand(const String &raw) {
  String cmd = raw;
  cmd.trim();

  if (cmd == "MODE:USB") {
    currentMode = WIRED;
    Serial.println("[ESP] Mode -> WIRED (USB)");

  } else if (cmd == "MODE:IDLE") {
    currentMode = IDLE;
    Serial.println("[ESP] Mode -> IDLE");

  } else if (cmd.startsWith("MODE:WIFI")) {
    // Format: MODE:WIFI|<ssid>|<password>|<host_ip>
    int a = cmd.indexOf('|');
    int b = cmd.indexOf('|', a + 1);
    int c = cmd.indexOf('|', b + 1);

    if (a > 0 && b > 0 && c > 0) {
      String ssid = cmd.substring(a + 1, b);
      String pass = cmd.substring(b + 1, c);
      wifiTargetIP = cmd.substring(c + 1);

      Serial.println("[ESP] Connecting WiFi: " + ssid);

      for (int i = 0; i < 20 && WiFi.status() != WL_CONNECTED; i++) {
        WiFi.begin(ssid.c_str(), pass.c_str());
        char buf[22];
        snprintf(buf, sizeof(buf), "Try %d/20...", i + 1);
        splashOLED("Connecting WiFi", ssid.c_str(), buf);
        delay(500);
      }

      if (WiFi.status() == WL_CONNECTED) {
        wifiReady  = true;
        currentMode = WIFI;
        Serial.println("[ESP] WiFi OK -> " + wifiTargetIP);
      } else {
        wifiReady  = false;
        currentMode = WIRED; // fall back — don't leave the user stranded
        Serial.println("[ESP] WiFi failed -> back to WIRED");
        splashOLED("WiFi failed!", "Back to WIRED");
        delay(1200);
      }
    }
  }

  drawDisplay(currentMode != IDLE);
}

// ── Button handler ────────────────────────────────────────────────────────────
static void handleButtons() {
  unsigned long now = millis();

  // BTN_CYCLE: WIRED -> WIFI (only if configured) / WIFI -> WIRED / IDLE -> WIRED
  if (digitalRead(BTN_CYCLE_PIN) == LOW && (now - lastBtn1Ms) > DEBOUNCE_MS) {
    lastBtn1Ms = now;

    if (currentMode == WIRED) {
      if (wifiReady) {
        currentMode = WIFI;
        Serial.println("[ESP] Btn1: WIRED -> WIFI");
      } else {
        splashOLED("No WiFi config", "Send MODE:WIFI|...", "from TUI first");
        delay(1500);
      }
    } else {
      currentMode = WIRED;
      Serial.println("[ESP] Btn1: -> WIRED");
    }
    drawDisplay(currentMode != IDLE);
  }

  // BTN_RESET: always go back to WIRED
  if (digitalRead(BTN_RESET_PIN) == LOW && (now - lastBtn2Ms) > DEBOUNCE_MS) {
    lastBtn2Ms  = now;
    currentMode = WIRED;
    Serial.println("[ESP] Btn2: -> WIRED");
    drawDisplay(true);
  }
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  pinMode(BTN_CYCLE_PIN, INPUT_PULLUP);
  pinMode(BTN_RESET_PIN, INPUT_PULLUP);

  // Shared I2C bus for MPU6050 (0x68) and SSD1306 (0x3C)
  Wire.begin(21, 22);

  // OLED — non-fatal if absent (firmware works headless)
  oledOk = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  if (!oledOk)
    Serial.println("[ESP] OLED not found — continuing headless");

  splashOLED("InertiaLink v2", "USB + WiFi only", "Booting...");

  // MPU6050 — fatal if absent
  if (mpu.begin() != 0) {
    splashOLED("MPU6050 Error!", "Check I2C wiring");
    Serial.println("[ESP] MPU6050 init failed — halting");
    while (true) delay(10);
  }

  splashOLED("Calibrating IMU", "Keep device still");
  delay(1000);
  mpu.calcOffsets();

  Serial.println("[ESP] Boot OK — streaming WIRED (USB) at 100 Hz");
  drawDisplay(true);
}

// ── Main loop ─────────────────────────────────────────────────────────────────
static unsigned long lastDispMs = 0;
static const unsigned long DISP_INTERVAL_MS = 100; // refresh OLED at 10 Hz

void loop() {
  handleButtons();

  // Inbound commands from the desktop TUI or any terminal
  if (Serial.available())
    parseCommand(Serial.readStringUntil('\n'));

  // Stream IMU data in active modes
  if (currentMode != IDLE) {
    mpu.update();
    pitch = mpu.getAngleX();
    roll  = mpu.getAngleY();
    yaw   = mpu.getAngleZ();

    // CSV format: pitch,roll,yaw  — matches SerialReader::readData() parser
    char payload[32];
    snprintf(payload, sizeof(payload), "%.2f,%.2f,%.2f\n", pitch, roll, yaw);

    if (currentMode == WIRED) {
      Serial.print(payload);
    } else { // WIFI
      udp.beginPacket(wifiTargetIP.c_str(), wifiTargetPort);
      udp.print(payload);
      udp.endPacket();
    }

    // Throttle OLED to 10 Hz — avoids I2C stealing time from the 100 Hz loop
    unsigned long now = millis();
    if (now - lastDispMs >= DISP_INTERVAL_MS) {
      lastDispMs = now;
      drawDisplay(true);
    }
  }

  delay(10); // 100 Hz cap
}
