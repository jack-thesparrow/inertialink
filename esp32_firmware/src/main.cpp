// InertiaLink ESP32 Firmware — USB + WiFi only (no Bluetooth)
//
// Default boot mode: WIRED.  The desktop apps open the port and expect
// data immediately; no MODE:USB command needed before streaming begins.
//
// Wiring:
//   MPU6050   SDA→GPIO 21   SCL→GPIO 22   I2C address 0x68
//   SSD1306   SDA→GPIO 21   SCL→GPIO 22   I2C address 0x3C   (no conflict)
//   Btn1      GPIO 14 ↔ GND   —  toggles WIRED ↔ WIFI
//   Btn2      GPIO 27 ↔ GND   —  forces  WIRED
//
// Serial commands at 115200 baud (TUI or any terminal):
//   MODE:USB                        switch to / stay in WIRED mode
//   MODE:WIFI|<ssid>|<pass>|<host>  connect WiFi, stream to host:5005
//   MODE:IDLE                       pause streaming
//
// Payload format (sent in WIRED/WIFI modes at 100 Hz):
//   pitch,roll,yaw,accel_z\n   (all floats, degrees + g's)
//   The 4th field (accel_z from mpu.getAccZ()) is the raw Z-axis
//   acceleration in g.  It is REQUIRED by the PC-side apps:
//     • data_collector / decoder  — wake-on-impact:  |az – prev_az| > 0.5
//     • visualizer 2D canvas      — pen-contact test: |az| >= 0.05
//   Without it the 2D canvas never draws and the collector never wakes.

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Arduino.h>
#include <MPU6050_light.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

// ── OLED ──────────────────────────────────────────────────────────────────────
// Robocraze 0.96" 4-pin yellow-blue OLED, SSD1306, 128×64.
// Hardware colour split (fixed by the panel, not the driver):
//   top 16 px  → yellow     bottom 48 px → blue
//
// Wire runs at 400 kHz (fast mode).
// Default 100 kHz ≈ 92 ms per full-frame push → would block the 100 Hz loop.
// 400 kHz ≈ 23 ms → acceptable overhead.
#define SCREEN_W  128
#define SCREEN_H   64
#define OLED_ADDR 0x3C
static Adafruit_SSD1306 display(SCREEN_W, SCREEN_H, &Wire, /*reset=*/-1);
static bool oledOk = false;

// ── Push buttons (active-LOW, internal pull-up) ───────────────────────────────
#define BTN_CYCLE_PIN  14   // toggle WIRED ↔ WIFI
#define BTN_RESET_PIN  27   // force WIRED
#define DEBOUNCE_MS   200
static unsigned long lastBtn1Ms = 0, lastBtn2Ms = 0;

// ── Non-blocking serial command buffer ────────────────────────────────────────
// Serial.readStringUntil() blocks up to Serial.getTimeout() (default 1 s) when
// bytes arrive without a '\n'.  That stalls the 100 Hz loop.  This char-by-char
// buffer never blocks.
static char cmdBuf[128];
static int  cmdPos = 0;

// ── Mode & radio state ────────────────────────────────────────────────────────
enum Mode { WIRED, WIFI, IDLE };

static Mode   currentMode    = WIRED;   // boot default: stream over USB immediately
static bool   wifiReady      = false;
static String wifiTargetIP   = "";
static int    wifiTargetPort = 5005;

static MPU6050 mpu(Wire);
static WiFiUDP udp;

// Live IMU values (globals so drawDisplay can read them without passing args)
static float pitch = 0.0f, roll = 0.0f, yaw = 0.0f, accelZ = 0.0f;

// ── OLED helpers ──────────────────────────────────────────────────────────────
static const char *modeName(Mode m) {
  switch (m) {
    case WIRED: return "WIRED";
    case WIFI:  return "WIFI ";
    default:    return "IDLE ";
  }
}

// ── drawDisplay ───────────────────────────────────────────────────────────────
//
// Display layout (128×64 px, textSize=1 → 6×8 px/char, 21 chars/row):
//
//  ┌─────── YELLOW zone (y 0–15) ────────┐
//  │ WIRED     [*] LIVE                  │  ← Row 0 y=1 : mode + live badge
//  │ P:+12.3     R:-4.6                  │  ← Row 1 y=9 : pitch & roll
//  ├─────── BLUE zone   (y 16–63) ───────┤
//  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  ← inverted box y=17–28
//  │▓  Yaw : +089.01 deg              ▓│  ← Row y=20, black-on-white
//  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
//  │  -> 192.168.1.100                   │  ← WiFi IP (WiFi mode only) y=31
//  │ ─────────────────────────────────── │  ← separator y=41
//  │ [B1] -> WIFI                        │  ← button hint y=45
//  │ [B2] -> WIRED                       │  ← button hint y=55
//  └─────────────────────────────────────┘
//
// The yellow zone is an inverted rectangle (fillRect WHITE → drawn in yellow
// on the physical display with black text on top).
// The "Yaw box" inside the blue zone is a second inverted strip — it appears
// as a bright blue band against the surrounding dark blue, making the yaw
// reading visually pop.

static uint8_t animTick = 0;  // incremented each drawDisplay() call

static void drawDisplay() {
  if (!oledOk) return;
  animTick++;
  // Live badge blinks at ~1.25 Hz: 4 calls × 100 ms = 400 ms on/off
  bool liveOn = (animTick >> 2) & 1;

  display.clearDisplay();

  // ════ YELLOW ZONE ══════════════════════════════════════════════════════════
  display.fillRect(0, 0, SCREEN_W, 16, SSD1306_WHITE);
  display.setTextColor(SSD1306_BLACK);
  display.setTextSize(1);

  if (currentMode == IDLE) {
    display.setCursor(10, 1); display.print("* InertiaLink v2 *");
    display.setCursor(14, 9); display.print("USB + WiFi   no BT");
  } else {
    // Row 0 — mode label + animated live badge
    char row0[22];
    snprintf(row0, sizeof(row0), "%-5s    %s",
             modeName(currentMode), liveOn ? "[*] LIVE" : "[ ] live");
    display.setCursor(0, 1);
    display.print(row0);

    // Row 1 — pitch & roll: most frequently glanced values
    char row1[22];
    snprintf(row1, sizeof(row1), "P:%+5.1f    R:%+5.1f", pitch, roll);
    display.setCursor(0, 9);
    display.print(row1);
  }

  // ════ BLUE ZONE ════════════════════════════════════════════════════════════
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  if (currentMode == IDLE) {
    display.setCursor(0, 18); display.print("Awaiting command...");
    display.setCursor(0, 30); display.print("[Btn1] Cycle mode");
    display.setCursor(0, 40); display.print("[Btn2] Force WIRED");
    display.setCursor(0, 52); display.print("TUI: MODE:WIFI|...");
    display.display();
    return;
  }

  // ── Inverted yaw box ────────────────────────────────────────────────────────
  // A white-filled strip in the blue zone appears as a bright cyan/blue band
  // on the physical display.  Black text on it creates strong contrast.
  display.fillRect(0, 17, SCREEN_W, 12, SSD1306_WHITE);
  display.setTextColor(SSD1306_BLACK);
  char yawBuf[22];
  snprintf(yawBuf, sizeof(yawBuf), "  Yaw : %+7.2f deg", yaw);
  display.setCursor(0, 20);
  display.print(yawBuf);

  // Return to normal white text for the rest of the blue zone
  display.setTextColor(SSD1306_WHITE);

  // WiFi target IP (only in WIFI mode)
  if (currentMode == WIFI && wifiTargetIP.length() > 0) {
    display.setCursor(0, 31);
    display.print("-> ");
    display.print(wifiTargetIP.substring(0, 18).c_str());
  }

  // Thin separator
  display.drawFastHLine(0, 41, SCREEN_W, SSD1306_WHITE);

  // Contextual button hints
  display.setCursor(0, 45);
  display.print(currentMode == WIRED ? "[B1] -> WIFI " : "[B1] -> WIRED");
  display.setCursor(0, 55);
  display.print("[B2] -> WIRED");

  display.display();
}

// splashOLED: used during boot, WiFi connecting, error screens.
// Yellow zone = inverted title / blue zone = up to 3 info lines.
static void splashOLED(const char *title,
                       const char *line1 = nullptr,
                       const char *line2 = nullptr,
                       const char *line3 = nullptr) {
  if (!oledOk) return;
  display.clearDisplay();

  display.fillRect(0, 0, SCREEN_W, 16, SSD1306_WHITE);
  display.setTextColor(SSD1306_BLACK);
  display.setTextSize(1);
  display.setCursor(2, 4);
  display.print(title);

  display.setTextColor(SSD1306_WHITE);
  if (line1) { display.setCursor(0, 20); display.print(line1); }
  if (line2) { display.setCursor(0, 32); display.print(line2); }
  if (line3) { display.setCursor(0, 44); display.print(line3); }

  display.display();
}

// ── Command parser ────────────────────────────────────────────────────────────
static void parseCommand(const char *cmd) {
  if (strcmp(cmd, "MODE:USB") == 0) {
    currentMode = WIRED;
    Serial.println("[ESP] Mode -> WIRED");

  } else if (strcmp(cmd, "MODE:IDLE") == 0) {
    currentMode = IDLE;
    Serial.println("[ESP] Mode -> IDLE");

  } else if (strncmp(cmd, "MODE:WIFI", 9) == 0) {
    // Expected: MODE:WIFI|<ssid>|<password>|<host_ip>
    String s(cmd);
    int a = s.indexOf('|');
    int b = s.indexOf('|', a + 1);
    int c = s.indexOf('|', b + 1);

    if (a > 0 && b > 0 && c > 0) {
      String ssid = s.substring(a + 1, b);
      String pass = s.substring(b + 1, c);
      wifiTargetIP = s.substring(c + 1);

      Serial.println("[ESP] WiFi -> " + ssid);
      WiFi.begin(ssid.c_str(), pass.c_str());

      for (int i = 0; i < 20 && WiFi.status() != WL_CONNECTED; i++) {
        char buf[22];
        snprintf(buf, sizeof(buf), "Attempt %d/20...", i + 1);
        splashOLED("Connecting WiFi", ssid.c_str(), buf);
        delay(500);
      }

      if (WiFi.status() == WL_CONNECTED) {
        wifiReady   = true;
        currentMode = WIFI;
        Serial.println("[ESP] WiFi OK -> " + wifiTargetIP);
      } else {
        wifiReady   = false;
        currentMode = WIRED;
        Serial.println("[ESP] WiFi failed -> WIRED");
        splashOLED("WiFi failed!", "Falling back to", "WIRED mode...");
        delay(1500);
      }
    }
  }

  drawDisplay();
}

// ── Non-blocking serial poll ───────────────────────────────────────────────────
// Reads all available bytes into cmdBuf one char at a time.
// Calls parseCommand() only when a complete '\n'-terminated line arrives.
// This never stalls the loop — Serial.readStringUntil() would block 1 s on
// a partial line.
static void pollSerial() {
  while (Serial.available()) {
    char c = static_cast<char>(Serial.read());
    if (c == '\n' || c == '\r') {
      if (cmdPos > 0) {
        cmdBuf[cmdPos] = '\0';
        parseCommand(cmdBuf);
        cmdPos = 0;
      }
    } else if (cmdPos < 127) {
      cmdBuf[cmdPos++] = c;
    }
  }
}

// ── Button handler ────────────────────────────────────────────────────────────
static void handleButtons() {
  unsigned long now = millis();

  // Btn1: WIRED → WIFI (if credentials ready) / WIFI or IDLE → WIRED
  if (digitalRead(BTN_CYCLE_PIN) == LOW && (now - lastBtn1Ms) > DEBOUNCE_MS) {
    lastBtn1Ms = now;
    if (currentMode == WIRED) {
      if (wifiReady) {
        currentMode = WIFI;
        Serial.println("[ESP] Btn1 -> WIFI");
      } else {
        splashOLED("No WiFi config!", "Send from TUI:", "MODE:WIFI|...");
        delay(1500);
      }
    } else {
      currentMode = WIRED;
      Serial.println("[ESP] Btn1 -> WIRED");
    }
    drawDisplay();
  }

  // Btn2: force WIRED from any mode
  if (digitalRead(BTN_RESET_PIN) == LOW && (now - lastBtn2Ms) > DEBOUNCE_MS) {
    lastBtn2Ms  = now;
    currentMode = WIRED;
    Serial.println("[ESP] Btn2 -> WIRED");
    drawDisplay();
  }
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  pinMode(BTN_CYCLE_PIN, INPUT_PULLUP);
  pinMode(BTN_RESET_PIN, INPUT_PULLUP);

  // 400 kHz I2C: reduces display.display() from ~92 ms to ~23 ms per frame.
  // Both MPU6050 and SSD1306 support 400 kHz.
  Wire.begin(21, 22);
  Wire.setClock(400000);

  // OLED — non-fatal if absent
  oledOk = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  if (!oledOk)
    Serial.println("[ESP] OLED absent — headless");

  splashOLED("InertiaLink v2", "Booting...", "USB + WiFi only");

  // MPU6050 — fatal if absent
  if (mpu.begin() != 0) {
    splashOLED("MPU6050 Error!", "Check I2C wiring",
               "SDA=21  SCL=22");
    Serial.println("[ESP] MPU6050 init failed — halting");
    while (true) delay(10);
  }

  splashOLED("Calibrating IMU", "Keep device still");
  delay(1000);
  mpu.calcOffsets();

  // Signal the PC: firmware is ready and already in WIRED mode.
  Serial.println("[ESP] Ready — WIRED 100 Hz   accel_z enabled");
  drawDisplay();
}

// ── Main loop ─────────────────────────────────────────────────────────────────
static unsigned long lastDispMs  = 0;
static const unsigned long DISP_INTERVAL_MS = 100; // ~10 Hz OLED refresh

void loop() {
  handleButtons();
  pollSerial();  // non-blocking; builds cmdBuf, calls parseCommand on '\n'

  if (currentMode != IDLE) {
    mpu.update();
    pitch  = mpu.getAngleX();
    roll   = mpu.getAngleY();
    yaw    = mpu.getAngleZ();
    accelZ = mpu.getAccZ();   // raw Z acceleration in g
                               // at rest ≈ 1.0 g (gravity); spikes on impact

    // 4-field CSV — accel_z is the 4th field that ALL desktop apps depend on:
    //   viz.cpp:326    abs(az) >= 0.05  → pen-contact, draws the 2-D trail
    //   viz.cpp:249    |az–prev| > 0.5  → stroke start, resets cube anchor
    //   data_collector |az–prev| > 0.5  → wake-on-impact
    //   decoder        |az–prev| > 0.5  → wake-on-impact
    char payload[40];
    snprintf(payload, sizeof(payload),
             "%.2f,%.2f,%.2f,%.2f\n", pitch, roll, yaw, accelZ);

    if (currentMode == WIRED) {
      Serial.print(payload);
    } else {
      udp.beginPacket(wifiTargetIP.c_str(), wifiTargetPort);
      udp.print(payload);
      udp.endPacket();
    }

    // Throttle OLED to ~10 Hz so the 23 ms I2C transfer doesn't eat into
    // every loop iteration.
    unsigned long now = millis();
    if (now - lastDispMs >= DISP_INTERVAL_MS) {
      lastDispMs = now;
      drawDisplay();
    }
  }

  delay(10); // 100 Hz cap
}
