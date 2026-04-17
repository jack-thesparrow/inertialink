// InertiaLink ESP32 Firmware — USB + SoftAP WiFi (no router required)
//
// Default boot mode: WIRED.  The desktop apps open the port and expect
// data immediately; no MODE:USB command needed before streaming begins.
//
// WiFi transport: ESP32 runs as a SoftAP — the laptop connects directly to
// the pen's own network.  No router, no credentials to flash, no host IP to
// configure.  The pen always broadcasts UDP to 192.168.4.255:5005 so any
// connected laptop receives it without knowing the pen's DHCP assignments.
//
//   AP SSID     : InertiaLink
//   AP password : inertia123
//   ESP32 IP    : 192.168.4.1  (fixed, default SoftAP address)
//   Laptop IP   : 192.168.4.2  (first DHCP lease — assigned automatically)
//
// Wiring:
//   MPU6050   SDA→GPIO 21   SCL→GPIO 22   I2C address 0x68
//   SSD1306   SDA→GPIO 21   SCL→GPIO 22   I2C address 0x3C   (no conflict)
//   Btn1      GPIO 14 ↔ GND   —  toggles WIRED ↔ WIFI
//   Btn2      GPIO 27 ↔ GND   —  toggles IMU view ↔ Prediction view
//
// Serial commands at 115200 baud (TUI or any terminal):
//   MODE:USB    switch to / stay in WIRED mode
//   MODE:WIFI   switch to SoftAP WiFi mode (AP is always running)
//   MODE:IDLE   pause streaming
//
// Payload format (sent in WIRED/WIFI modes at 100 Hz):
//   ax,ay,az,gx,gy,gz\n   (floats: g-force + deg/s)

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Arduino.h>
#include <MPU6050_light.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

// ── SoftAP credentials (hardcoded — no NVS needed) ───────────────────────────
// #define so string-literal concatenation ("prefix" AP_SSID) compiles.
#define AP_SSID "InertiaLink"
#define AP_PASS "inertia123"
// Broadcast to all clients on the AP subnet — no laptop IP config required.
static constexpr const char *UDP_TARGET  = "192.168.4.255";
static constexpr int UDP_PORT_DECODER    = 5005; // decoder
static constexpr int UDP_PORT_VISUALIZER = 5006; // visualizer

// ── OLED
// ──────────────────────────────────────────────────────────────────────
#define SCREEN_W 128
#define SCREEN_H 64
#define OLED_ADDR 0x3C
static Adafruit_SSD1306 display(SCREEN_W, SCREEN_H, &Wire, /*reset=*/-1);
static bool oledOk = false;

// ── Push buttons (active-LOW, internal pull-up)
// ───────────────────────────────
#define BTN_CYCLE_PIN 14 // toggle WIRED ↔ WIFI
#define BTN_RESET_PIN 27 // toggle IMU ↔ Prediction view
#define DEBOUNCE_MS 200
static unsigned long lastBtn1Ms = 0, lastBtn2Ms = 0;

// ── Non-blocking serial command buffer
// ────────────────────────────────────────
static char cmdBuf[128];
static int cmdPos = 0;

// ── Mode & radio state
// ────────────────────────────────────────────────────────
enum Mode { WIRED, WIFI, IDLE };

static Mode currentMode = WIRED; // boot default: stream over USB immediately
static bool apReady = false;     // set true once SoftAP is up

static MPU6050 mpu(Wire);
static WiFiUDP udp;

// Live IMU values (globals so drawDisplay can read them without passing args)
static float imu_ax = 0.0f, imu_ay = 0.0f, imu_az = 0.0f;
static float imu_gx = 0.0f, imu_gy = 0.0f, imu_gz = 0.0f;

// ── Display view & prediction state ──────────────────────────────────────────
enum DisplayView { IMU_VIEW, PRED_VIEW };
static DisplayView displayView = IMU_VIEW;

// Updated by STATE:xxx / PRED:char:conf feedback from the host decoder.
static char predStateStr[20] = "IDLE";
static char predChar[8]      = "";
static int  predConf         = 0;

// UDP socket that receives STATE/PRED feedback from the host in Wi-Fi mode.
static WiFiUDP feedbackUdp;
static constexpr int UDP_FEEDBACK_PORT = 5007;

// ── OLED helpers
// ──────────────────────────────────────────────────────────────
static const char *modeName(Mode m) {
  switch (m) {
  case WIRED:
    return "WIRED";
  case WIFI:
    return "WIFI ";
  default:
    return "IDLE ";
  }
}

// ── drawDisplay
// ───────────────────────────────────────────────────────────────
//
// Panel colour zones (fixed by hardware — not configurable):
//   y  0–15  → YELLOW   (16 px, 2 text rows)
//   y 16–63  → BLUE     (48 px, 6 text rows)
//
//  ┌──────── YELLOW zone ────────────────────────────────────────┐
//  │ ▓WIRED▓   WIFI    IDLE                                     │ y=0 tabs
//  │ [*] LIVE   AP:InertiaLink  /  USB 115200                   │ y=9
//  ├──────── BLUE zone ──────────────────────────────────────────┤
//  │ AX:+1.23  AY:-0.45                                         │ y=18
//  │ AZ:+0.98  GZ: +89.0                                        │ y=27
//  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ inverted band
//  │▓ GX:+12.3  GY: -5.6                                      ▓│ y=37
//  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
//  │──────────────────────────────────────────────────────────── │ sep y=47
//  │ B1:->WIFI   B2:WIRED                                       │ y=55
//  └─────────────────────────────────────────────────────────────┘

// ── drawDisplay
// ───────────────────────────────────────────────────────────────
//
// Panel colour zones (fixed by hardware — not configurable):
//   y  0–15  → YELLOW   (16 px, 2 text rows)
//   y 16–63  → BLUE     (48 px, 6 text rows)
//
// Tab bar (Row 0): WIRED & WIFI show transport mode; PRED shows view mode.
// Multiple tabs can be active simultaneously (e.g. WIRED + PRED both lit).
//
//  IMU view:
//  ┌──────── YELLOW ────────────────────────────────────────────┐
//  │ ▓WIRED▓   WIFI    PRED                                    │ y=0
//  │ [*] USB 115200 baud                                       │ y=9
//  ├──────── BLUE ──────────────────────────────────────────────┤
//  │ AX:+1.23  AY:-0.45                                        │ y=18
//  │ AZ:+0.98  GZ: +89.0                                       │ y=27
//  │▓▓ GX:+12.3  GY: -5.6 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ y=37
//  │─────────────────────────────────────────────────────────── │ y=47
//  │ B1:->WIFI  B2:PRED                                        │ y=55
//  └────────────────────────────────────────────────────────────┘
//
//  Prediction view (Btn2 active):
//  ┌──────── YELLOW ────────────────────────────────────────────┐
//  │  WIRED   ▓WIFI▓  ▓PRED▓                                  │ y=0
//  │ [*] AP:InertiaLink                                        │ y=9
//  ├──────── BLUE ──────────────────────────────────────────────┤
//  │ WRITING                                                   │ y=18
//  │         >> "3"          (TextSize=2)                      │ y=28
//  │                                                           │
//  │             CONF:  87%                                    │ y=46
//  │─────────────────────────────────────────────────────────── │ y=54
//  │ B1:->WIRED B2:IMU                                         │ y=55
//  └────────────────────────────────────────────────────────────┘

static uint8_t animTick = 0;

static void drawDisplay() {
  if (!oledOk)
    return;
  animTick++;
  bool liveOn = (animTick >> 2) & 1; // blinks ~1.25 Hz

  display.clearDisplay();
  display.setTextSize(1);

  // ════ YELLOW ZONE ══════════════════════════════════════════════════════════
  display.fillRect(0, 0, SCREEN_W, 16, SSD1306_WHITE);

  // ── Row 0: WIRED / WIFI track transport; PRED tracks view ─────────────────
  static const struct {
    const char *label;
    uint8_t     x;
  } TABS[3] = {{"WIRED", 2}, {"WIFI", 50}, {"PRED", 96}};
  bool tabActive[3] = {currentMode == WIRED, currentMode == WIFI,
                       displayView == PRED_VIEW};
  for (int i = 0; i < 3; i++) {
    if (tabActive[i]) {
      uint8_t w = static_cast<uint8_t>(strlen(TABS[i].label) * 6);
      display.fillRect(TABS[i].x - 1, 0, w + 2, 9, SSD1306_BLACK);
      display.setTextColor(SSD1306_WHITE);
    } else {
      display.setTextColor(SSD1306_BLACK);
    }
    display.setCursor(TABS[i].x, 1);
    display.print(TABS[i].label);
  }

  // ── Row 1: live badge + transport detail ───────────────────────────────────
  display.setTextColor(SSD1306_BLACK);
  display.setCursor(0, 9);
  if (currentMode == WIFI) {
    char r[22];
    snprintf(r, sizeof(r), "%s AP:%-11s", liveOn ? "[*]" : "[ ]", AP_SSID);
    display.print(r);
  } else if (currentMode == WIRED) {
    char r[22];
    snprintf(r, sizeof(r), "%s USB 115200 baud", liveOn ? "[*]" : "[ ]");
    display.print(r);
  } else {
    display.setCursor(22, 9);
    display.print("--- paused ---");
  }

  // ════ BLUE ZONE ════════════════════════════════════════════════════════════
  display.setTextColor(SSD1306_WHITE);

  if (displayView == PRED_VIEW) {
    // ── Prediction screen ─────────────────────────────────────────────────────
    char stateLabel[21];
    snprintf(stateLabel, sizeof(stateLabel), "%-20s", predStateStr);
    display.setCursor(0, 18);
    display.print(stateLabel);

    // Large prediction centred at TextSize=2 (16 px tall), capped at 3 chars
    display.setTextSize(2);
    if (predChar[0] != '\0') {
      char disp[4];
      strncpy(disp, predChar, 3);
      disp[3] = '\0';
      char predLine[12];
      snprintf(predLine, sizeof(predLine), ">> \"%s\"", disp);
      int px = (SCREEN_W - static_cast<int>(strlen(predLine)) * 12) / 2;
      display.setCursor(px < 0 ? 0 : px, 28);
      display.print(predLine);
    } else {
      display.setCursor(22, 28); // "WAITING": 7 chars × 12 px = 84 px centred
      display.print("WAITING");
    }
    display.setTextSize(1);

    char confLine[14];
    if (predChar[0] != '\0')
      snprintf(confLine, sizeof(confLine), "CONF:  %3d%%", predConf);
    else
      snprintf(confLine, sizeof(confLine), "CONF:   --%%");
    display.setCursor(28, 46);
    display.print(confLine);

    display.drawFastHLine(0, 54, SCREEN_W, SSD1306_WHITE);
    display.setCursor(0, 55);
    display.print(currentMode == WIRED ? "B1:->WIFI  B2:IMU " : "B1:->WIRED B2:IMU ");

  } else {
    // ── IMU screen ────────────────────────────────────────────────────────────
    if (currentMode == IDLE) {
      display.setCursor(0, 19); display.print("[B1] WIFI / WIRED");
      display.setCursor(0, 29); display.print("[B2] Pred view");
      display.setCursor(0, 41); display.print("AP:  " AP_SSID);
      display.setCursor(0, 51); display.print("pw:  " AP_PASS);
      display.display();
      return;
    }

    char buf[22];
    snprintf(buf, sizeof(buf), "AX:%+5.2f  AY:%+5.2f", imu_ax, imu_ay);
    display.setCursor(0, 18);
    display.print(buf);

    snprintf(buf, sizeof(buf), "AZ:%+5.2f  GZ:%+6.1f", imu_az, imu_gz);
    display.setCursor(0, 27);
    display.print(buf);

    display.fillRect(0, 36, SCREEN_W, 9, SSD1306_WHITE);
    display.setTextColor(SSD1306_BLACK);
    snprintf(buf, sizeof(buf), "GX:%+5.1f  GY:%+5.1f", imu_gx, imu_gy);
    display.setCursor(0, 37);
    display.print(buf);

    display.setTextColor(SSD1306_WHITE);
    display.drawFastHLine(0, 47, SCREEN_W, SSD1306_WHITE);
    display.setCursor(0, 55);
    display.print(currentMode == WIRED ? "B1:->WIFI  B2:PRED" : "B1:->WIRED B2:PRED");
  }

  display.display();
}

// splashOLED: used during boot and error screens.
static void splashOLED(const char *title, const char *line1 = nullptr,
                       const char *line2 = nullptr,
                       const char *line3 = nullptr) {
  if (!oledOk)
    return;
  display.clearDisplay();

  display.fillRect(0, 0, SCREEN_W, 16, SSD1306_WHITE);
  display.setTextColor(SSD1306_BLACK);
  display.setTextSize(1);
  display.setCursor(2, 4);
  display.print(title);

  display.setTextColor(SSD1306_WHITE);
  if (line1) {
    display.setCursor(0, 20);
    display.print(line1);
  }
  if (line2) {
    display.setCursor(0, 32);
    display.print(line2);
  }
  if (line3) {
    display.setCursor(0, 44);
    display.print(line3);
  }

  display.display();
}

// ── Command parser
// ────────────────────────────────────────────────────────────
static void parseCommand(const char *cmd) {
  if (strcmp(cmd, "MODE:USB") == 0) {
    currentMode = WIRED;
    Serial.println("[ESP] Mode -> WIRED");

  } else if (strcmp(cmd, "MODE:IDLE") == 0) {
    currentMode = IDLE;
    Serial.println("[ESP] Mode -> IDLE");

  } else if (strcmp(cmd, "MODE:WIFI") == 0) {
    if (apReady) {
      currentMode = WIFI;
      Serial.println("[ESP] Mode -> WIFI (SoftAP: " AP_SSID ")");
      Serial.println("[ESP] WiFi OK");
    } else {
      Serial.println("[ESP] WiFi failed -> SoftAP not ready");
    }

  } else if (strncmp(cmd, "STATE:", 6) == 0) {
    strncpy(predStateStr, cmd + 6, sizeof(predStateStr) - 1);
    predStateStr[sizeof(predStateStr) - 1] = '\0';
    if (displayView == PRED_VIEW) drawDisplay();
    return; // skip the drawDisplay() at the end

  } else if (strncmp(cmd, "PRED:", 5) == 0) {
    // Format: PRED:<char>:<confidence>   e.g. "PRED:3:87"  or "PRED::0"
    const char *rest  = cmd + 5;
    const char *colon = strchr(rest, ':');
    if (colon) {
      int charLen = static_cast<int>(colon - rest);
      if (charLen > 7) charLen = 7;
      strncpy(predChar, rest, charLen);
      predChar[charLen] = '\0';
      predConf = atoi(colon + 1);
    }
    if (displayView == PRED_VIEW) drawDisplay();
    return;
  }

  drawDisplay();
}

// ── Non-blocking serial poll
// ───────────────────────────────────────────────────
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

// ── Feedback UDP poll (Wi-Fi mode — receives STATE/PRED from host decoder)
// ────────
static void pollFeedbackUdp() {
  int n = feedbackUdp.parsePacket();
  if (n <= 0)
    return;
  char buf[64];
  int len = feedbackUdp.read(buf, sizeof(buf) - 1);
  if (len <= 0)
    return;
  while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
    --len;
  buf[len] = '\0';
  if (len > 0)
    parseCommand(buf);
}

// ── Button handler
// ────────────────────────────────────────────────────────────
static void handleButtons() {
  unsigned long now = millis();

  // Btn1: toggle WIRED ↔ WIFI
  if (digitalRead(BTN_CYCLE_PIN) == LOW && (now - lastBtn1Ms) > DEBOUNCE_MS) {
    lastBtn1Ms = now;
    if (currentMode != WIFI) {
      if (apReady) {
        currentMode = WIFI;
        Serial.println("[ESP] Btn1 -> WIFI");
      } else {
        splashOLED("SoftAP error!", "AP failed to start", "Try reboot");
        delay(1500);
      }
    } else {
      currentMode = WIRED;
      Serial.println("[ESP] Btn1 -> WIRED");
    }
    drawDisplay();
  }

  // Btn2: toggle IMU view ↔ Prediction view
  if (digitalRead(BTN_RESET_PIN) == LOW && (now - lastBtn2Ms) > DEBOUNCE_MS) {
    lastBtn2Ms = now;
    displayView = (displayView == IMU_VIEW) ? PRED_VIEW : IMU_VIEW;
    Serial.println(displayView == PRED_VIEW ? "[ESP] Btn2 -> PRED view"
                                            : "[ESP] Btn2 -> IMU view");
    drawDisplay();
  }
}

// ── Setup
// ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  pinMode(BTN_CYCLE_PIN, INPUT_PULLUP);
  pinMode(BTN_RESET_PIN, INPUT_PULLUP);

  // 400 kHz I2C: reduces display.display() from ~92 ms to ~23 ms per frame.
  Wire.begin(21, 22);
  Wire.setClock(400000);

  // OLED — non-fatal if absent
  oledOk = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  if (!oledOk)
    Serial.println("[ESP] OLED absent — headless");

  splashOLED("InertiaLink v2", "Booting...", "USB + SoftAP WiFi");

  // MPU6050 — fatal if absent
  if (mpu.begin() != 0) {
    splashOLED("MPU6050 Error!", "Check I2C wiring", "SDA=21  SCL=22");
    Serial.println("[ESP] MPU6050 init failed — halting");
    while (true)
      delay(10);
  }

  splashOLED("Calibrating IMU", "Keep device still");
  delay(1000);
  mpu.calcOffsets();

  // Start SoftAP — always on, no router required.
  // WiFi.softAP() returns immediately; the AP is up within a few ms.
  splashOLED("Starting SoftAP", AP_SSID, "pw: " AP_PASS);
  WiFi.mode(WIFI_AP);
  if (WiFi.softAP(AP_SSID, AP_PASS)) {
    apReady = true;
    feedbackUdp.begin(UDP_FEEDBACK_PORT);
    Serial.println("[ESP] SoftAP up: " AP_SSID " -> 192.168.4.1");
    Serial.println("[ESP] Feedback UDP listening on port " +
                   String(UDP_FEEDBACK_PORT));
  } else {
    Serial.println("[ESP] SoftAP failed — WiFi mode unavailable");
    splashOLED("SoftAP failed!", "WIRED only");
    delay(1500);
  }

  Serial.println("[ESP] Ready");
  drawDisplay();
}

// ── Main loop
// ─────────────────────────────────────────────────────────────────
static unsigned long lastDispMs = 0;
static const unsigned long DISP_INTERVAL_MS = 100; // ~10 Hz OLED refresh

void loop() {
  handleButtons();
  pollSerial();
  if (currentMode == WIFI)
    pollFeedbackUdp();

  if (currentMode != IDLE) {
    mpu.update();

    float ax = mpu.getAccX();
    float ay = mpu.getAccY();
    float az = mpu.getAccZ();
    float gx = mpu.getGyroX();
    float gy = mpu.getGyroY();
    float gz = mpu.getGyroZ();

    imu_ax = ax;
    imu_ay = ay;
    imu_az = az;
    imu_gx = gx;
    imu_gy = gy;
    imu_gz = gz;

    char payload[80];
    snprintf(payload, sizeof(payload), "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", ax,
             ay, az, gx, gy, gz);

    if (currentMode == WIRED) {
      Serial.print(payload);
    } else {
      // Broadcast to both ports so the decoder and visualizer each
      // receive every packet independently (no SO_REUSEPORT load-balancing).
      udp.beginPacket(UDP_TARGET, UDP_PORT_DECODER);
      udp.print(payload);
      udp.endPacket();
      udp.beginPacket(UDP_TARGET, UDP_PORT_VISUALIZER);
      udp.print(payload);
      udp.endPacket();
    }

    unsigned long now = millis();
    if (now - lastDispMs >= DISP_INTERVAL_MS) {
      lastDispMs = now;
      drawDisplay();
    }
  }

  delay(10); // 100 Hz cap
}
