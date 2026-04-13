#include "stb_easy_font.h"

#include "pen/viz.hpp"
#include <glad/glad.h>
// glad before GLFW
#include <GLFW/glfw3.h>
#include <cstdio>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace pen {

// --- CUBE SHADERS ---
const char *vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
out vec3 vertexColor;
uniform mat4 MVP;
void main() {
    gl_Position = MVP * vec4(aPos, 1.0);
    vertexColor = aColor;
}
)";

const char *fragmentShaderSource = R"(
#version 330 core
in vec3 vertexColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vertexColor, 1.0);
}
)";

// --- TRAIL SHADERS (Neon Orange) ---
const char *trailVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 MVP;
void main() {
    gl_Position = MVP * vec4(aPos, 1.0);
}
)";

const char *trailFragmentShader = R"(
#version 330 core
uniform vec3 uColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(uColor, 1.0);
}
)";

Visualizer::Visualizer(int width, int height, const char *title) {
  if (!glfwInit()) {
    std::cerr << "[Viz] FATAL: glfwInit() failed.\n"
              << "      Is a display server running? (check $DISPLAY / "
                 "$WAYLAND_DISPLAY)\n";
    exit(-1);
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 4);

  window = glfwCreateWindow(width, height, title, nullptr, nullptr);
  if (!window) {
    // Retry without MSAA — some Mesa / Intel Arc drivers reject multisampling
    glfwWindowHint(GLFW_SAMPLES, 0);
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
  }
  if (!window) {
    std::cerr << "[Viz] FATAL: glfwCreateWindow() failed.\n"
              << "      OpenGL 3.3 Core Profile may not be supported by this "
                 "driver.\n"
              << "      Try: glxinfo | grep 'OpenGL version'\n";
    glfwTerminate();
    exit(-1);
  }

  glfwMakeContextCurrent(window);
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "[Viz] FATAL: gladLoadGLLoader() failed — GL function "
                 "pointers unavailable.\n";
    glfwTerminate();
    exit(-1);
  }

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_MULTISAMPLE);

  initOpenGL();
  setupGeometry();
}

Visualizer::~Visualizer() { glfwTerminate(); }

bool Visualizer::isOpen() const { return !glfwWindowShouldClose(window); }

void Visualizer::update() {
  glfwSwapBuffers(window);
  glfwPollEvents();
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

  // Press 'C' to clear the canvas!
  if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS)
    strokeTrail.clear();
}

void Visualizer::initOpenGL() {
  auto compileShader = [](GLenum type, const char *src) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    return shader;
  };

  // Compile Cube Shader
  unsigned int vs = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
  unsigned int fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
  shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vs);
  glAttachShader(shaderProgram, fs);
  glLinkProgram(shaderProgram);
  glDeleteShader(vs);
  glDeleteShader(fs);

  // Compile Trail Shader
  unsigned int tvs = compileShader(GL_VERTEX_SHADER, trailVertexShader);
  unsigned int tfs = compileShader(GL_FRAGMENT_SHADER, trailFragmentShader);
  trailShaderProgram = glCreateProgram();
  glAttachShader(trailShaderProgram, tvs);
  glAttachShader(trailShaderProgram, tfs);
  glLinkProgram(trailShaderProgram);
  glDeleteShader(tvs);
  glDeleteShader(tfs);
}

void Visualizer::setupGeometry() {
  // --- Cube Geometry Setup ---
  float vertices[] = {
      -0.25f, -0.25f, 0.25f,  0.0f,   0.0f,   1.0f,   0.25f,  -0.25f, 0.25f,
      0.0f,   1.0f,   1.0f,   0.25f,  0.25f,  0.25f,  0.0f,   1.0f,   1.0f,
      -0.25f, 0.25f,  0.25f,  0.0f,   0.0f,   1.0f,   -0.25f, -0.25f, -0.25f,
      1.0f,   0.0f,   0.0f,   0.25f,  -0.25f, -0.25f, 1.0f,   1.0f,   0.0f,
      0.25f,  0.25f,  -0.25f, 1.0f,   1.0f,   0.0f,   -0.25f, 0.25f,  -0.25f,
      1.0f,   0.0f,   0.0f,   -0.25f, 0.25f,  -0.25f, 0.0f,   0.8f,   0.0f,
      0.25f,  0.25f,  -0.25f, 0.3f,   1.0f,   0.3f,   0.25f,  0.25f,  0.25f,
      0.3f,   1.0f,   0.3f,   -0.25f, 0.25f,  0.25f,  0.0f,   0.8f,   0.0f,
      -0.25f, -0.25f, -0.25f, 0.5f,   0.0f,   0.5f,   0.25f,  -0.25f, -0.25f,
      1.0f,   0.0f,   1.0f,   0.25f,  -0.25f, 0.25f,  1.0f,   0.0f,   1.0f,
      -0.25f, -0.25f, 0.25f,  0.5f,   0.0f,   0.5f,   0.25f,  -0.25f, -0.25f,
      1.0f,   0.5f,   0.0f,   0.25f,  -0.25f, 0.25f,  1.0f,   0.7f,   0.0f,
      0.25f,  0.25f,  0.25f,  1.0f,   0.7f,   0.0f,   0.25f,  0.25f,  -0.25f,
      1.0f,   0.5f,   0.0f,   -0.25f, -0.25f, -0.25f, 0.0f,   0.5f,   0.5f,
      -0.25f, -0.25f, 0.25f,  0.0f,   0.8f,   0.8f,   -0.25f, 0.25f,  0.25f,
      0.0f,   0.8f,   0.8f,   -0.25f, 0.25f,  -0.25f, 0.0f,   0.5f,   0.5f,
  };
  unsigned int indices[] = {0,  1,  2,  0,  2,  3,  4,  6,  5,  4,  7,  6,
                            8,  9,  10, 8,  10, 11, 12, 14, 13, 12, 15, 14,
                            16, 17, 18, 16, 18, 19, 20, 22, 21, 20, 23, 22};
  unsigned int edges[] = {0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6,
                          6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7};

  glGenVertexArrays(1, &faceVAO);
  glGenBuffers(1, &faceVBO);
  glGenBuffers(1, &faceEBO);
  glBindVertexArray(faceVAO);
  glBindBuffer(GL_ARRAY_BUFFER, faceVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faceEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glGenVertexArrays(1, &edgeVAO);
  glGenBuffers(1, &edgeVBO);
  glGenBuffers(1, &edgeEBO);
  glBindVertexArray(edgeVAO);
  glBindBuffer(GL_ARRAY_BUFFER, edgeVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(edges), edges, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // --- Trail Geometry Setup ---
  glGenVertexArrays(1, &trailVAO);
  glGenBuffers(1, &trailVBO);
  glBindVertexArray(trailVAO);
  glBindBuffer(GL_ARRAY_BUFFER, trailVBO);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);
  glEnableVertexAttribArray(0);

  // --- Text Geometry Setup (dynamic; filled each frame by renderWord) ---
  glGenVertexArrays(1, &textVAO);
  glGenBuffers(1, &textVBO);
  glBindVertexArray(textVAO);
  glBindBuffer(GL_ARRAY_BUFFER, textVBO);
  // 3 floats per vertex (x, y, z); stride = 3 * sizeof(float)
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
}

void Visualizer::drawCube(const IMUData &imu) {
  // ── HUD polling (every 60 frames ≈ 600 ms at 100 Hz) ────────────────────
  // mock_esp32.py  writes /tmp/inertialink_word  before each stream.
  // decoder_main   writes /tmp/inertialink_mode  at each state transition.
  if (++frameCount % 60 == 0) {
    auto readTmpFile = [](const char *path, std::string &out) {
      std::FILE *f = std::fopen(path, "r");
      if (!f)
        return;
      char buf[64] = {};
      std::fgets(buf, sizeof(buf), f);
      std::fclose(f);
      out = buf;
      while (!out.empty() && (out.back() == '\n' || out.back() == '\r'))
        out.pop_back();
    };

    std::string word, mode;
    readTmpFile("/tmp/inertialink_word", word);
    readTmpFile("/tmp/inertialink_mode", mode);

    if (!word.empty() && word != activeWord) {
      activeWord = word;
      std::string title = "Inertialink Visualizer  —  " + activeWord;
      glfwSetWindowTitle(window, title.c_str());
    }
    activeMode = mode;
  }

  // ── Constants ────────────────────────────────────────────────────────────
  constexpr float DT = 0.01f; // 100 Hz
  constexpr float DEG2RAD = static_cast<float>(M_PI / 180.0);
  constexpr float G_MPS2 = 9.80665f; // m/s² per g

  // ── Tilt compensation (cube rotation only) ──────────────────────────────
  // Calibration showed tilt comp causes 598% cross-axis bleed on the 2D
  // canvas, so it is ONLY used for the 3D cube rotation display.
  // The canvas uses raw gyro axes (GZ=horizontal, GY=vertical) directly.
  const float TILT_RAD = pen::Defaults::tiltAngleDeg * DEG2RAD;
  const float cos_t = std::cos(TILT_RAD);
  const float sin_t = std::sin(TILT_RAD);

  float gx_comp = imu.gx * cos_t + imu.gz * sin_t;  // pitch (cube only)
  float gz_comp = -imu.gx * sin_t + imu.gz * cos_t; // roll  (cube only)
  float gy_comp = imu.gy;                           // yaw   (cube only)

  // ── Complementary filter: estimate gravity (raw sensor frame) ────────
  constexpr float GRAV_ALPHA = 0.98f;
  gravity.x = GRAV_ALPHA * gravity.x + (1.0f - GRAV_ALPHA) * imu.ax;
  gravity.y = GRAV_ALPHA * gravity.y + (1.0f - GRAV_ALPHA) * imu.ay;
  gravity.z = GRAV_ALPHA * gravity.z + (1.0f - GRAV_ALPHA) * imu.az;

  // Dynamic acceleration = raw minus gravity estimate (in m/s²)
  float dyn_ax = (imu.ax - gravity.x) * G_MPS2;
  float dyn_ay = (imu.ay - gravity.y) * G_MPS2;
  float dyn_az = (imu.az - gravity.z) * G_MPS2;

  // ── Pen activity detection ────────────────────────────────────────────────
  // Two complementary signals:
  //   1. Jerk  = rate of change of total accel magnitude (detects taps/impacts)
  //   2. Gyro magnitude = angular velocity (detects pen rotation during writing)
  // Either signal above its threshold → pen is WRITING.
  float curAccelMag =
      std::sqrt(imu.ax * imu.ax + imu.ay * imu.ay + imu.az * imu.az);
  float jerk = std::abs(curAccelMag - prevAccelMag) / DT; // g/s
  // Smooth jerk to avoid single-frame noise spikes
  constexpr float JERK_SMOOTH = 0.3f;
  float smoothJerk = JERK_SMOOTH * jerk + (1.0f - JERK_SMOOTH) * prevJerk;
  prevJerk = smoothJerk;
  prevAccelMag = curAccelMag;

  // Gyroscope angular velocity magnitude (deg/s)
  float gyroMag =
      std::sqrt(imu.gx * imu.gx + imu.gy * imu.gy + imu.gz * imu.gz);

  // Thresholds — tuned for normal handwriting on paper, not violent shaking.
  // Jerk of ~5-15 g/s is typical during gentle pen strokes.
  // Gyro of ~10-50 deg/s is typical wrist rotation while writing.
  constexpr float JERK_IMPACT_THRESHOLD = 12.0f;  // g/s — pen touches paper
  constexpr float JERK_QUIET_THRESHOLD = 2.0f;    // g/s — pen is still
  constexpr float GYRO_ACTIVE_THRESHOLD = 15.0f;  // deg/s — pen is moving
  constexpr float GYRO_QUIET_THRESHOLD = 5.0f;    // deg/s — pen is still
  constexpr int LIFT_QUIET_FRAMES = 50;           // 500 ms of quiet = lifted

  // Combined activity: pen is active if EITHER signal exceeds threshold
  bool jerkActive = smoothJerk > JERK_IMPACT_THRESHOLD;
  bool gyroActive = gyroMag > GYRO_ACTIVE_THRESHOLD;
  bool isActive = jerkActive || gyroActive;

  bool jerkQuiet = smoothJerk < JERK_QUIET_THRESHOLD;
  bool gyroQuiet = gyroMag < GYRO_QUIET_THRESHOLD;
  bool isQuiet = jerkQuiet && gyroQuiet;

  switch (penState) {
  case PenState::IDLE:
    if (isActive) {
      penState = PenState::WRITING;
      quietFrames = 0;
      // Reset stroke state for a new stroke
      strokeTrail.clear();
      intPitch = 0.0f;
      intRoll = 0.0f;
      intYaw = 0.0f;
      intRawGZ = 0.0f;
      anchorPitch = 0.0f;
      anchorRoll = 0.0f;
      anchorYaw = 0.0f;
      anchorRawGZ = 0.0f;
      velocity = glm::vec3(0.0f);
      position = glm::vec3(0.0f);
      posAnchor = glm::vec3(0.0f);
    }
    break;

  case PenState::WRITING:
    if (isQuiet) {
      quietFrames++;
      if (quietFrames > LIFT_QUIET_FRAMES) {
        penState = PenState::LIFTED;
        quietFrames = 0;
      }
    } else {
      quietFrames = 0;
    }
    break;

  case PenState::LIFTED:
    // Transition back to WRITING on any activity, or to IDLE after
    // a longer period of silence.
    if (isActive) {
      penState = PenState::WRITING;
      quietFrames = 0;
    } else {
      quietFrames++;
      // After ~2 seconds of quiet after lift, go fully idle
      if (quietFrames > 200) {
        penState = PenState::IDLE;
        quietFrames = 0;
      }
    }
    break;
  }

  // ── Legacy new-stroke detection (shock-based reset) ───────────────────
  // Keep as a fallback: a massive shock always resets the canvas.
  if (prevIMUValid) {
    float prevMag =
        std::sqrt(prevIMU.ax * prevIMU.ax + prevIMU.ay * prevIMU.ay +
                  prevIMU.az * prevIMU.az);
    float shock = std::abs(curAccelMag - prevMag);
    if (shock > pen::Defaults::wakeThresholdZ && penState == PenState::IDLE) {
      strokeTrail.clear();
      intPitch = 0.0f;
      intRoll = 0.0f;
      intYaw = 0.0f;
      intRawGZ = 0.0f;
      anchorPitch = 0.0f;
      anchorRoll = 0.0f;
      anchorYaw = 0.0f;
      anchorRawGZ = 0.0f;
      velocity = glm::vec3(0.0f);
      position = glm::vec3(0.0f);
      posAnchor = glm::vec3(0.0f);
      penState = PenState::WRITING;
    }
  } else {
    intPitch = 0.0f;
    intRoll = 0.0f;
    intYaw = 0.0f;
    intRawGZ = 0.0f;
    anchorPitch = 0.0f;
    anchorRoll = 0.0f;
    anchorYaw = 0.0f;
    anchorRawGZ = 0.0f;
  }
  prevIMU = imu;
  prevIMUValid = true;

  // ── Integrate gyroscope (deg/s → radians) ────────────────────────────
  // Tilt-compensated → cube rotation display
  intPitch += gx_comp * DT * DEG2RAD;
  intYaw += gy_comp * DT * DEG2RAD;
  intRoll += gz_comp * DT * DEG2RAD;
  // Raw GZ → canvas horizontal axis (calibration: GZ is the true horizontal signal)
  intRawGZ += imu.gz * DT * DEG2RAD;

  // ── Double-integrate dynamic acceleration for translation ────────────
  // Heavy velocity decay prevents unbounded drift.  The position tracks
  // short bursts of lateral hand movement that the gyro alone can't see.
  constexpr float VEL_DECAY = 0.92f;  // per-frame multiplier (< 1 = decay)
  constexpr float ACCEL_GATE = 0.15f; // m/s² — dead zone to reject noise
  float dynMag = std::sqrt(dyn_ax * dyn_ax + dyn_ay * dyn_ay + dyn_az * dyn_az);

  if (dynMag > ACCEL_GATE && penState == PenState::WRITING) {
    velocity.x = velocity.x * VEL_DECAY + dyn_ax * DT;
    velocity.y = velocity.y * VEL_DECAY + dyn_ay * DT;
    velocity.z = velocity.z * VEL_DECAY + dyn_az * DT;
  } else {
    velocity *= VEL_DECAY * 0.8f; // decay faster when idle
  }
  position += velocity * DT;

  // ── Viewport setup ──────────────────────────────────────────────────────
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  int halfWidth = width / 2;

  glViewport(0, 0, width, height);
  glClearColor(0.06f, 0.06f, 0.08f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glm::mat4 view =
      glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -4.0f));
  glm::mat4 projection = glm::perspective(
      glm::radians(45.0f), (float)halfWidth / (float)height, 0.1f, 100.0f);

  // ==========================================
  // LEFT VIEWPORT: 3D Cube (relative rotation)
  // ==========================================
  glViewport(0, 0, halfWidth, height);

  // Rotate by delta from the anchor set at stroke start, so the cube sits at
  // neutral when the pen first touches paper and shows only the writing motion.
  float dYaw = intYaw - anchorYaw;
  float dPitch = intPitch - anchorPitch;
  float dRoll = intRoll - anchorRoll;

  glm::mat4 cubeModel = glm::mat4(1.0f);
  cubeModel = glm::rotate(cubeModel, dYaw, glm::vec3(0.0f, 1.0f, 0.0f));
  cubeModel = glm::rotate(cubeModel, dPitch, glm::vec3(1.0f, 0.0f, 0.0f));
  cubeModel = glm::rotate(cubeModel, dRoll, glm::vec3(0.0f, 0.0f, 1.0f));

  glm::mat4 cubeMVP = projection * view * cubeModel;

  glUseProgram(shaderProgram);
  glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "MVP"), 1, GL_FALSE,
                     glm::value_ptr(cubeMVP));

  glBindVertexArray(faceVAO);
  glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

  glBindVertexArray(edgeVAO);
  glLineWidth(2.0f);
  glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);

  // ==========================================
  // RIGHT VIEWPORT: 2D Stroke Canvas
  // ==========================================
  glViewport(halfWidth, 0, halfWidth, height);

  // ── Canvas position from calibrated raw gyro ─────────────────────────
  // Calibration (5 cm square trace) revealed:
  //   • Raw GZ reliably tracks HORIZONTAL movement (sign flips L↔R)
  //   • Raw GY tracks VERTICAL movement (weaker, sign flips U↔D)
  //   • Tilt compensation mixed GX into GZ → 598% cross-axis bleed
  //   • Accel double-integration is mostly drift (same in all directions)
  //
  // Direct per-axis scale factors (radians → GL units):
  //   Horizontal: 50 mm ≈ 0.108 rad of raw GZ → 5.5 GL/rad
  //   Vertical:   50 mm ≈ 0.012 rad of raw GY → 25.0 GL/rad
  constexpr float SCALE_HORIZ = 5.5f;   // GL units per radian (from GZ)
  constexpr float SCALE_VERT  = 25.0f;  // GL units per radian (from GY)
  constexpr float POS_TO_GL   = 6.0f;   // accel position scale (calibrated)

  // Gyro component: raw axes, signs from calibration
  //   rightward → GZ negative → negate for +X
  //   upward    → GY negative → negate for +Y
  float gyro_x = -intRawGZ * SCALE_HORIZ;
  float gyro_y = -intYaw * SCALE_VERT;  // intYaw = integrated raw GY

  // Accel component: heavily reduced — calibration showed it's mostly drift.
  float accel_x = -position.y * POS_TO_GL;
  float accel_y = -position.x * POS_TO_GL;

  // Blend: gyro dominates, accel just adds micro-impulse texture.
  constexpr float GYRO_WEIGHT  = 0.95f;
  constexpr float ACCEL_WEIGHT = 0.05f;

  glm::vec3 pt(GYRO_WEIGHT * gyro_x + ACCEL_WEIGHT * accel_x,
               GYRO_WEIGHT * gyro_y + ACCEL_WEIGHT * accel_y, 0.0f);

  // Only accumulate trail while the pen is on paper (WRITING state).
  if (penState == PenState::WRITING) {
    if (strokeTrail.empty()) {
      strokeAnchor = pt;
      strokeTrail.push_back(glm::vec3(0.0f));
    } else {
      glm::vec3 rel = pt - strokeAnchor;
      if (glm::length(rel - strokeTrail.back()) > 0.001f) {
        strokeTrail.push_back(rel);
        if (strokeTrail.size() > 4000)
          strokeTrail.erase(strokeTrail.begin());
      }
    }
  }

  // ── Auto-scale: fit stroke bounding box to viewport ──────────────────
  float aspect = static_cast<float>(halfWidth) / static_cast<float>(height);
  float orthoHalfH = 1.5f; // default

  if (strokeTrail.size() > 2) {
    float minX = 1e9f, maxX = -1e9f;
    float minY = 1e9f, maxY = -1e9f;
    for (const auto &p : strokeTrail) {
      if (p.x < minX)
        minX = p.x;
      if (p.x > maxX)
        maxX = p.x;
      if (p.y < minY)
        minY = p.y;
      if (p.y > maxY)
        maxY = p.y;
    }
    float rangeX = maxX - minX;
    float rangeY = maxY - minY;
    float range = std::max(rangeX / aspect, rangeY);
    // Add 30% padding, but enforce a minimum so tiny strokes don't get
    // blown up to fill the entire screen.
    orthoHalfH = std::max(range * 0.65f, 0.3f);
  }

  glm::mat4 ortho = glm::ortho(-orthoHalfH * aspect, orthoHalfH * aspect,
                               -orthoHalfH, orthoHalfH, -1.0f, 1.0f);

  // Center the view on the stroke midpoint
  if (strokeTrail.size() > 2) {
    float cx = 0.0f, cy = 0.0f;
    for (const auto &p : strokeTrail) {
      cx += p.x;
      cy += p.y;
    }
    cx /= static_cast<float>(strokeTrail.size());
    cy /= static_cast<float>(strokeTrail.size());
    ortho = glm::translate(ortho, glm::vec3(-cx, -cy, 0.0f));
  }

  glUseProgram(trailShaderProgram);
  glUniformMatrix4fv(glGetUniformLocation(trailShaderProgram, "MVP"), 1,
                     GL_FALSE, glm::value_ptr(ortho));
  // Neon orange for the stroke trail
  glUniform3f(glGetUniformLocation(trailShaderProgram, "uColor"), 1.0f, 0.6f,
              0.1f);

  glBindVertexArray(trailVAO);
  glBindBuffer(GL_ARRAY_BUFFER, trailVBO);
  glBufferData(GL_ARRAY_BUFFER,
               static_cast<GLsizeiptr>(strokeTrail.size() * sizeof(glm::vec3)),
               strokeTrail.data(), GL_DYNAMIC_DRAW);
  glLineWidth(4.0f);
  glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(strokeTrail.size()));

  // ── Pen state indicator ─────────────────────────────────────────────────
  {
    const char *stateStr = nullptr;
    float cr = 1.0f, cg = 1.0f, cb = 1.0f;
    switch (penState) {
    case PenState::IDLE:
      stateStr = "IDLE";
      cr = 0.5f;
      cg = 0.5f;
      cb = 0.5f;
      break;
    case PenState::WRITING:
      stateStr = "WRITING";
      cr = 0.2f;
      cg = 1.0f;
      cb = 0.3f;
      break;
    case PenState::LIFTED:
      stateStr = "LIFTED";
      cr = 1.0f;
      cg = 0.8f;
      cb = 0.2f;
      break;
    }
    if (stateStr)
      renderText(stateStr, 8.0f, 8.0f, cr, cg, cb, halfWidth, height);
  }

  // ── Word + mode HUD at the bottom of the canvas ─────────────────────────
  // Layout (y increases downward, y=0 = top of viewport):
  //   mode line  — above word, cyan (reading) or yellow (predicting)
  //   word line  — near bottom, white
  constexpr float kS = 2.5f; // label scale (stb units → screen px)
  if (!activeWord.empty()) {
    float wW = static_cast<float>(
        stb_easy_font_width(const_cast<char *>(activeWord.c_str())));
    float tx = (halfWidth - wW * kS) * 0.5f;
    float ty = height - kS * 12.0f - 6.0f;
    renderText(activeWord, tx, ty, 1.0f, 1.0f, 1.0f, halfWidth, height);
  }
  if (!activeMode.empty() && activeMode != "idle") {
    float mW = static_cast<float>(
        stb_easy_font_width(const_cast<char *>(activeMode.c_str())));
    float mtx = (halfWidth - mW * kS) * 0.5f;
    float mty = height - kS * 12.0f * 2.5f - 8.0f; // one line above word
    bool predicting = (activeMode.rfind("Predict", 0) == 0);
    float pcr = predicting ? 1.0f : 0.2f;
    float pcg = predicting ? 0.9f : 1.0f;
    float pcb = predicting ? 0.2f : 0.4f;
    renderText(activeMode, mtx, mty, pcr, pcg, pcb, halfWidth, height);
  }
}

void Visualizer::renderText(const std::string &str, float tx, float ty,
                            float cr, float cg, float cb, int vpWidth,
                            int vpHeight) {
  // stb_easy_font generates quads (4 verts × 16 bytes = 64 bytes/quad).
  // GL Core Profile removed GL_QUADS — convert each quad to two triangles.
  static char stbBuf[32 * 1024];
  int numQuads =
      stb_easy_font_print(0.0f, 0.0f, const_cast<char *>(str.c_str()), nullptr,
                          stbBuf, static_cast<int>(sizeof(stbBuf)));
  if (numQuads <= 0)
    return;

  std::vector<float> verts;
  verts.reserve(static_cast<std::size_t>(numQuads) * 6 * 3);
  for (int q = 0; q < numQuads; ++q) {
    const int base = q * 64;
    float qx[4], qy[4];
    for (int v = 0; v < 4; ++v) {
      qx[v] = *reinterpret_cast<const float *>(stbBuf + base + v * 16 + 0);
      qy[v] = *reinterpret_cast<const float *>(stbBuf + base + v * 16 + 4);
    }
    auto push = [&](int i) {
      verts.push_back(qx[i]);
      verts.push_back(qy[i]);
      verts.push_back(0.0f);
    };
    push(0);
    push(1);
    push(2);
    push(0);
    push(2);
    push(3);
  }

  glBindVertexArray(textVAO);
  glBindBuffer(GL_ARRAY_BUFFER, textVBO);
  glBufferData(GL_ARRAY_BUFFER,
               static_cast<GLsizeiptr>(verts.size() * sizeof(float)),
               verts.data(), GL_DYNAMIC_DRAW);

  // Pixel-space ortho (y down = stb convention).  Scale by kLabelScale so
  // characters are readable (~30 px tall) rather than the default ~12 px.
  constexpr float kLabelScale = 2.5f;
  glm::mat4 proj = glm::ortho(0.0f, static_cast<float>(vpWidth),
                              static_cast<float>(vpHeight), 0.0f, -1.0f, 1.0f);
  glm::mat4 model =
      glm::translate(glm::mat4(1.0f), glm::vec3(tx, ty, 0.0f)) *
      glm::scale(glm::mat4(1.0f), glm::vec3(kLabelScale, kLabelScale, 1.0f));

  glUseProgram(trailShaderProgram);
  glUniformMatrix4fv(glGetUniformLocation(trailShaderProgram, "MVP"), 1,
                     GL_FALSE, glm::value_ptr(proj * model));
  glUniform3f(glGetUniformLocation(trailShaderProgram, "uColor"), cr, cg, cb);

  glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(verts.size() / 3));
}

} // namespace pen
