#pragma once
#include "pen/io.hpp"
#include <glm/glm.hpp>
#include <string>
#include <vector>

struct GLFWwindow;

namespace pen {

// Pen contact state machine
enum class PenState { IDLE, WRITING, LIFTED };

class Visualizer {
public:
  Visualizer(int width, int height, const char *title);
  ~Visualizer();

  bool isOpen() const;
  void update();
  void drawCube(const IMUData &imuState);

private:
  GLFWwindow *window;

  // Cube Data
  unsigned int shaderProgram;
  unsigned int faceVAO, faceVBO, faceEBO;
  unsigned int edgeVAO, edgeVBO, edgeEBO;

  // Trail Data
  unsigned int trailShaderProgram;
  unsigned int trailVAO, trailVBO;
  std::vector<glm::vec3> strokeTrail;
  glm::vec3 strokeAnchor{0.0f};  // trail origin — stroke is stored relative to this

  // Stroke detection state
  IMUData  prevIMU;
  bool     prevIMUValid{false};

  // Integrated gyro orientation (radians) — used for cube rotation
  float intPitch{0.0f}, intRoll{0.0f}, intYaw{0.0f};
  // Translation-gated canvas accumulators:
  // Only accumulate when dynamic accel confirms pen tip is actually translating
  // (not just rotating/tilting in place around a fixed tip).
  float canvasGZ{0.0f};  // horizontal canvas position (from raw GZ, gated)
  float canvasGY{0.0f};  // vertical canvas position (from raw GY, gated)
  float translationGate{0.0f};  // smoothed 0..1 confidence of tip translation
  // Anchor values at stroke start — cube rotates relative to these
  float anchorPitch{0.0f}, anchorRoll{0.0f}, anchorYaw{0.0f};

  // ── Translation tracking (complementary filter + double integration) ──
  glm::vec3 gravity{0.0f, 0.0f, 1.0f};   // estimated gravity in sensor frame (g)
  glm::vec3 velocity{0.0f};                // integrated dynamic acceleration (m/s)
  glm::vec3 position{0.0f};                // integrated velocity (m)
  glm::vec3 posAnchor{0.0f};               // position at stroke start

  // ── Jerk-based pen state ──
  PenState  penState{PenState::IDLE};
  float     prevAccelMag{1.0f};             // previous total accel magnitude
  float     prevJerk{0.0f};                 // previous jerk (for smoothing)
  int       quietFrames{0};                 // consecutive low-jerk frames

  // HUD state (polled from /tmp/inertialink_word and /tmp/inertialink_mode)
  int         frameCount{0};
  std::string activeWord;
  std::string activeMode;

  // Text rendering (stb_easy_font quads → triangles)
  unsigned int textVAO{0}, textVBO{0};

  void initOpenGL();
  void setupGeometry();
  // Render `str` at viewport pixel position (tx, ty) scaled by kLabelScale,
  // coloured (cr, cg, cb).  vpWidth/vpHeight are the current viewport dims.
  void renderText(const std::string &str, float tx, float ty,
                  float cr, float cg, float cb, int vpWidth, int vpHeight);
};
} // namespace pen
