#pragma once
#include "pen/io.hpp"
#include <glm/glm.hpp>
#include <string>
#include <vector>

struct GLFWwindow;

namespace pen {
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
  IMUData  cubeAnchor;           // IMU angles at stroke start; cube rotates relative to this

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
