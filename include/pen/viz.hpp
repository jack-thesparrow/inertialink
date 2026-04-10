#pragma once
#include "pen/io.hpp"
#include <glm/glm.hpp>
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

  void initOpenGL();
  void setupGeometry();
};
} // namespace pen
