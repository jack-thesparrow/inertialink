#include "pen/viz.hpp"
#include <glad/glad.h>
// glad before GLFW
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>

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
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0, 0.6, 0.1, 1.0); // Bright Neon Orange
}
)";

Visualizer::Visualizer(int width, int height, const char *title) {
  if (!glfwInit())
    exit(-1);

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 4);

  window = glfwCreateWindow(width, height, title, nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    exit(-1);
  }

  glfwMakeContextCurrent(window);
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    exit(-1);

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
}

void Visualizer::drawCube(const IMUData &imu) {
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  int halfWidth = width / 2;

  // Clear the ENTIRE window once
  glViewport(0, 0, width, height);
  glClearColor(0.06f, 0.06f, 0.08f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Both sides will share this camera setup, adjusted for the half-screen
  // aspect ratio
  glm::mat4 view =
      glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -4.0f));
  glm::mat4 projection = glm::perspective(
      glm::radians(45.0f), (float)halfWidth / (float)height, 0.1f, 100.0f);

  // ==========================================
  // LEFT VIEWPORT: 3D Spinning Cube
  // ==========================================
  glViewport(0, 0, halfWidth,
             height); // Tell OpenGL to only draw on the left half

  glm::mat4 cubeModel = glm::mat4(1.0f);
  // The cube stays centered and just spins to show orientation
  cubeModel = glm::rotate(cubeModel, imu.yaw, glm::vec3(0.0f, 1.0f, 0.0f));
  cubeModel = glm::rotate(cubeModel, imu.pitch, glm::vec3(1.0f, 0.0f, 0.0f));
  cubeModel = glm::rotate(cubeModel, imu.roll, glm::vec3(0.0f, 0.0f, 1.0f));

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
  glViewport(halfWidth, 0, halfWidth,
             height); // Tell OpenGL to only draw on the right half

  // Calculate the 2D coordinate directly from pitch and yaw
  float canvasScale = 4.0f;
  glm::vec3 newPoint(-imu.yaw * canvasScale, imu.pitch * canvasScale, 0.0f);

  // Add the point to our stroke trail
  if (strokeTrail.empty() ||
      glm::length(strokeTrail.back() - newPoint) > 0.005f) {
    strokeTrail.push_back(newPoint);
    if (strokeTrail.size() > 2000)
      strokeTrail.erase(strokeTrail.begin());
  }

  glm::mat4 trailModel = glm::mat4(1.0f); // Trail stays static on the canvas
  glm::mat4 trailMVP = projection * view * trailModel;

  glUseProgram(trailShaderProgram);
  glUniformMatrix4fv(glGetUniformLocation(trailShaderProgram, "MVP"), 1,
                     GL_FALSE, glm::value_ptr(trailMVP));

  glBindVertexArray(trailVAO);
  glBindBuffer(GL_ARRAY_BUFFER, trailVBO);
  glBufferData(GL_ARRAY_BUFFER, strokeTrail.size() * sizeof(glm::vec3),
               strokeTrail.data(), GL_DYNAMIC_DRAW);

  glLineWidth(4.0f);
  glDrawArrays(GL_LINE_STRIP, 0, strokeTrail.size());
}

} // namespace pen
