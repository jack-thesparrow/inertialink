# ==========================================
# APP 1: Data Collector
# ==========================================
add_executable(data_collector data_collector_main.cpp)
target_link_libraries(data_collector PRIVATE PenCore)

# ==========================================
# APP 2: ML Text Decoder
# ==========================================
add_executable(decoder decoder_main.cpp)
target_link_libraries(decoder PRIVATE PenCore)

# Ask FetchContent exactly where it downloaded ONNX
FetchContent_GetProperties(onnxruntime)
target_include_directories(decoder PRIVATE "${onnxruntime_SOURCE_DIR}/include")
target_link_libraries(decoder PRIVATE "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so")

# ==========================================
# APP 3: 3D OpenGL Visualizer
# ==========================================
add_executable(visualizer visualizer_main.cpp)
# Link the Core backend, plus our graphics drivers (glad, glfw) and math library (glm)
target_link_libraries(visualizer PRIVATE PenCore glad glfw glm)

# ==========================================
# APP 4: Terminal UI Launcher
# ==========================================
add_executable(tui tui_main.cpp)
target_link_libraries(tui PRIVATE PenCore
    ftxui::screen ftxui::dom ftxui::component)
