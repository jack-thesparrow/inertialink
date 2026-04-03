add_executable(visualizer visualizer_main.cpp)
target_link_libraries(visualizer PRIVATE PenCore)

add_executable(decoder decoder_main.cpp)
target_link_libraries(decoder PRIVATE PenCore)
