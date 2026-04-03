add_library(PenCore io.cpp viz.cpp)
target_include_directories(PenCore PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(PenCore PUBLIC glad glfw glm)
