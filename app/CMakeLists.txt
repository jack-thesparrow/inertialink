add_executable(visualizer visualizer_main.cpp)
target_link_libraries(visualizer PRIVATE PenCore)

add_executable(decoder decoder_main.cpp)
target_link_libraries(decoder PRIVATE PenCore)

add_executable(data_collector data_collector_main.cpp)
target_link_libraries(data_collector PRIVATE PenCore)
