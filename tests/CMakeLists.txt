add_executable(io_device_checks_test
  io_device_checks_test.cpp
  ${CMAKE_SOURCE_DIR}/src/io.cpp
)

target_include_directories(io_device_checks_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME io_device_checks_test COMMAND io_device_checks_test)
