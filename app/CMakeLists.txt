# App 1: The Machine Learning Text Decoder
add_executable(decoder decoder_main.cpp)

# 1. Keep the target link for PenCore
target_link_libraries(decoder PRIVATE PenCore)

# 2. Ask FetchContent exactly where it downloaded ONNX
FetchContent_GetProperties(onnxruntime)

# 3. Explicitly include the ONNX headers (Ghost folder removed!)
target_include_directories(decoder PRIVATE 
    "${onnxruntime_SOURCE_DIR}/include"
)

# 4. Explicitly link the ONNX .so library binary (Ghost folder removed!)
target_link_libraries(decoder PRIVATE 
    "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so"
)
