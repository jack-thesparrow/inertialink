include(FetchContent)

# Some environments stall for a long time during GLFW's
# `check_function_exists(shmat ...)`. Pre-seed the result to skip that probe.
set(HAVE_SHMAT 1 CACHE INTERNAL "Skip GLFW shmat probe" FORCE)

# --- GLFW SETUP ---
# Disable GLFW tests, examples, and docs to prevent warnings and bin/ clutter!
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.3.8
)
FetchContent_MakeAvailable(glfw)

# --- GLM SETUP ---
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG       1.0.3
)
FetchContent_MakeAvailable(glm)

# --- GLAD SETUP ---
add_library(glad STATIC glad/src/glad.c)
target_include_directories(glad PUBLIC glad/include)

# --- ONNX Runtime --
FetchContent_Declare(
    onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(onnxruntime)

# --- Auto-detect extracted directory ---
file(GLOB ONNX_DIRS LIST_DIRECTORIES true "${onnxruntime_SOURCE_DIR}/*")

# Filter only directories that look like onnxruntime-*
set(ONNX_ROOT "")
foreach(dir ${ONNX_DIRS})
    if(IS_DIRECTORY "${dir}" AND dir MATCHES "onnxruntime-")
        set(ONNX_ROOT "${dir}")
    endif()
endforeach()

if(NOT ONNX_ROOT)
    message(FATAL_ERROR "Could not find extracted ONNX Runtime directory")
endif()

# --- Imported target ---
add_library(ONNXRuntime SHARED IMPORTED)

set_target_properties(ONNXRuntime PROPERTIES
    IMPORTED_LOCATION "${ONNX_ROOT}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNX_ROOT}/include"
)

# --- STB easy font (header-only, no deps, public domain) ---
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/stb)

# --- FTXUI (Terminal UI) ---
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
FetchContent_Declare(ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI
    GIT_TAG        v5.0.0
)
FetchContent_MakeAvailable(ftxui)
