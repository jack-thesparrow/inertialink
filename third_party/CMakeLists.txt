include(FetchContent)

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
