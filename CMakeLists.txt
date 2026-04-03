cmake_minimum_required(VERSION 3.20)
project(Inertialink VERSION 1.0 LANGUAGES CXX C)
## --- Custom Progress Bar Macro ---
#function(print_progress percentage message_text)
#    math(EXPR num_bars "${percentage} / 10")
#    math(EXPR num_spaces "10 - ${num_bars}")
#    
#    set(bar_string "")
#    if(num_bars GREATER 0)
#        foreach(i RANGE 1 ${num_bars})
#            string(APPEND bar_string "█")
#        endforeach()
#    endif()
#    
#    set(space_string "")
#    if(num_spaces GREATER 0)
#        foreach(i RANGE 1 ${num_spaces})
#            string(APPEND space_string " ")
#        endforeach()
#    endif()
#
#    message(STATUS "[${bar_string}${space_string}] ${percentage}% - ${message_text}")
#endfunction()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force CMake to generate compile_commands.json for Neovim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Force CMake to generate compile_commands.json for Neovim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Automatically symlink compile_commands.json to the project root
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${CMAKE_SOURCE_DIR}/compile_commands.json"
)
# Route executables to the /bin folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Route static/shared libraries to the /lib folder
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# 1. Load external libraries
add_subdirectory(third_party)

# 2. Build your core smart pen library
add_subdirectory(src)

# 3. Build the executables
add_subdirectory(app)
