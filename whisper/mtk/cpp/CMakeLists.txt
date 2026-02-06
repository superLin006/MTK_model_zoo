cmake_minimum_required(VERSION 3.10)
project(whisper_mtk)

# C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/utils
)

# Source files
set(AUDIO_UTILS_SOURCES
    utils/audio_utils.cpp
)

set(INFERENCE_SOURCES
    src/whisper_inference.cpp
    src/neuron_executor.cpp
)

set(MAIN_SOURCES
    src/main.cpp
)

# Build audio utils library
add_library(audio_utils STATIC ${AUDIO_UTILS_SOURCES})

# Build inference library
add_library(whisper_inference STATIC ${INFERENCE_SOURCES})
target_link_libraries(whisper_inference audio_utils)

# Build main executable
add_executable(whisper_test ${MAIN_SOURCES})
target_link_libraries(whisper_test
    whisper_inference
    audio_utils
    dl  # For dlopen/dlsym
    m   # For math library
)

# Compiler flags
target_compile_options(whisper_test PRIVATE
    -Wall
    -Wextra
    -O2
)

target_compile_options(audio_utils PRIVATE
    -Wall
    -Wextra
    -O2
)

target_compile_options(whisper_inference PRIVATE
    -Wall
    -Wextra
    -O2
)

# Installation
install(TARGETS whisper_test
    RUNTIME DESTINATION bin
)
