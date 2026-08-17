cmake_minimum_required(VERSION 3.20)

project(nano-vllm.cpp VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(USE_CUDA "Enable CUDA support" OFF)
option(USE_CPU "Enable CPU optimizations" ON)

# --- xxHash ---
add_library(xxhash STATIC
  third_party/xxHash/xxhash.c
)
target_include_directories(xxhash PUBLIC third_party/xxHash)

# --- safetensors-cpp ---
add_library(safetensors-cpp STATIC
  third_party/safetensors-cpp/safetensors.cc
)
target_include_directories(safetensors-cpp PUBLIC third_party/safetensors-cpp)

# --- tokenizers-cpp ---
add_subdirectory(third_party/tokenizers-cpp)

# --- ggml ---
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Disable ggml examples" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "Disable ggml tests" FORCE)

if(USE_CUDA)
  set(GGML_CUDA ON CACHE BOOL "ggml: use CUDA" FORCE)
  find_package(CUDAToolkit REQUIRED)
endif()

add_subdirectory(third_party/ggml)

# --- main library ---
# CONFIGURE_DEPENDS: re-glob on every build so new source files are picked up
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(nano-vllm SHARED ${SRC_FILES})

if(USE_CPU)
  target_compile_definitions(nano-vllm PRIVATE USE_CPU)
endif()

if(USE_CUDA)
  target_compile_definitions(nano-vllm PRIVATE USE_CUDA)
endif()

target_include_directories(nano-vllm
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include/
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/safetensors-cpp/
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp/include
)

target_link_libraries(nano-vllm
  PRIVATE
  ggml
  xxhash
  safetensors-cpp
  tokenizers_cpp
  pthread
)

add_subdirectory(examples)
