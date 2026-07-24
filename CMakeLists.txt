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
find_library(GGML_LIB ggml PATHS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/build/src REQUIRED)
find_library(GGML_CPU_LIB ggml-cpu PATHS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/build/src REQUIRED)

# --- main library ---
file(GLOB SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cpp
)

add_library(nano-vllm SHARED ${SRC_FILES})

if(USE_CPU)
  target_compile_definitions(nano-vllm PRIVATE USE_CPU)
endif()

if(USE_CUDA)
  find_package(CUDA REQUIRED)
  target_compile_definitions(nano-vllm PRIVATE USE_CUDA)
  target_include_directories(nano-vllm PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(nano-vllm PRIVATE ${CUDA_LIBRARIES})
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
  ${GGML_LIB}
  ${GGML_CPU_LIB}
  xxhash
  safetensors-cpp
  tokenizers_cpp
  pthread
)

add_subdirectory(examples)
add_subdirectory(test)
