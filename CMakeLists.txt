cmake_minimum_required(VERSION 3.20)

project(nano-vllm.cpp VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

option(USE_CUDA "Enable CUDA support" OFF)
option(USE_CPU "Enable CPU optimizations" ON)

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

find_package(GTest REQUIRED)

target_include_directories(nano-vllm
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include/
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xxHash/
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/safetensors-cpp/
)

target_link_libraries(nano-vllm
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/build/src/libggml.so
)


add_subdirectory(examples)
add_subdirectory(test)
