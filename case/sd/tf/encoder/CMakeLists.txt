cmake_minimum_required(VERSION 3.16)
INCLUDE(CMakeForceCompiler) 
set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_C_FLAGS "-march=rv64gcv0p7 -mabi=lp64d -g")
set(CMAKE_CXX_FLAGS "-march=rv64gcv0p7 -mabi=lp64d -g")
project(minimal C CXX)

set(TENSORFLOW_SOURCE_DIR "/home/sipeed/tensorflow/tensorflowbak")

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

find_package(ZLIB REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
add_executable(decoder decoder.cpp)
target_link_libraries(decoder tensorflow-lite ZLIB::ZLIB ${OpenCV_LIBS})
