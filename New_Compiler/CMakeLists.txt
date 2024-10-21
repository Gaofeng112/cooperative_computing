# 设置CMake的最低版本要求
cmake_minimum_required(VERSION 3.10)
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
SET(CMAKE_CXX_STANDARD 17)

# 设置项目名称
project(CoComputingCompiler)
# #堆栈大小
# MATH(EXPR stack_size "100*1024*1024")
# set(CMAKE_EXE_LINKER_FLAGS "-Wl,--stack,${stack_size}")

# 查找protobuf库
find_package(Protobuf REQUIRED)
# 设置生成的源文件路径
set(PROTO_FILES onnx.proto)  # 替换为你的.proto文件路径
# 生成对应的C++文件
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
# 将生成的头文件的目录添加到include路径中
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# 添加头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/Device)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${PROJECT_NAME} ${Python3_INCLUDE_DIRS})

# 将所有源文件添加到一个变量中
file(GLOB SOURCES "lib/*.cpp" "lib/Device/*.cpp")

# 编译main.cpp
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
# 添加可执行文件
add_executable(CoComputingCompiler main.cpp ${SOURCES} ${PROTO_SRCS} ${PROTO_HDRS})  # 将生成的头文件添加到add_executable命令中
# 链接protobuf库
target_link_libraries(CoComputingCompiler protobuf ${Python3_LIBRARIES})