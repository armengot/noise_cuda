cmake_minimum_required(VERSION 3.10)
#set(CMAKE_VERBOSE_MAKEFILE ON)
project(test_noise)
set(CMAKE_CXX_STANDARD 14)
# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -isystem=/usr/local/include/opencv4")
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
cuda_add_library(noisemeter STATIC noisemeter.cu)
add_executable(${PROJECT_NAME} test_noise.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE noisemeter ${OpenCV_LIBS})
