cmake_minimum_required(VERSION 3.10)
project(test_noise)
set(CMAKE_CXX_STANDARD 14)

# CUDA configuration
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.7)
find_package(CUDA REQUIRED)

# Set the architecture for CUDA (here tested RTX 3060 Ti uses sm_80 for Ampere architecture)
set(CMAKE_CUDA_ARCHITECTURES "80")

# remove warnings cuda
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -diag-suppress=611")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -w")

# remove warnings c++
add_compile_options(-w)
add_compile_options(-Wno-overloaded-virtual)

# OpenCV configuration
find_package(OpenCV REQUIRED)

# Include directories for CUDA and OpenCV
include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# CUDA sources - define your own CUDA sources here
cuda_add_library(noisemeter STATIC noisemeter.cu)

# Build the final executable
add_executable(${PROJECT_NAME} test_noise.cpp)

# Link libraries: CUDA, OpenCV, and your custom CUDA library
target_link_libraries(${PROJECT_NAME} PRIVATE noisemeter ${OpenCV_LIBS})

# Link CUDA libraries if OpenCV with CUDA support is found
if(OpenCV_CUDA_LIBRARIES)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_CUDA_LIBRARIES})
endif()

