cmake_minimum_required(VERSION 3.10)

project(IBADeviceData LANGUAGES CXX CUDA)

# Set C++/CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe")


# Create a static library from your source files
# Note: Typically, you do NOT include main.cpp in a static library.
#       main.cpp is usually part of an executable target.
add_library(IBADeviceData STATIC
        CudaData.cu
        deviceData.h
        # Headers are optional to list here, but it's okay for clarity
)

add_executable(DeviceTester
    main.cpp)

target_link_libraries(DeviceTester PRIVATE IBADeviceData)