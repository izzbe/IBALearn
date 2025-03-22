cmake_minimum_required(VERSION 3.10)

project(IBADeviceData LANGUAGES CXX CUDA)




# Set C++/CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Create a static library from your source files
# Note: Typically, you do NOT include main.cpp in a static library.
#       main.cpp is usually part of an executable target.
add_library(IBADeviceData STATIC
        CPUData.cpp
        CudaData.cu
        deviceData.h      # Headers are optional to list here, but it's okay for clarity
)