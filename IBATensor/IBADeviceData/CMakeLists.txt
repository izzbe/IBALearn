cmake_minimum_required(VERSION 3.10)

# Name of your project and specify that it has C++ and CUDA components
project(IBADeviceData LANGUAGES CXX CUDA)

# Set C++/CUDA standards (optional but recommended)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# In case you need to control CUDA-specific compilation flags:
# For example, to compile for a certain GPU architecture:
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

# Option A: Directly add an executable with all source files (CPU + CUDA):
# This is the simplest approach if your project is small.
add_executable(IBADeviceData
        main.cpp        # Your main entry point
        CPUDATA.cpp     # CPU-only sources
        CudaData.cu     # CUDA sources
        deviceData.h    # Headers (not strictly necessary to list, but can be included for clarity)
)

# Option B (alternative): Separate CPU and CUDA portions into libraries, then link.
# This is useful if you have more complex structure or plan to reuse libraries.
#
# add_library(cpu_data STATIC CPUDATA.cpp deviceData.h)
# add_library(gpu_data STATIC CudaData.cu)
# target_link_libraries(gpu_data PUBLIC cuda) # or other libraries if needed
# add_executable(IBADeviceData main.cpp)
# target_link_libraries(IBADeviceData PRIVATE cpu_data gpu_data)

# If you need to link other libraries (e.g., CUDA runtime, external libs), do so here.
# For example:
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(IBADeviceData PRIVATE CUDA::cudart)

