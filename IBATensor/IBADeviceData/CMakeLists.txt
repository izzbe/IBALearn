cmake_minimum_required(VERSION 3.18)  # Minimum required CMake version
project(matrix_opsV2 LANGUAGES CXX CUDA)  # Enable CUDA & C++

set(CMAKE_CXX_STANDARD 17)  # Use C++17 or later
set(CMAKE_CUDA_STANDARD 17)  # Use CUDA C++17

add_library(cuda_ops STATIC CudaData.cu)
set_target_properties(cuda_ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ✅ Add CPU and common device files
add_library(cpu_ops STATIC CPUData.cpp deviceData.cpp)

# ✅ Link CUDA and CPU libraries together
add_executable(matrix_ops main.cpp)  # Replace main.cpp with your entry point
target_link_libraries(matrix_ops cpu_ops cuda_ops)