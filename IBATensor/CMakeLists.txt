cmake_minimum_required(VERSION 3.10)

project(IBADeviceData LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

# Set C++/CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# 1) Declare a prebuilt static library called "IBADeviceData".
#    Replace the paths below with your actual paths:
# -----------------------------------------------------------------------------
add_library(IBADeviceData STATIC IMPORTED)

set(IBADEVICE_DATA_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/IBADeviceData")
set(IBADEVICE_DATA_LIB   "${IBADEVICE_DATA_DIR}/build/Debug/IBADeviceData.lib")

set_target_properties(IBADeviceData PROPERTIES
        IMPORTED_LOCATION           "${IBADEVICE_DATA_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${IBADEVICE_DATA_DIR}"
)

add_library(IBATensorLib STATIC
    IBATensor.cpp
)

# If your library requires the CUDA runtime, or other libs, you can link them too:
# For example:
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(IBADeviceData INTERFACE CUDA::cudart)

# -----------------------------------------------------------------------------
# 2) Create an executable (or library) that uses the prebuilt IBADeviceData library.
# -----------------------------------------------------------------------------
target_link_libraries(IBADeviceData INTERFACE CUDA::cudart)
target_link_libraries(IBATensorLib PRIVATE IBADeviceData)

add_executable(TestExecutable main.cpp)

target_link_libraries(TestExecutable
        PRIVATE
        IBATensorLib
        CUDA::cudart
)