cmake_minimum_required(VERSION 3.10)

project(MyApp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# 1) Declare a prebuilt static library called "IBADeviceData".
#    Replace the paths below with your actual paths:
# -----------------------------------------------------------------------------
add_library(IBADeviceData STATIC IMPORTED)

set_target_properties(IBADeviceData PROPERTIES
        IMPORTED_LOCATION "IBADeviceData/build/Debug/IBADeviceData.lib"         # or .lib on Windows
        INTERFACE_INCLUDE_DIRECTORIES "IBADeviceData"       # where deviceData.h, etc. live
)

# If your library requires the CUDA runtime, or other libs, you can link them too:
# For example:
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(IBADeviceData INTERFACE CUDA::cudart)

# -----------------------------------------------------------------------------
# 2) Create an executable (or library) that uses the prebuilt IBADeviceData library.
# -----------------------------------------------------------------------------
add_executable(TestExecutable main.cpp)

# Link the prebuilt library to the executable
target_link_libraries(TestExecutable PRIVATE IBADeviceData)