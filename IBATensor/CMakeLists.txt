cmake_minimum_required(VERSION 3.10)
project(IBADeviceData LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)  # Make sure Pybind11 is installed

# Set C++/CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# 1) Prebuilt static library "IBADeviceData" (imported)
#    Adjust the paths to match your actual library location.
# ------------------------------------------------------------------------------
add_library(IBADeviceData STATIC IMPORTED)

set(IBADEVICE_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/IBADeviceData")
set(IBADEVICE_DATA_LIB "${IBADEVICE_DATA_DIR}/build/Debug/IBADeviceData.lib")

set_target_properties(IBADeviceData PROPERTIES
        IMPORTED_LOCATION "${IBADEVICE_DATA_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${IBADEVICE_DATA_DIR}"
)

target_link_libraries(IBADeviceData INTERFACE CUDA::cudart)

# ------------------------------------------------------------------------------
# 2) Build your IBATensorLib as a static library.
#    It uses the prebuilt IBADeviceData internally.
# ------------------------------------------------------------------------------
add_library(IBATensorLib STATIC
        IBATensor.cpp
)

target_link_libraries(IBATensorLib PRIVATE IBADeviceData)

# ------------------------------------------------------------------------------
# 3) Build the Pybind11 module (shared library), for Python import.
#    "bind_ibatensor.cpp" must contain: PYBIND11_MODULE(ibatensor, m) {...}
# ------------------------------------------------------------------------------
add_library(IBATensorPython MODULE
        iba_tensor_binding.cpp
)

# Link with IBATensorLib and Pybind11
target_link_libraries(IBATensorPython
        PRIVATE
        IBATensorLib
        pybind11::module
)

# On Linux/macOS, we remove the 'lib' prefix and set the extension to .so
# On Windows, this will become a .pyd, but the principle is the same.
set_target_properties(IBATensorPython PROPERTIES
        PREFIX ""
        SUFFIX ".pyd"
)

# ------------------------------------------------------------------------------
# 4) (Optional) Build your test executable as before.
# ------------------------------------------------------------------------------
add_executable(TestExecutable main.cpp)

target_link_libraries(TestExecutable
        PRIVATE
        IBATensorLib
        CUDA::cudart
)
