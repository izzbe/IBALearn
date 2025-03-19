cmake_minimum_required(VERSION 3.10)

# Set the project name
project(IBATensor)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add source files
set(SOURCES
        IBATensor.cpp
        IBATensor_binding.cpp
        main.cpp
)

set(HEADERS
        IBATensor.h
)

# Create the executable
add_executable(IBATensorExecutable ${SOURCES})

# Include header files
target_include_directories(IBATensorExecutable PRIVATE ${CMAKE_SOURCE_DIR}/IBADeviceData)

# Link libraries (if any)
# target_link_libraries(IBATensorExecutable <LIBRARY_NAME>)