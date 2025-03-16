#!/bin/bash

# Set the CUDA path (modify if your CUDA version is different)
CUDA_PATH="/usr/local/cuda"

# Compiler flags
CXX_FLAGS="-std=c++17 -Wall -Wextra -O2 -g"
NVCC_FLAGS="-g -G --compiler-options -fPIC"
LINK_FLAGS="-lcudart"

# Output binary name (force .exe on Windows)
OUTPUT="matrix_ops.exe"

echo "🚀 Cleaning previous builds..."
rm -f main.o CPUData.o CudaData.o deviceData.o $OUTPUT

echo "🔧 Compiling CUDA source files..."
nvcc $NVCC_FLAGS -c CudaData.cu -o CudaData.o

echo "🔧 Compiling C++ source files..."
g++ $CXX_FLAGS -c main.cpp CPUData.cpp -I"$CUDA_PATH/include"

echo "🔗 Linking..."
g++ $CXX_FLAGS -o $OUTPUT main.o CPUData.o CudaData.o -L"$CUDA_PATH/lib64" $LINK_FLAGS

echo "✅ Build complete! Run with ./$OUTPUT"