cmake_minimum_required(VERSION 3.30)
project(matrix_ops CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_library(matrix_ops STATIC library.cu)

set_target_properties(matrix_ops PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
