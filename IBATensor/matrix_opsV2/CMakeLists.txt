cmake_minimum_required(VERSION 3.30)
project(matrix_opsV2 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_library(matrix_opsV2 STATIC library.cu)

set_target_properties(matrix_opsV2 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
