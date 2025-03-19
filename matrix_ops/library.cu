#include "library.cuh"
#include <iostream>
#include <cuda_runtime.h>

__global__ void matMulTiledKernel(const float A[], int A_shape[], const float B[], int B_shape[], float C[], int C_shape[]) {


    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];


    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;

    float c_value = 0.0f;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (A_shape[1] + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        if (global_row < A_shape[0] && (m * TILE_SIZE + col) < A_shape[1]) {
            shared_A[row][col] = A[global_row * A_shape[1] + m * TILE_SIZE + col];
        } else {
            shared_A[row][col] = 0.0f;
        }

        if (global_col < B_shape[1] && (m * TILE_SIZE + row) < B_shape[0]) {
            shared_B[row][col] = B[(m * TILE_SIZE + row) * B_shape[1] + global_col];
        } else {
            shared_B[row][col] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            c_value += shared_A[row][k] * shared_B[k][col];
        }

        __syncthreads();
    }

    if (global_row < C_shape[0] && global_col < C_shape[1]) {
        C[global_row * C_shape[1] + global_col] = c_value;
    }

}

__global__ void matTensorAdd(const float A[], int A_shape[], const float B[], int B_shape[], float C[], int C_shape[]) {
  	int global_row = blockIdx.x * blockDim.x + threadIdx.x;

	int A_dim =
}


