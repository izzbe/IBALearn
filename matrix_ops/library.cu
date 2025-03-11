#include "library.cuh"
#include <iostream>
#include <cuda_runtime.h>

const int TILE_SIZE = 16;
const int MATRIX_SIZE = 18000;

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

void matMulCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

int main() {
    int M = MATRIX_SIZE, N = MATRIX_SIZE, K = MATRIX_SIZE;
    size_t matrixSizeA = M * K * sizeof(float);
    size_t matrixSizeB = K * N * sizeof(float);
    size_t matrixSizeC = M * N * sizeof(float);

    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrixSizeA);
    cudaMalloc((void**)&d_B, matrixSizeB);
    cudaMalloc((void**)&d_C, matrixSizeC);

    cudaMemcpy(d_A, h_A, matrixSizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSizeB, cudaMemcpyHostToDevice);

    int h_A_shape[2] = {M, K};
    int h_B_shape[2] = {K, N};
    int h_C_shape[2] = {M, N};

    int *d_A_shape, *d_B_shape, *d_C_shape;
    cudaMalloc((void**)&d_A_shape, 2 * sizeof(int));
    cudaMalloc((void**)&d_B_shape, 2 * sizeof(int));
    cudaMalloc((void**)&d_C_shape, 2 * sizeof(int));

    cudaMemcpy(d_A_shape, h_A_shape, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_shape, h_B_shape, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_shape, h_C_shape, 2 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulTiledKernel<<<gridDim, blockDim>>>(d_A, d_A_shape, d_B, d_B_shape, d_C, d_C_shape);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTimeMs = 0.0f;
    cudaEventElapsedTime(&gpuTimeMs, start, stop);

    std::cout << "CUDA Matrix Multiplication Time: " << gpuTimeMs / 1000.0 << " seconds" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_shape);
    cudaFree(d_B_shape);
    cudaFree(d_C_shape);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
