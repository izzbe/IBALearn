#include "CudaData.cuh"
#include <iostream>
#include <cfloat>
// -------------------------------------------------------- BACKWARD --------------------------------------------------------------
//
//
//
//
//
//
// ---------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------decls-------------------------------------------------------------
__global__ void conv2d_backward_kernel(const float *sigma, const float *input,
                                       int C_k, int K, int C_in, int H_in, int W_in,
                                       int C_sigma, int H_sigma, int W_sigma, int P, int S, float *out);

__global__ void rotate_180_kernel(int H, int W, int C, int N, float *in, float *out);

__global__ void conv2d_backward_kernel_wr_input(const float *sigma, const float *kernel, int H_in, int W_in, int K, int C_in_k, int C_out_k,
                                                                 int sigma_H, int sigma_W, int sigma_C, int sigma_N, int P, int S, float *out);
// ------------------------------------------------- Matrix Ops -------------------------------------------------------
std::unique_ptr<DeviceData> CudaData::conv2d_backward_wr_kernel(const DeviceData *sigma,
                                            int C_k, int K, int H_in, int W_in, int C_in, int C_sigma, int H_sigma, int W_sigma, int P, int S, int N) const {

    dim3 blockDim(TILE_SIZE, TILE_SIZE);

    dim3 gridDim(((K + TILE_SIZE - 1) / TILE_SIZE) * ((K + TILE_SIZE - 1) / TILE_SIZE),
                C_in * C_k,
                N);

    int size = C_k * C_in * K * K;

    float *C;
    cudaMalloc(&C, size * sizeof(float));

    conv2d_backward_kernel<<<gridDim, blockDim>>>(sigma->getData(), head, C_k, K, C_in, H_in, W_in, C_sigma, H_sigma, W_sigma, P, S, C);

    cudaDeviceSynchronize();
    return std::make_unique<CudaData>(C, size);
}

std::unique_ptr<DeviceData> CudaData::conv2d_backward_wr_input(const DeviceData *sigma, const DeviceData *kernel, int H_in, int W_in, int K, int C_in_k, int C_out_k,
                                                                 int sigma_H, int sigma_W, int sigma_C, int sigma_N, int P, int S) const {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((W_in + TILE_SIZE - 1) / TILE_SIZE, (H_in + TILE_SIZE - 1) / TILE_SIZE, C_in_k * sigma_N);

    int size = sigma_N * C_in_k * H_in * W_in;

    float *out;

    cudaMalloc(&out, size * sizeof(float));

    conv2d_backward_kernel_wr_input<<<gridDim, blockDim>>>(sigma->getData(), kernel->getData(), H_in, W_in, K, C_in_k, C_out_k, sigma_H, sigma_W, sigma_C, sigma_N, P, S, out);

    return std::make_unique<CudaData>(out, size);
}

// ------------------------------------------------- Kernels ------------------------------------------------------------

__global__ void conv2d_backward_kernel_wr_input(const float *sigma, const float *kernel, int H_in, int W_in, int K, int C_in_k, int C_out_k,
                                                                 int sigma_H, int sigma_W, int sigma_C, int sigma_N, int P, int S, float *out) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z % C_in_k;
    int out_n = blockIdx.z / C_in_k;

    if (out_x < 0 || out_y < 0 || out_c < 0 || out_n < 0 || out_x >= W_in || out_y >= H_in || out_c >= C_in_k || out_n >= sigma_N) {
        return;
    }

    float grad_val = 0.0f;
    for (int c = 0; c < C_out_k; c++) {
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                int current_top_left_x = out_x - i;
                int current_top_left_y = out_y - j;
                if ( (current_top_left_x + P) % S != 0 || (current_top_left_y + P) % S != 0 ) {
                    continue;
                }

                if ( current_top_left_x < -P || current_top_left_y < -P || current_top_left_x > (W_in + 2 * P - K) || current_top_left_y > (H_in + 2 * P - K) ) {
                    continue;
                }

                int sigma_val_x = (current_top_left_x + P) / S;
                int sigma_val_y = (current_top_left_y + P) / S;

                grad_val += kernel[c * C_in_k * K * K + out_c * K * K + j * K + i] *
                    sigma[out_n * sigma_C * sigma_W * sigma_H + c * sigma_W * sigma_H + sigma_val_y * sigma_W + sigma_val_x];

            }
        }
    }
    out[out_n * C_in_k * H_in * W_in + out_c * H_in * W_in + out_y * W_in + out_x] = grad_val;


}

__global__ void conv2d_backward_kernel(const float *sigma, const float *input,
                                       int C_k, int K, int C_in, int H_in, int W_in,
                                       int C_sigma, int H_sigma, int W_sigma, int P, int S, float *out) {
    int block_x = blockIdx.x % ((K + TILE_SIZE - 1) / TILE_SIZE);
    int block_y = blockIdx.x / ((K + TILE_SIZE - 1) / TILE_SIZE);

    int out_x = block_x * blockDim.x + threadIdx.x;
    int out_y = block_y * blockDim.y + threadIdx.y;

    int out_c_in = blockIdx.y % C_in;
    int out_c_k = blockIdx.y / C_in;

    int out_batch_n = blockIdx.z;

    if (out_x >= K || out_y >= K) {
        return;
    }

    float grad_val = 0.0f;
    for (int i = 0; i < W_sigma; i++) {
        for (int j = 0; j < H_sigma; j++) {
            int input_x = i * S - P + out_x;
            int input_y = j * S - P + out_y;

            if (input_x >= W_in || input_y >= H_in || input_x < 0|| input_y < 0) {
                continue;
            }

            grad_val += input[out_batch_n * C_in * H_in * W_in + out_c_in * H_in * W_in + input_y * W_in + input_x] *
                        sigma[out_batch_n * C_sigma * H_sigma * W_sigma + out_c_k * H_sigma * W_sigma + j * W_sigma + i];
        }
    }
    atomicAdd(&out[ out_c_k * C_in * K * K + out_c_in * K * K + out_y * K + out_x], grad_val);

}

