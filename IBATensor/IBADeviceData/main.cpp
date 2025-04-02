#include "deviceData.h"
#include "CudaData.cuh"
#include <random>

void printDeviceData(const DeviceData* data, int N, int C, int H, int W) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            std::cout << "\n=== N = " << n << ", C = " << c << " ===\n";
            for (int h = 0; h < H; ++h) {
                for (int w_ = 0; w_ < W; ++w_) {
                    // Compute the linear index in your flattened array
                    int idx = n * C * H * W
                              + c * H * W
                              + h * W
                              + w_;
                    std::cout << data->iloc(idx) << "\t";
                }
                std::cout << "\n";
            }
        }
    }
    std::cout << std::endl;
}


int main() {
    const int N = 1, C = 1, H = 3, W = 3;
    const int K = 2, P = 0, S = 0;

    const int H_out = (H + 2 * P - K) / S + 1;
    const int W_out = (W + 2 * P - K) / S + 1;

    // Input: 1x1x2x2
    std::vector<float> input_data = {
        1.0f, 3.0f, 4.0f,
        3.0f, 1.0f, 4.0f,
        1.0f, 2.0f, 3.0f
    };

    std::vector<float> sigma_data = {
        1.0f // gradient from next layer, shape = (1, 1, 1, 1)
    };

    // Allocate input and sigma
    CudaData input(input_data);
    CudaData sigma(sigma_data);


    // === Forward: max_pool ===
    auto pool_result = input.max_pool(N, C, H, W, H_out, W_out, K, P, S);


    std::cout << "=== MaxPool Output ===\n";
    printDeviceData(pool_result.result.get(), N, C, H_out, W_out);

    // === Backward ===
    auto d_input = input.max_pool_backward_wr_input(
        pool_result.max_inds.get(),
        &sigma,
        N, C, H, W,
        N, C, H_out, W_out,
        K, P, S
    );

    std::cout << "\n=== Grad w.r.t Input ===\n";
    printDeviceData(d_input.get(), N, C, H, W);
    cudaFree(pool_result.max_inds.release());


    return 0;
}