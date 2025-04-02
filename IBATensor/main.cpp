#include "IBATensor.h"
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace ibatensor;

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
    const int N = 1, C = 1, H = 2, W = 2;
    const int K = 2, P = 0, S = 2;

    // Compute output size
    const int H_out = (H + 2 * P - K) / S + 1;
    const int W_out = (W + 2 * P - K) / S + 1;

    // Input: shape (1, 1, 2, 2)
    std::vector<float> input_data = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    Tensor input({N, C, H, W}, input_data, CUDA);
    std::cout << "Input:\n";
    input.print();

    // === Forward: avg_pool ===
    Tensor pooled = input.avg_pool(K, P, S);
    std::cout << "\nAvgPool Output:\n";
    pooled.print(); // Should be avg([1,2,3,4]) = 2.5

    // === Backward: assume dL/dY = 1.0 ===
    std::vector<float> sigma_data = {1.0f};
    Tensor sigma({N, C, H_out, W_out}, sigma_data, CUDA);

    Tensor dX = avg_pool_backward_wr_input(input, sigma, K, P, S);

    std::cout << "\nGrad w.r.t. Input (AvgPool Backward):\n";
    dX.print(); // Expect each value = 1.0 / 4 = 0.25

    return 0;
}