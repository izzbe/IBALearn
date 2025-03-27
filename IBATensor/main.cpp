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
    int N = 2, C_in = 2, C_out = 3;
    int H_in = 6, W_in = 6;
    int K = 3, P = 2, S = 2;

    int H_out = (H_in + 2 * P - K) / S + 1;
    int W_out = H_out;

    int input_size = N * C_in * H_in * W_in;
    int sigma_size = N * C_out * H_out * W_out;
    int kernel_size = C_out * C_in * K * K;

    // Fill input and sigma with sequential values
    std::vector<float> input_data(input_size), sigma_data(sigma_size), kernel_data(kernel_size);
    for (int i = 0; i < input_size; ++i) input_data[i] = static_cast<float>(i);
    for (int i = 0; i < sigma_size; ++i) sigma_data[i] = static_cast<float>(i);
    for (int i = 0; i < kernel_size; ++i) kernel_data[i] = static_cast<float>(i);

    Tensor input({N, C_in, H_in, W_in}, input_data, CUDA);
    Tensor sigma({N, C_out, H_out, W_out}, sigma_data, CUDA);
    Tensor kernel({C_out, C_in, K, K}, kernel_data, CUDA);

    Tensor dW = conv2d_backward_wr_kernel(input, sigma, kernel, P, S);
    Tensor dX = conv2d_backward_wr_input(input, sigma, kernel, P, S);

    std::cout << "=== ∂L/∂W (kernel grad) ===" << std::endl;
    dW.print();

    std::cout << "=== ∂L/∂X (input grad) ===" << std::endl;
    dX.print();

    return 0;
}