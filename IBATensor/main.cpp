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
    std::vector<int> shape = {1, 1, 2, 3}; // (N=1, C=1, H=2, W=3)

    // Input tensor with both positive and negative values
    std::vector<float> input_values = {
        1.0f, -2.0f, 3.0f,
        -4.0f, 5.0f, -6.0f
    };

    // Sigma (gradient) tensor with non-trivial values
    std::vector<float> sigma_values = {
        0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f
    };

    // Construct CUDA tensors
    Tensor input(shape, input_values, CUDA);
    std::cout <<input.to_string();
    Tensor sigma(shape, sigma_values, CUDA);
	std::cout <<sigma.to_string();
    // ReLU backward
    Tensor relu_back = relu_backwards(sigma, input);
    std::cout << "ReLU Backward Result:" << std::endl;
    relu_back.print();

    // Bias backward
    Tensor bias_back = bias_backwards(sigma);
    std::cout << "Bias Backward Result:" << std::endl;
    bias_back.print();

    // Conv2D backward bias
    Tensor conv2d_bias_back = conv2d_backwards_bias_wr_sigma(sigma);
    std::cout << "Conv2D Bias Backward Result:" << std::endl;
    std::cout << conv2d_bias_back.to_string();

    std::cout << "max_pool:" << std::endl;
    auto max_pool_res = input.max_pool(1, 0, 1);

    std::cout << max_pool_res.output.shape[0] << max_pool_res.output.shape[1] << max_pool_res.output.shape[2] << max_pool_res.output.shape[3] << std::endl;
    return 0;
}