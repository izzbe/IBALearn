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

using namespace ibatensor;

int main() {
    // --- Forward Pass ---
    // Create an input tensor of shape [5, 5] with values from 1 to 25.
    std::vector<int> input_shape = {5, 5};
    std::vector<float> input_values = {
        1,  2,  3,  4,  5,
        6,  7,  8,  9, 10,
       11, 12, 13, 14, 15,
       16, 17, 18, 19, 20,
       21, 22, 23, 24, 25
   };
    // Use CPU (0) for this test.
    Tensor input(input_shape, input_values, CUDA);

    // Create a convolution kernel of shape [3, 3].
    // For example, use a kernel that performs a simple edge-detection:
    //   1  0 -1
    //   1  0 -1
    //   1  0 -1
    std::vector<int> kernel_shape = {3, 3};
    std::vector<float> kernel_values = {
        1,  0, -1,
        1,  0, -1,
        1,  0, -1
   };
    Tensor kernel(kernel_shape, kernel_values, CUDA);

    // Define convolution parameters: no padding, stride 1.
    int padding = 0;
    int stride  = 1;

    // Perform the forward convolution.
    Tensor output = input.conv2d(kernel, padding, stride);
    std::cout << "Forward pass (Convolution output):" << std::endl;
    output.print();

    // --- Backward Pass ---
    // For the backward pass, assume the gradient (sigma) coming from the next layer
    // is a tensor of ones with the same shape as the output.
    // For a 5x5 input with a 3x3 kernel (padding=0, stride=1), output shape is [3, 3].
    std::vector<int> sigma_shape = output.shape;  // Expected: {3, 3}
    std::vector<float> sigma_values(sigma_shape[0] * sigma_shape[1], 1.0f);  // All ones.
    Tensor sigma(sigma_shape, sigma_values, CPU);

    // Compute the gradient with respect to the input using conv2d_backward.
    Tensor grad_input = input.conv2d_backward(sigma, kernel, padding, stride);
    std::cout << "Backward pass (Gradient with respect to input):" << std::endl;
    grad_input.print();

    return 0;
}