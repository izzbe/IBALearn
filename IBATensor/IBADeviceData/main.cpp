#include "deviceData.h"
#include "CudaData.cuh"
#include "CPUData.h"
#include <iostream>
#include <vector>
#include <cstdlib>

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
    // We'll create a tensor (N=2, C=2, H=3, W=4).
    // That is 48 elements total. We'll fill from -10..37
    int N = 2;
    int C = 2;
    int H = 3;
    int W = 4;

    std::vector<float> hostData(N * C * H * W);
    for (int i = 0; i < (N*C*H*W); i++) {
        hostData[i] = static_cast<float>(i - 10); // values from -10..37
    }

    // Create your GPU-based data object
    std::unique_ptr<DeviceData> inputCuda = std::make_unique<CudaData>(hostData);

    // Print the original data
    std::cout << "=== Original Tensor (N=2, C=2, H=3, W=4) ===" << std::endl;
    printDeviceData(inputCuda.get(), N, C, H, W);

    // Call relu(H, W, C, N) => returns a new device data object with ReLU applied
    std::unique_ptr<DeviceData> reluCuda = inputCuda->relu(H, W, C, N);

    // Print the result
    std::cout << "=== After ReLU ===" << std::endl;
    printDeviceData(reluCuda.get(), N, C, H, W);

    return 0;
}