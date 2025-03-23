#include "deviceData.h"
#include "CudaData.cuh"
#include "CPUData.h"
#include "IBATensor.h"

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

    std::vector<float>

    const int m = 20000;
    const int k = 20000;
    const int n = 20000;

    // Calculate the total number of elements in each matrix.
    size_t sizeA = static_cast<size_t>(m) * k;
    size_t sizeB = static_cast<size_t>(k) * n;

    // Initialize matrices with some test values.
    // For this example, we'll fill them with 1.0.
    std::vector<float> A(sizeA, 1.0f);
    std::vector<float> B(sizeB, 1.0f);

    // Create CudaData objects using the vector constructor.
    CudaData matrixA(A);
    CudaData matrixB(B);

    // Multiply the matrices.
    // This multiplies an m x k matrix with a k x n matrix.
    // The method returns a unique_ptr<DeviceData> that wraps the result.
    std::unique_ptr<DeviceData> result = matrixA.mat_mult(&matrixB, m, k, n);
}