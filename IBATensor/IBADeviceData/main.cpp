#include "deviceData.h"
#include "CudaData.cuh"
#include "CPUData.h"
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
    int N = 5, C_in = 3, C_out = 4;
    int H_in = 7, W_in = 7;
    int K = 3, S = 2, P = 2;

    int H_out = (H_in + 2 * P - K) / S + 1;
    int W_out = H_out;

    // Allocate input and sigma
    std::vector<float> x_data(N * C_in * H_in * W_in);
    std::vector<float> sigma_data(N * C_out * H_out * W_out);

    // Fill with sequential values
    for (int i = 0; i < x_data.size(); ++i) x_data[i] = static_cast<float>(i);
    for (int i = 0; i < sigma_data.size(); ++i) sigma_data[i] = static_cast<float>(i);


    // Create CUDA wrapper
    std::unique_ptr<DeviceData> x = std::make_unique<CudaData>(x_data);
    std::unique_ptr<DeviceData> sigma = std::make_unique<CudaData>(sigma_data);

    // Run conv2d_backward_wr_kernel
    std::cout << "func call:" << C_out << K << H_in << W_in << C_in << C_out << H_out << W_out << P << S << N << std::endl;
    auto kernel_grad = x->conv2d_backward_wr_kernel(
        sigma.get(),
        C_out, K,
        H_in, W_in,
        C_in,
        C_out,
        H_out, W_out,
        P, S, N
    );

    printDeviceData(kernel_grad.get(), C_out, C_in, K, K);
    std::cout << std::endl;

    return 0;
}