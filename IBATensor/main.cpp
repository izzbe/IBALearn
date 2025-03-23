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
    std::vector<float> A{1.00f, 2.00f, 3.00f, 4.00f, 5.00f, 6.00f, 7.00f, 8.00f, 9.00f};
    std::vector<float> B{10.00f, 11.00f, 12.00f ,13.00f ,14.00f, 15.00f, 16.00f, 17.00f, 18.00f, 19.00f};
    std::vector<int> shape_A{3, 3};
    std::vector<int> shape_B{3, 3};
    Tensor T_A(shape_A, A, CUDA);
    Tensor T_B(shape_B, B, CUDA);

    Tensor C = T_A * T_B;

    C.print();
    std::cout << C.size << std::endl;

    std::cout << C.getData()->getSize() <<std::endl;
}