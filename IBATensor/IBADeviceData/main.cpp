#include <iostream>
#include <vector>
#include <memory>
#include "deviceData.h"
#include "CPUData.h"  // Include your CPUData implementation
#include "CudaData.cuh"

void printData(const std::string &name, const DeviceData &data) {
    std::cout << name << " (size " << data.getSize() << "): ";
    for (size_t i = 0; i < data.getSize(); i++) {
        std::cout << data.iloc(i) << " ";
    }
    std::cout << std::endl;
}

int main() {
    // Sample Data
    std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> dataB = {5.0f, 6.0f, 7.0f, 8.0f};

    size_t free_mem, total_mem;

    std::cout << "Free Memory: " << free_mem << " / " << total_mem << std::endl;
    // Create DeviceData instances (CPU-based for testing)
    std::unique_ptr<DeviceData> A = std::make_unique<CudaData>(dataA);

    std::cout << "Free Memory: " << free_mem << " / " << total_mem << std::endl;

    std::unique_ptr<DeviceData> B = std::make_unique<CudaData>(dataB);

    std::cout << "Free Memory: " << free_mem << " / " << total_mem << std::endl;

    // Print original data
    printData("A", *A);
    printData("B", *B);

    // Perform element-wise addition
    std::unique_ptr<DeviceData> C = A->elemAdd(B.get());
    printData("A + B", *C);

    std::cout << "Free Memory: " << free_mem << " / " << total_mem << std::endl;

    // Perform element-wise multiplication
    std::unique_ptr<DeviceData> D = A->elemMult(B.get());
    printData("A * B", *D);

    std::cout << "Free Memory: " << free_mem << " / " << total_mem << std::endl;

    // Perform matrix multiplication (assuming A and B are 2x2 matrices)
    std::unique_ptr<DeviceData> E = A->mat_mult(B.get(), 2, 2, 2);
    printData("A x B", *E);

    std::cout <<"Test" <<std::endl;

    A.reset();
    std::cout << "Deleted" << std::endl;

    B.reset();
    std::cout << "Deleted" << std::endl;

    C.reset();
    std::cout << "Deleted" << std::endl;

    D.reset();
    std::cout << "Deleted" << std::endl;

    E.reset();
    std::cout << "Deleted" << std::endl;

    return 0;
}