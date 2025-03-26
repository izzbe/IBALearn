//
// Created by ianzh on 3/12/2025.
//

#ifndef CPUDATA_H
#define CPUDATA_H
#include "deviceData.h"
#include <vector>
#include <stdexcept>
#include <memory>
class CPUData : public DeviceData {
    float *head;
    size_t size;
public:
    CPUData(size_t size);

    CPUData(const std::vector<float> &data_to_copy);

    CPUData(float *C, size_t size);

    CPUData(const DeviceData &other);

    CPUData &operator=(const DeviceData &other);

    CPUData(DeviceData &&other) noexcept;

    CPUData &operator=(DeviceData &&other) noexcept;

    ~CPUData();
    float *getData() const;
    size_t getSize() const;
    float *&getData();
    size_t &getSize();
    float iloc(int i) const;
    void set_index(int i, float val);

    std::unique_ptr<DeviceData> elemAdd(const DeviceData *other) const;
    std::unique_ptr<DeviceData> elemSub(const DeviceData *other) const;
    std::unique_ptr<DeviceData> elemMult(const DeviceData *other) const;
    std::unique_ptr<DeviceData> mat_mult(const DeviceData *other, int m, int k, int n) const;

    std::unique_ptr<DeviceData> conv2d(const DeviceData *kern, int N, int C_in,
                                             int H, int W, int H_out, int W_out, int K, int P, int S, int C_out) const;
    std::unique_ptr<DeviceData> avg_pool(int N, int C_in, int H, int W, int H_out, int W_out, int K, int P, int S) const;

    std::unique_ptr<DeviceData> max_pool(int N, int C_in, int H, int W, int H_out, int W_out, int K, int P, int S) const;

    std::unique_ptr<DeviceData> mat_transpose(int H, int W, int C, int N) const;

    std::unique_ptr<DeviceData> relu(int H, int W, int C, int N) const;

//    ----------------------------------------------------- backwards ----------------------------------------------------
    std::unique_ptr<DeviceData> conv2d_backward_wr_kernel(const DeviceData *sigma,
                                            int C_k, int K, int H_in, int W_in, int C_in, int C_sigma, int H_sigma, int W_sigma, int P, int S, int N) const;

    std::unique_ptr<DeviceData> conv2d_backward_wr_input(const DeviceData *sigma, const DeviceData *kernel, int H_in, int W_in, int K, int C_in_k, int C_out_k,
                                                                 int sigma_H, int sigma_W, int sigma_C, int sigma_N, int P, int S) const;




};

#endif //CPUDATA_H
