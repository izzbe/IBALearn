//
// Created by ianzh on 3/12/2025.
//

#ifndef DEVICEDATA_H


#define DEVICEDATA_H
#include <memory>

class DeviceData {

public:
    virtual ~DeviceData() = default;
    virtual float *getData() const = 0;
    virtual size_t getSize() const = 0;
    virtual float *&getData() = 0;
    virtual size_t &getSize() = 0;
    virtual float iloc(int i) const = 0;
    virtual void set_index(int i, float val) = 0;

    virtual std::unique_ptr<DeviceData> elemAdd(const DeviceData *other) const = 0;
    virtual std::unique_ptr<DeviceData> elemSub(const DeviceData *other) const = 0;
    virtual std::unique_ptr<DeviceData> elemMult(const DeviceData *other) const = 0;
    virtual std::unique_ptr<DeviceData> mat_mult(const DeviceData *other, int m, int k, int n) const = 0;

    virtual std::unique_ptr<DeviceData> conv2d(const DeviceData *kern, int N, int C_in,
                                             int H, int W, int H_out, int W_out, int K, int P, int S, int C_out) const = 0;
    virtual std::unique_ptr<DeviceData> avg_pool(int N, int C_in, int H, int W, int H_out, int W_out, int K, int P, int S) const = 0;

    virtual std::unique_ptr<DeviceData> max_pool(int N, int C_in, int H, int W, int H_out, int W_out, int K, int P, int S) const = 0;

    virtual std::unique_ptr<DeviceData> mat_transpose(int H, int W, int C, int N) const = 0;

    virtual std::unique_ptr<DeviceData> relu(int H, int W, int C, int N) const = 0;

    virtual std::unique_ptr<DeviceData> conv2d_backward_wr_kernel(const DeviceData *sigma,
                                            int C_k, int K, int H_in, int W_in, int C_in, int C_sigma, int H_sigma, int W_sigma, int P, int S, int N) const = 0;

    virtual std::unique_ptr<DeviceData> conv2d_backward_wr_input(const DeviceData *sigma, const DeviceData *kernel, int H_in, int W_in, int K, int C_in_k, int C_out_k,
                                                                 int sigma_H, int sigma_W, int sigma_C, int sigma_N, int P, int S) const = 0;



};

#endif //DEVICEDATA_H
