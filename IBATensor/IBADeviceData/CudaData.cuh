#ifndef MATRIX_OPSV2_LIBRARY_CUH
#define MATRIX_OPSV2_LIBRARY_CUH
#include <iostream>
#include <vector>
#include "deviceData.h"
#include <stdexcept>
#include <cuda_runtime.h>

const int TILE_SIZE = 16;
enum class Operation { Add, Sub, Mult };
enum class Pool {Average, Max};

class CudaData : public DeviceData {
private:
    float *head;
    size_t size;

public:
    CudaData(size_t size);

    CudaData(const std::vector<float> &data_to_copy);

    CudaData(float *C, size_t size);

    CudaData(const DeviceData &other);

    CudaData &operator=(const DeviceData &other);

    CudaData(DeviceData &&other) noexcept;

    CudaData &operator=(DeviceData &&other) noexcept;

    ~CudaData();

    float *getData() const;
    size_t getSize() const;
    float *&getData();
    size_t &getSize();
    float iloc(int i) const;
    void set_index(int i, float val);

    std::unique_ptr<DeviceData> elemAdd(const DeviceData *other, int B_grouping, int B_stride) const;
    std::unique_ptr<DeviceData> elemSub(const DeviceData *other, int B_grouping, int B_stride) const;
    std::unique_ptr<DeviceData> elemMult(const DeviceData *other, int B_grouping, int B_stride) const;
    std::unique_ptr<DeviceData> mat_mult(const DeviceData *other, int H, int shared_axis, int W, int N) const;

    std::unique_ptr<DeviceData> conv2d(const DeviceData *kern, int N, int C_in,
                                             int H, int W, int H_out, int W_out, int K, int P, int S, int C_out) const;

    std::unique_ptr<DeviceData> avg_pool(int N, int C_in, int H, int W, int H_out, int W_out, int K, int P, int S) const;

    DeviceData::max_pool_return max_pool(int N, int C_in, int H, int W, int H_out, int W_out, int K, int P, int S) const;

    std::unique_ptr<DeviceData> mat_transpose(int H, int W, int C, int N) const;

    std::unique_ptr<DeviceData> relu(int H, int W, int C, int N) const;

    // ------------------------------------------------------------------- BACKWARDS -------------------------------------------------------------

    std::unique_ptr<DeviceData> conv2d_backward_wr_kernel(const DeviceData *sigma,
                                            int C_k, int K, int H_in, int W_in, int C_in, int C_sigma, int H_sigma, int W_sigma, int P, int S, int N) const;

    std::unique_ptr<DeviceData> conv2d_backward_wr_input(const DeviceData *sigma, const DeviceData *kernel, int H_in, int W_in, int K, int C_in_k, int C_out_k,
                                                                 int sigma_H, int sigma_W, int sigma_C, int sigma_N, int P, int S) const;

    std::unique_ptr<DeviceData> max_pool_backward_wr_input(const int *max_inds,
                                                           const DeviceData *sigma, int N_in, int C_in, int H_in,
                                                           int W_in, int N_sigma, int C_sigma, int H_sigma,
                                                           int W_sigma, int K, int P, int S) const;

    std::unique_ptr<DeviceData> avg_pool_backward_wr_input(const DeviceData *sigma,
                                                                int N_in, int C_in, int H_in, int W_in,
                                                                int N_sigma, int C_sigma, int H_sigma, int W_sigma,
                                                                int K, int P, int S) const;

};






#endif //MATRIX_OPSV2_LIBRARY_CUH