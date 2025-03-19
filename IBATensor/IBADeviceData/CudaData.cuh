#ifndef MATRIX_OPSV2_LIBRARY_CUH
#define MATRIX_OPSV2_LIBRARY_CUH
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "deviceData.h"
#include <stdexcept>

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




};

__host__ std::unique_ptr<DeviceData> mat_mult_base(const CudaData *A, const CudaData *B, int m, int k, int n);

__host__ std::unique_ptr<DeviceData> elem_wise(const CudaData *A, const CudaData *B, Operation o);

#endif //MATRIX_OPSV2_LIBRARY_CUH