#ifndef MATRIX_OPSV2_LIBRARY_CUH
#define MATRIX_OPSV2_LIBRARY_CUH
#include <iostream>
#include <cuda_runtime.h>
#include <vector>

const int TILE_SIZE = 16;

enum class Operation;

class Data {
private:
    float *head;
    size_t size;

public:
    Data(size_t size);
    Data(const std::vector<float> &data_to_copy, size_t size);

    Data(float *C, size_t size);

    Data(const Data &other);

    Data &operator=(const Data &other);

    Data(Data &&other) noexcept;

    Data &operator=(Data &&other) noexcept;

    ~Data();

    float *getData() const;

    size_t getSize() const;
};

__host__ Data cuda_allocate(const std::vector<float> &data);

__host__ Data mat_mult(Data A, Data B, int m, int k, int n);

__host__ Data elem_wise(Data A, Data B, Operation o);

#endif //MATRIX_OPSV2_LIBRARY_CUH