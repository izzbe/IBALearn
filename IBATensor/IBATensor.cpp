//
// Created by ianzh on 3/9/2025.
//
#include "IBATensor.h"
#include <algorithm>
#include <numeric>

namespace ibatensor {
    Tensor::Tensor() : size(0) {}

    Tensor::Tensor(const std::vector<int>& shape) : shape(shape) {
        size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        data = std::vector<float>(size);

        stride.resize(shape.size());
        int curr_stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            stride[i] = curr_stride;
            curr_stride *= shape[i];
        }
    }

    Tensor::Tensor(const std::vector<int>& shape, std::vector<float> values) : Tensor(shape) {
        if (values.size()) {
            data = values;
        }
    }
}