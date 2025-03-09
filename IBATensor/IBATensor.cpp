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

Tensor::Iterator::Iterator(std::vector<int> dim_order, Tensor &T, bool end) :
                                                                cur_loc({0, 0, 0, 0}),
                                                                dim_order(std::move(dim_order)),
                                                                T(T),
                                                                shape(T.shape),
                                                                stride_sizes(T.stride)
{
    int last_dim = dim_order[dim_order.size() - 1];
    if (end == true) {
        cur_loc[last_dim] = shape[last_dim];
    }
}

void Tensor::Iterator::operator++() {
    int total_dims = dim_order.size();
    for (int dim = 0; dim < total_dims; dim++) {
        int dim_total_size = shape[dim];

        cur_loc[dim]++;

        if (cur_loc[dim] == dim_total_size && dim != total_dims - 1) {
            cur_loc[dim] == 0;
        } else {
            return;
        }
    }
}

float &Tensor::Iterator::operator*() {
    return T.get(cur_loc);
}

bool Tensor::Iterator::operator==(const Iterator &other) const {
    return (&other.T == &T) && (other.cur_loc == cur_loc);
}

bool Tensor::Iterator::operator!=(const Iterator &other) const {
    return !Tensor::Iterator::operator==(other);
}

Tensor::Iterator Tensor::begin(std::vector<int> stride_order) {
    return {std::move(stride_order), *this};
}

Tensor::Iterator Tensor::end(std::vector<int> stride_order) {
    return {std::move(stride_order), *this, true};
}

