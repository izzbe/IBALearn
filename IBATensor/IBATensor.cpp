//
// Created by ianzh on 3/9/2025.
//
#include "IBATensor.h"
#include <algorithm>
#include <numeric>

namespace ibatensor {
    std::vector<int> range(int size, int start = 0) {
        std::vector<int> vec;
        for (int i = start; i < size; ++i) {
            vec.push_back(i);
        }
        return vec;
    }

    std::vector<int> values(int val, int size) {
        std::vector<int> vec;
        for (int i = 0; i < size; ++i) {
            vec.push_back(val);
        }
        return vec;
    }

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

    float& Tensor::get(std::vector<int> indices) {
        return data[0];
    }

    Tensor::Iterator::Iterator(Tensor &T, bool end, std::vector<int> slice, std::vector<int> dim_order) :
                                                                    cur_loc({0, 0, 0, 0}),
                                                                    T(T),
                                                                    shape(T.shape),
                                                                    stride_sizes(T.stride),
                                                                    dim_order(),
                                                                    slice{std::move(slice)}
    {

        for (int i = 0; i < slice.size(); i++) {
            if (slice[i] != -1) {
                cur_loc[i] = slice[i];
            } else if (dim_order.empty()) {
                dim_order.push_back(i);
            }
        }

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

}