//
// Created by ianzh on 3/9/2025.
//
#include "IBATensor.h"

namespace ibatensor {
    Tensor::Tensor() : size(0) {}

    Tensor::Tensor(const std::vector<int>& shape) : shape(shape) {

    }

    Tensor::Tensor(const std::vector<int>& shape, std::vector<float> values) : Tensor(shape) {
 
    }
}