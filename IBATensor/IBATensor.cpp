//
// Created by ianzh on 3/9/2025.
//
#include "IBATensor.h"
#include <algorithm>
#include <numeric>
#include <iostream>

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
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("Number of indices doesn't match tensor dimensions");
        }
        
        // calculate actual index in vector using strides
        int flat_idx = 0;
        for (int i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            flat_idx += indices[i] * stride[i];
        }
        
        return data[flat_idx];    
    }

    float Tensor::get(std::vector<int> indices) const {
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("Number of indices doesn't match tensor dimensions");
        }
        
        // calculate actual index in vector using strides
        int flat_idx = 0;
        for (int i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            flat_idx += indices[i] * stride[i];
        }
        
        return data[flat_idx];  
    }


    void Tensor::print() const {
        
        if (shape.size() == 0 || data.empty()) {
            std::cout << "[]" << std::endl;
            return;
        }
        
        // 1D tensors
        if (shape.size() == 1) {
            std::cout << "[";
            for (int i = 0; i < shape[0]; i++) {
                std::cout << data[i];
                if (i < shape[0] - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        } 
        // 2D tensors
        if (shape.size() == 2) {
            std::cout << "[";
            for (int i = 0; i < shape[0]; i++) {
                std::cout << "[";
                for (int j = 0; j < shape[1]; j++) {
                    std::cout << get({i, j});
                    if (j < shape[1] - 1) std::cout << ", ";
                }
                std::cout << "]";
                if (i < shape[0] - 1) std::cout << ",";
            }
            std::cout << "]" << std::endl;
        }
        // todo: print higher dimensional tensors
        else
            std::cout << "3+ dimensional tensor print not yet implemented" << std::endl;
    }

}