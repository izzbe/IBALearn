//
// Created by ianzh on 3/9/2025.
//
#include "IBATensor.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <memory>
namespace ibatensor {

// ---------------------------------------------- constructors --------------------------------------------------------
	std::unique_ptr<DeviceData> Tensor::construct_data(std::vector<float> data, int cuda_or_cpu) {
    	if (cuda_or_cpu == CPU) {
        	return std::make_unique<CPUData>(data);
        } else if (cuda_or_cpu == CUDA) {
        	return std::make_unique<CudaData>(data);
        }

    }

    Tensor::Tensor(int cuda_or_cpu) : size(0), data(nullptr) {
	    std::vector<float> empty_vec;
	    data = construct_data(empty_vec, cuda_or_cpu);
	}

    Tensor::Tensor(const std::vector<int>& shape, int cuda_or_cpu) : shape(shape), data(nullptr) {
        size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

	    std::vector<float> zeros;
	    for (int i = 0; i < size; i ++) {
	        zeros.push_back(0.0f);
	    }

	    data = construct_data(zeros, cuda_or_cpu);

        stride.resize(shape.size());
        int curr_stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            stride[i] = curr_stride;
            curr_stride *= shape[i];
        }
    }

    Tensor::Tensor(const std::vector<int>& shape, std::vector<float> values, int cuda_or_cpu) : size(size), shape(shape) {
        if (values.size()) {
            data = construct_data(values, cuda_or_cpu);
        }

	    stride.resize(shape.size());
	    int curr_stride = 1;
	    for (int i = shape.size() - 1; i >= 0; --i) {
	        stride[i] = curr_stride;
	        curr_stride *= shape[i];
	    }

    }

    Tensor::Tensor(const std::vector<int>& shape, std::unique_ptr<DeviceData> data) : data(std::move(data)),
                                                                                      size(this->data->getSize()) {}

// ---------------------------------------------- getters/setters --------------------------------------------------------
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
        
        return data->iloc(flat_idx);
    }

    const DeviceData *Tensor::getData() const {
	    return data.get();
	}

// ---------------------------------------------- matrixops --------------------------------------------------------


    void Tensor::print() const {
        
        if (shape.size() == 0 || data->getSize() == 0) {
            std::cout << "[]" << std::endl;
            return;
        }
        
        // 1D tensors
        if (shape.size() == 1) {
            std::cout << "[";
            for (int i = 0; i < shape[0]; i++) {
                std::cout << data->iloc(i);
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

    Tensor Tensor::operator%(const Tensor &other) const {
	    std::unique_ptr<DeviceData> data_new = data->mat_mult(other.getData(),
	                                                            this->shape[0],
	                                                            this->shape[1],
	                                                            other.shape[1]);
	    std::vector<int> shape_new = {this->shape[0], other.shape[1]};
	    return {shape_new, std::move(data_new)};
	}

    Tensor Tensor::operator*(const Tensor &other) const {
	    std::unique_ptr<DeviceData> data_new = data->elemMult(other.getData());
	    std::vector<int> shape_new = {this->shape[0], this->shape[1]};

	    return {shape_new, std::move(data_new)};
	}

    Tensor Tensor::operator+(const Tensor &other) const {
	    std::unique_ptr<DeviceData> data_new = data->elemAdd(other.getData());
	    std::vector<int> shape_new = {this->shape[0], this->shape[1]};

	    return {shape_new, std::move(data_new)};
	}

	Tensor Tensor::operator-(const Tensor &other) const {
		std::unique_ptr<DeviceData> data_new = data->elemSub(other.getData());
		std::vector<int> shape_new = {this->shape[0], this->shape[1]};

		return {shape_new, std::move(data_new)};
	}

	Tensor Tensor::ReLu() const {
		int H = shape[0];
		int W = (shape.size() >= 2) ? shape[1] : 1;
		int C = (shape.size() >= 3) ? shape[2] : 1;
		int N = (shape.size() >= 4) ? shape[3] : 1;
		std::unique_ptr<DeviceData> data_new = data->relu(H, W, C, N);
	}

	Tensor Tensor::conv2d(const Tensor &kernel, int padding, int stride) const {
		int H = this->shape[0];
		int W = (this->shape.size() >= 2) ? this->shape[1] : 1;
		int C = (this->shape.size() >= 3) ? this->shape[2] : 1;
		int N = (this->shape.size() >= 4) ? this->shape[3] : 1;

		int K = kernel.shape[0];
		int K_c = (kernel.shape.size() >= 3) ? kernel.shape[3] : 1;

		int H_out = ((H + ( 2 * padding ) - K) / stride) + 1;

		int W_out = ((W + ( 2 * padding ) - K) / stride) + 1;

		std::unique_ptr<DeviceData> data_new = data->conv2d(kernel.getData(),
															N, C, H, W, H_out, W_out, K,
															padding, stride, K_c);

		std::vector<int> size_new = {H_out, W_out, K_c, N};

		return {size_new, std::move(data_new)};
	}

	Tensor Tensor::avg_pool(int K, int padding, int stride) const {
		int H = this->shape[0];
		int W = (this->shape.size() >= 2) ? this->shape[1] : 1;
		int C = (this->shape.size() >= 3) ? this->shape[2] : 1;
		int N = (this->shape.size() >= 4) ? this->shape[3] : 1;

		int H_out = ((H + ( 2 * padding ) - K) / stride) + 1;

		int W_out = ((W + ( 2 * padding ) - K) / stride) + 1;

		std::unique_ptr<DeviceData> data_new = data->avg_pool(N, C, H, W, H_out, W_out, K, padding, stride);

		std::vector<int> size_new = {H_out, W_out, C, N};

		return {size_new, std::move(data_new)};

	}

	Tensor Tensor::max_pool(int K, int padding, int stride) const {
		int H = this->shape[0];
		int W = (this->shape.size() >= 2) ? this->shape[1] : 1;
		int C = (this->shape.size() >= 3) ? this->shape[2] : 1;
		int N = (this->shape.size() >= 4) ? this->shape[3] : 1;

		int H_out = ((H + ( 2 * padding ) - K) / stride) + 1;

		int W_out = ((W + ( 2 * padding ) - K) / stride) + 1;

		std::unique_ptr<DeviceData> data_new = data->max_pool(N, C, H, W, H_out, W_out, K, padding, stride);

		std::vector<int> size_new = {H_out, W_out, C, N};

		return {size_new, std::move(data_new)};

	}

	Tensor Tensor::mat_transpose() const {
		int H = this->shape[0];
		int W = (this->shape.size() >= 2) ? this->shape[1] : 1;
		int C = (this->shape.size() >= 3) ? this->shape[2] : 1;
		int N = (this->shape.size() >= 4) ? this->shape[3] : 1;

		std::unique_ptr<DeviceData> data_new = data->mat_transpose(H, W, C, N);
		std::vector<int> size_new = {H, W, C, N};

		return {size_new, std::move(data_new)};
	}



}
