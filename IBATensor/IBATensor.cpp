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
		if (cuda_or_cpu == CUDA) {
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

    Tensor::Tensor(const std::vector<int>& shape, std::vector<float> values, int cuda_or_cpu) : size(values.size()), shape(shape) {
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
                                                                                      size(this->data->getSize()),
																					  shape(shape) {}

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

		int N = shape[0];
		int C = (shape.size() >= 2) ? shape[1] : 1;
		int H = (shape.size() >= 3) ? shape[2] : 1;
		int W = (shape.size() >= 4) ? shape[3] : 1;

		for (int n = 0; n < N; ++n) {
			for (int c = 0; c < C; ++c) {
				std::cout << "\n=== N = " << n << ", C = " << c << " ===\n";
				for (int h = 0; h < H; ++h) {
					for (int w_ = 0; w_ < W; ++w_) {
						// Compute the linear index in your flattened array
						int idx = n * C * H * W
								  + c * H * W
								  + h * W
								  + w_;
						std::cout << data->iloc(idx) << "\t";
					}
					std::cout << "\n";
				}
			}
		}
		std::cout << std::endl;
	}

    Tensor Tensor::operator%(const Tensor &other) const {
		int N = this->shape[0];
		int shared_axis = this->shape[3];
		int H = this->shape[2];
		int W = other.shape[3];

	    std::unique_ptr<DeviceData> data_new = data->mat_mult(other.getData(),
	                                                            H,
	                                                            shared_axis,
	                                                            W,
	                                                            N);
	    std::vector<int> shape_new = {N, 1, H, W};
	    return {shape_new, std::move(data_new)};
	}

    Tensor Tensor::operator*(const Tensor &other) const {
	    std::unique_ptr<DeviceData> data_new = data->elemMult(other.getData(), other.getData()->getSize(), 1);
	    std::vector<int> shape_new = {this->shape[0], this->shape[1], this->shape[2], this->shape[3]};

	    return {shape_new, std::move(data_new)};
	}

    Tensor Tensor::operator+(const Tensor &other) const {
	    std::unique_ptr<DeviceData> data_new = data->elemAdd(other.getData(), other.getData()->getSize(), 1);
	    std::vector<int> shape_new = {this->shape[0], this->shape[1], this->shape[2], this->shape[3]};

	    return {shape_new, std::move(data_new)};
	}

	Tensor Tensor::operator-(const Tensor &other) const {
		std::unique_ptr<DeviceData> data_new = data->elemSub(other.getData(), other.getData()->getSize(), 1);
		std::vector<int> shape_new = {this->shape[0], this->shape[1], this->shape[2], this->shape[3]};

		return {shape_new, std::move(data_new)};
	}

	Tensor Tensor::add_bias_for_conv2d(const Tensor &bias) const {
		int B_grouping = this->shape[1];
		int B_stride = this->shape[2] * this->shape[3];
		std::unique_ptr<DeviceData> data_new = data->elemAdd(bias.getData(), B_grouping, B_stride);
		std::vector<int> shape_new = {this->shape[0], this->shape[1], this->shape[2], this->shape[3]};

		return {shape_new, std::move(data_new)};
	}

	Tensor Tensor::ReLu() const {
		int N = shape[0];
		int C = (shape.size() >= 2) ? shape[1] : 1;
		int H = (shape.size() >= 3) ? shape[2] : 1;
		int W = (shape.size() >= 4) ? shape[3] : 1;
		std::unique_ptr<DeviceData> data_new = data->relu(H, W, C, N);
		std::vector<int> shape_new = {N, C, H, W};
		return {shape_new , std::move(data_new)};
	}

	Tensor Tensor::conv2d(const Tensor &kernel, int padding, int stride) const {
		int N = this->shape[0];
		int C = (this->shape.size() >= 2) ? this->shape[1] : 1;
		int H = (this->shape.size() >= 3) ? this->shape[2] : 1;
		int W = (this->shape.size() >= 4) ? this->shape[3] : 1;

		int K_c = kernel.shape[0];
		int K = (kernel.shape.size() >= 3) ? kernel.shape[3] : 1;

		int H_out = ((H + ( 2 * padding ) - K) / stride) + 1;

		int W_out = ((W + ( 2 * padding ) - K) / stride) + 1;

		std::unique_ptr<DeviceData> data_new = data->conv2d(kernel.getData(),
															N, C, H, W, H_out, W_out, K,
															padding, stride, K_c);

		std::vector<int> size_new = {N, K_c, H_out, W_out};

		return {size_new, std::move(data_new)};
	}

	Tensor Tensor::avg_pool(int K, int padding, int stride) const {
		int N= this->shape[0];
		int C = (this->shape.size() >= 2) ? this->shape[1] : 1;
		int H = (this->shape.size() >= 3) ? this->shape[2] : 1;
		int W = (this->shape.size() >= 4) ? this->shape[3] : 1;

		int H_out = ((H + ( 2 * padding ) - K) / stride) + 1;

		int W_out = ((W + ( 2 * padding ) - K) / stride) + 1;

		std::unique_ptr<DeviceData> data_new = data->avg_pool(N, C, H, W, H_out, W_out, K, padding, stride);

		std::vector<int> size_new = {N, C, H_out, W_out};

		return {size_new, std::move(data_new)};

	}

	max_pool_tensor_return Tensor::max_pool(int K, int padding, int stride) const {
		int N = this->shape[0];
		int C = (this->shape.size() >= 2) ? this->shape[1] : 1;
		int H = (this->shape.size() >= 3) ? this->shape[2] : 1;
		int W = (this->shape.size() >= 4) ? this->shape[3] : 1;

		int H_out = ((H + ( 2 * padding ) - K) / stride) + 1;

		int W_out = ((W + ( 2 * padding ) - K) / stride) + 1;

		DeviceData::max_pool_return data_new = data->max_pool(N, C, H, W, H_out, W_out, K, padding, stride);

		std::vector<int> size_new = {N, C, H_out, W_out};

		return {Tensor(size_new, std::move(data_new.result)), std::move(data_new.max_inds)};

	}

	Tensor Tensor::mat_transpose() const {
		int N = this->shape[0];
		int C = (this->shape.size() >= 2) ? this->shape[1] : 1;
		int H = (this->shape.size() >= 3) ? this->shape[2] : 1;
		int W = (this->shape.size() >= 4) ? this->shape[3] : 1;

		std::unique_ptr<DeviceData> data_new = data->mat_transpose(H, W, C, N);
		std::vector<int> size_new = {N, C, W, H};

		return {size_new, std::move(data_new)};
	}

	Tensor conv2d_backward_wr_kernel(const Tensor &input, const Tensor &sigma, const Tensor &kernel, int padding, int stride) {

		int N_in = input.shape[0];
		int C_in = (input.shape.size() >= 2) ? input.shape[1] : 1;
		int H_in = (input.shape.size() >= 3) ? input.shape[2] : 1;
		int W_in = (input.shape.size() >= 4) ? input.shape[3] : 1;

		std::cout << "input: (" << H_in << ", " << W_in << ", " << C_in << ", " << N_in << ")" << std::endl;
		int N_sigma = sigma.shape[0];
		int C_sigma = (sigma.shape.size() >= 2) ? sigma.shape[1] : 1;
		int H_sigma = (sigma.shape.size() >= 3) ? sigma.shape[2] : 1;
		int W_sigma = (sigma.shape.size() >= 4) ? sigma.shape[3] : 1;

		std::cout << "sigma: (" << H_sigma << ", " << W_sigma << ", " << C_sigma << ", " << N_sigma << ")" << std::endl;
		int C_out_kernel = kernel.shape[0];
		int C_in_kernel = (kernel.shape.size() >= 2) ? kernel.shape[1] : 1;
		int K_kernel = (kernel.shape.size() >= 3) ? kernel.shape[2] : 1;

		std::cout << "Kernel: (" << K_kernel << ", " << K_kernel << ", " << C_in_kernel << ", " << C_out_kernel << ")" << std::endl;

		std::vector<int> shape_new = {C_out_kernel, C_in_kernel, K_kernel, K_kernel};
		std::cout << "func call:" << C_out_kernel << K_kernel << H_in << W_in << C_in << C_sigma << H_sigma << W_sigma << padding << stride << N_in << std::endl;
 		std::unique_ptr<DeviceData>data_new = input.getData()->conv2d_backward_wr_kernel(sigma.data.get(), C_out_kernel, K_kernel, H_in, W_in, C_in, C_sigma, H_sigma, W_sigma, padding, stride, N_in);
		return {shape_new, std::move(data_new)};
	}

	Tensor conv2d_backward_wr_input(const Tensor &input, const Tensor &sigma, const Tensor &kernel, int padding, int stride) {
		int N_in= input.shape[0];
		int C_in = (input.shape.size() >= 2) ? input.shape[1] : 1;
		int H_in = (input.shape.size() >= 3) ? input.shape[2] : 1;
		int W_in = (input.shape.size() >= 4) ? input.shape[3] : 1;

		int N_sigma= sigma.shape[0];
		int C_sigma = (sigma.shape.size() >= 2) ? sigma.shape[1] : 1;
		int H_sigma = (sigma.shape.size() >= 3) ? sigma.shape[2] : 1;
		int W_sigma = (sigma.shape.size() >= 4) ? sigma.shape[3] : 1;

		int C_out_kernel = kernel.shape[0];
		int C_in_kernel = (kernel.shape.size() >= 2) ? kernel.shape[1] : 1;
		int K_kernel = (kernel.shape.size() >= 3) ? kernel.shape[2] : 1;

		std::vector<int> shape_new = {N_in, C_in, H_in, W_in};
		std::unique_ptr<DeviceData>data_new = input.data->conv2d_backward_wr_input(sigma.data.get(), kernel.data.get(), H_in, W_in, K_kernel, C_in_kernel, C_out_kernel,
																					H_sigma, W_sigma, C_sigma, N_sigma, padding, stride);

		return {shape_new, std::move(data_new)};
	}

	Tensor max_pool_backward_wr_input(const Tensor &input, const Tensor &sigma, const int *max_inds, int K, int padding, int stride) {
		int N_in= input.shape[0];
		int C_in = (input.shape.size() >= 2) ? input.shape[1] : 1;
		int H_in = (input.shape.size() >= 3) ? input.shape[2] : 1;
		int W_in = (input.shape.size() >= 4) ? input.shape[3] : 1;

		int N_sigma= sigma.shape[0];
		int C_sigma = (sigma.shape.size() >= 2) ? sigma.shape[1] : 1;
		int H_sigma = (sigma.shape.size() >= 3) ? sigma.shape[2] : 1;
		int W_sigma = (sigma.shape.size() >= 4) ? sigma.shape[3] : 1;

		std::unique_ptr<DeviceData> data_new = input.data->max_pool_backward_wr_input(max_inds, sigma.data.get(), N_in, C_in, H_in, W_in, N_sigma, C_sigma, H_sigma, W_sigma, K, padding, stride );
		std::vector<int> size_new = {N_in, C_in, H_in, W_in};

		return {size_new, std::move(data_new)};
	}

	Tensor avg_pool_backward_wr_input(const Tensor &input, const Tensor &sigma, int K, int padding, int stride) {
		int N_in= input.shape[0];
		int C_in = (input.shape.size() >= 2) ? input.shape[1] : 1;
		int H_in = (input.shape.size() >= 3) ? input.shape[2] : 1;
		int W_in = (input.shape.size() >= 4) ? input.shape[3] : 1;

		int N_sigma= sigma.shape[0];
		int C_sigma = (sigma.shape.size() >= 2) ? sigma.shape[1] : 1;
		int H_sigma = (sigma.shape.size() >= 3) ? sigma.shape[2] : 1;
		int W_sigma = (sigma.shape.size() >= 4) ? sigma.shape[3] : 1;

		std::unique_ptr<DeviceData> data_new = input.data->avg_pool_backward_wr_input(sigma.data.get(), N_in, C_in, H_in, W_in,
																						N_sigma, C_sigma, H_sigma, W_sigma,
																						K, padding, stride);
		std::vector<int> shape_new = {N_in, C_in, H_in, W_in};
		return {shape_new, std::move(data_new)};
	}


}
