//
// Created by ianzh on 3/9/2025.
//

#include <vector>
#include "IBADeviceData/deviceData.h"
#include "IBADeviceData/CudaData.cuh"
#ifndef IBATENSOR_H
#define IBATENSOR_H

const int CUDA = 1;
const int CPU = 0;

namespace ibatensor {
    class max_pool_tensor_return;
    class Tensor {
     public:
        std::unique_ptr<DeviceData> data;
        size_t size; // total elements
        std::vector<int> stride;
        std::vector<int> shape;
    private:
    	static std::unique_ptr<DeviceData> construct_data(std::vector<float> data, int cuda_or_cpu);
    public:
        Tensor(int cuda_or_cpu = 0);
        Tensor(const std::vector<int>& shape, int cuda_or_cpu = 0);
        Tensor(const std::vector<int>& shape, std::vector<float> values, int cuda_or_cpu = 0);
        Tensor(const std::vector<int>& shape, std::unique_ptr<DeviceData> data);
        Tensor(Tensor &&other);


        float get(std::vector<int> indices) const;
        void set(std::vector<int> indices, float value);
        void print_shape() const;
        void print() const;
        std::string to_string() const;
        const DeviceData *getData() const;

        Tensor operator%(const Tensor &other) const;
        Tensor operator*(const Tensor &other) const;
        Tensor operator+(const Tensor &other) const; //element wise addition
        Tensor operator-(const Tensor &other) const; //element wise subtraction
        Tensor add_bias_for_conv2d(const Tensor &bias) const;
        Tensor ReLu() const;
        Tensor softmax() const;
        Tensor conv2d(const Tensor &kernel, int padding, int stride) const;
        Tensor avg_pool(int K, int padding, int stride) const;


        max_pool_tensor_return max_pool(int K, int padding, int stride) const;
        Tensor mat_transpose() const;

    };

    class max_pool_tensor_return {
        public:
        Tensor output;
        int *max_inds;
        int max_inds_size;
        ~max_pool_tensor_return() {
            cudaFree(max_inds);
        }
        max_pool_tensor_return(Tensor T, int *max_inds, int max_inds_size) : output(std::move(T)), max_inds(max_inds), max_inds_size(max_inds_size) {}
        max_pool_tensor_return(max_pool_tensor_return && other) noexcept : output(std::move(other.output)), max_inds(other.max_inds), max_inds_size(other.max_inds_size) {
            other.max_inds = nullptr;
            other.max_inds_size = 0;
        }
    };

    Tensor conv2d_backward_wr_kernel(const Tensor &input, const Tensor &sigma, const Tensor &kernel, int padding, int stride);
    Tensor conv2d_backward_wr_input(const Tensor &input, const Tensor &sigma, const Tensor &kernel, int padding, int stride);

    Tensor max_pool_backward_wr_input(const Tensor &input, const Tensor &sigma, const int *max_inds, int K, int padding, int stride);
    Tensor avg_pool_backward_wr_input(const Tensor &input, const Tensor &sigma, int K, int padding, int stride);
    Tensor conv2d_backwards_bias_wr_sigma(const Tensor &sigma);
    Tensor relu_backwards(const Tensor &sigma, const Tensor &in);
    Tensor bias_backwards(const Tensor &sigma);

}

#endif //IBATENSOR_H
