//
// Created by ianzh on 3/9/2025.
//

#include <vector>
#include "IBADeviceData/deviceData.h"
#include "IBADeviceData/CudaData.cuh"
#include "IBADeviceData/CPUData.h"
#ifndef IBATENSOR_H
#define IBATENSOR_H

const int CUDA = 1;
const int CPU = 0;

namespace ibatensor {

    class Tensor {
     public:
        std::unique_ptr<DeviceData> data;
        int size; // total elements
        std::vector<int> stride;
        std::vector<int> shape;
    private:
    	static std::unique_ptr<DeviceData> construct_data(std::vector<float> data, int cuda_or_cpu);
    public:
        Tensor(int cuda_or_cpu);
        Tensor(const std::vector<int>& shape, int cuda_or_cpu);
        Tensor(const std::vector<int>& shape, std::vector<float> values, int cuda_or_cpu);

        float get(std::vector<int> indices) const;
        void set(std::vector<int> indices, float value);
        void print_shape() const;
        void print() const;

        Tensor operator*(Tensor other);
        Tensor operator+(Tensor other); //element wise addition
        Tensor operator/(Tensor other); //element wise division
        Tensor operator-(Tensor other); //element wise subtraction

        Tensor apply(Tensor other, float (*func)(float));


    };

}

#endif //IBATENSOR_H
