//
// Created by ianzh on 3/9/2025.
//

#include <vector>

#ifndef IBATENSOR_H
#define IBATENSOR_H


namespace ibatensor {

    std::vector<int> range(int size, int start = 0);

    std::vector<int> values(int val, int size);

    class Tensor {
        public:
        std::vector<float> data;
        int size; // total elements
        std::vector<int> stride;
        std::vector<int> shape;
    public:
        Tensor();
        Tensor(const std::vector<int>& shape);
        Tensor(const std::vector<int>& shape, std::vector<float> values);
        int index(std::vector<int> indices) const;
        float& get(std::vector<int> indices);
        float get(std::vector<int> indices) const;
        void set(std::vector<int> indices, float value);
        void print_shape() const;
        void print() const;

        std::vector<Tensor> split();
        Tensor operator*(Tensor other);
        Tensor operator+(Tensor other); //element wise addition
        Tensor operator/(Tensor other); //element wise division
        Tensor operator-(Tensor other); //element wise subtraction

        Tensor apply(Tensor other, float (*func)(float));

    };

}

#endif //IBATENSOR_H
