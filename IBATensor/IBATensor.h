//
// Created by ianzh on 3/9/2025.
//

#include <vector>

#ifndef IBATENSOR_H
#define IBATENSOR_H
class Tensor {
    std::vector<float> data;
    std::vector<float> size;
    std::vector<float> stride;

public:
    Tensor(std::vector<float> values, std::vector<int> shape);
    int index(std::vector<int> indices) const;
    float get(std::vector<int> indices) const;
    void set(std::vector<int> indices, float value);
    void print_shape() const;

    Tensor operator*(Tensor other);
    Tensor operator+(Tensor other); //element wise addition
    Tensor operator/(Tensor other); //element wise division
    Tensor operator-(Tensor other); //element wise subtraction

    Tensor apply(Tensor other, float (*func)(float));

};

#endif //IBATENSOR_H
