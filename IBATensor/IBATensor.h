//
// Created by ianzh on 3/9/2025.
//

#include <vector>

#ifndef IBATENSOR_H
#define IBATENSOR_H
class Tensor {
    std::vector<float> data;
    std::vector<int> size;
    std::vector<int> stride;
    std::vector<int> shape;

    class Iterator {
        std::vector<int> cur_loc;
        const std::vector<int> dim_order;
        Tensor &T;
        std::vector<int> shape;
        std::vector<int> stride_sizes;
        Iterator(std::vector<int> dim_order, Tensor &T, bool end = false);
        friend class Tensor;
    public:
        void operator++();
        float &operator*();
        bool operator==(const Iterator &other) const;
        bool operator!=(const Iterator &other) const;
    };

public:
    Tensor();
    Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape, std::vector<float> values);
    int index(std::vector<int> indices) const;
    float &get(std::vector<int> indices);
    void set(std::vector<int> indices, float value);
    void print_shape() const;

    Tensor operator*(Tensor other);
    Tensor operator+(Tensor other); //element wise addition
    Tensor operator/(Tensor other); //element wise division
    Tensor operator-(Tensor other); //element wise subtraction

    Tensor apply(Tensor other, float (*func)(float));

    Tensor::Iterator begin(std::vector<int> stride_order);
    Tensor::Iterator end(std::vector<int> stride_order);

};

#endif //IBATENSOR_H
