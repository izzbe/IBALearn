//
// Created by ianzh on 3/9/2025.
//

#include <vector>

#ifndef IBATENSOR_H
#define IBATENSOR_H


namespace ibatensor{

    std::vector<int> range(int size, int start = 0);

    std::vector<int> values(int val, int size);

    class Tensor {
        public:
        std::vector<float> data;
        int size; // total elements
        std::vector<int> stride;
        std::vector<int> shape;

        class Iterator {
            std::vector<int> cur_loc;
            Tensor &T;
            std::vector<int> shape;
            std::vector<int> stride_sizes;
            const std::vector<int> dim_order;
            std::vector<int> slice;
            explicit Iterator(Tensor &T, bool end = false,
                std::vector<int> slice = values(-1, T.shape.size()), std::vector<int> dim_order = {});
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

        Tensor::Iterator begin(std::vector<int> stride_order);
        Tensor::Iterator end(std::vector<int> stride_order);

    };

}

#endif //IBATENSOR_H
