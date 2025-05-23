#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "IBATensor.h"

namespace py = pybind11;

PYBIND11_MODULE(ibatensor, m) {
    m.doc() = "IBATensor";

    py::class_<ibatensor::Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const std::vector<int>&>())
        .def(py::init([](py::array_t<float> array) {
            // cast python array to vector<float>
            py::buffer_info buf = array.request();
            std::vector<int> shape;
            for (auto dim : buf.shape) {
                shape.push_back(dim);
            }
            float* data_ptr = static_cast<float*>(buf.ptr);
            std::vector<float> data(data_ptr, data_ptr + buf.size);

            return new ibatensor::Tensor(shape, data, 1);
        }))
        .def_readonly("shape", &ibatensor::Tensor::shape) 
        .def_readonly("stride", &ibatensor::Tensor::stride)
        .def_readonly("size", &ibatensor::Tensor::size)
        .def("print", &ibatensor::Tensor::print)
        .def("get", [](ibatensor::Tensor& self, std::vector<int> indices) {
            return self.get(indices);
        })
        .def("__matmul__", &ibatensor::Tensor::operator%)
        .def("conv2d", &ibatensor::Tensor::conv2d)
        ;
}