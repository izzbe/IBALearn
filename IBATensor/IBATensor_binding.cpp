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
            py::buffer_info buf = array.request();
            std::vector<int> shape;
            for (auto dim : buf.shape) {
                shape.push_back(dim);
            }
            float* data_ptr = static_cast<float*>(buf.ptr);
            std::vector<float> data(data_ptr, data_ptr + buf.size);

            return new ibatensor::Tensor(shape, data);
        }));
        // .def("__repr__", [](const ibatensor::Tensor& self) {
        //     std::stringstream ss;
        //     ss << "Tensor(shape=[";
        //     const auto& shape = self.shape();
        //     for (size_t i = 0; i < shape.size(); ++i) {
        //         ss << shape[i];
        //         if (i < shape.size() - 1) ss << ", ";
        //     }
        //     ss << "])";
        //     return ss.str();
        // });
}