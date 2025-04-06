#include <pybind11/pybind11.h>
#include <pybind11/stl.h>        // for std::vector
#include <pybind11/numpy.h>      // for py::array_t if needed elsewhere
#include "IBATensor.h"

namespace py = pybind11;

PYBIND11_MODULE(ibatensor, m) {
    m.doc() = "Pybind11 bindings for ibatensor::Tensor and free functions";

    // ---------------------------------------------------------------------
    // Binding for max_pool_tensor_return (exposed as MaxPoolResult)
    // ---------------------------------------------------------------------
    py::class_<ibatensor::max_pool_tensor_return>(m, "MaxPoolResult")
        // Expose the output Tensor by reference (move-only Tensor)
        .def_property_readonly("output", [](const ibatensor::max_pool_tensor_return& self) -> const ibatensor::Tensor& {
            return self.output;
        }, "The Tensor result of max_pool.")
        // Expose the size of the max_inds array
        .def_property_readonly("max_inds_size", [](const ibatensor::max_pool_tensor_return& self) {
            return self.max_inds_size;
        }, "The number of indices in max_inds.")
        // Expose the GPU pointer as an integer (so it can be passed to backprop)
        .def_property_readonly("max_inds_ptr", [](const ibatensor::max_pool_tensor_return& self) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(self.max_inds);
        }, "The GPU pointer for max_inds, as an integer.");

    // ---------------------------------------------------------------------
    // Binding for the Tensor class
    // ---------------------------------------------------------------------
    py::class_<ibatensor::Tensor>(m, "Tensor")
        // Constructors
        .def(py::init<int>(), py::arg("cuda_or_cpu") = 0,
             "Construct an empty Tensor on CPU=0 or GPU=1.")
        .def(py::init<const std::vector<int>&, int>(), py::arg("shape"), py::arg("cuda_or_cpu") = 0,
             "Construct a Tensor with given shape and device type (CPU=0, GPU=1).")
        .def(py::init<const std::vector<int>&, std::vector<float>, int>(),
             py::arg("shape"), py::arg("values"), py::arg("cuda_or_cpu") = 0,
             "Construct a Tensor from shape, data values, and device type.")
        // Python convenience constructor from a NumPy float array:
        .def(py::init([](py::array_t<float> arr, int cuda_or_cpu) {
            py::buffer_info buf = arr.request();
            std::vector<int> shape(buf.shape.begin(), buf.shape.end());
            float* ptr = static_cast<float*>(buf.ptr);
            std::vector<float> values(ptr, ptr + buf.size);
            return ibatensor::Tensor(shape, values, cuda_or_cpu);
        }), py::arg("array"), py::arg("cuda_or_cpu") = 0,
           "Construct a Tensor from a NumPy float array on CPU=0 or GPU=1.")

        // Read-only properties
        .def_readonly("shape", &ibatensor::Tensor::shape, "Shape of the Tensor, e.g. [N, C, H, W].")
        .def_readonly("stride", &ibatensor::Tensor::stride, "Stride for each dimension.")
        .def_readonly("size", &ibatensor::Tensor::size, "Total number of elements in the Tensor.")

        // Methods
        .def("get", &ibatensor::Tensor::get, py::arg("indices"),
             "Get a single element by multi-dimensional indices.")
        .def("print", &ibatensor::Tensor::print,
             "Print the Tensor contents for debugging.")
        .def("to_string", &ibatensor::Tensor::to_string,
         "Return the Tensor contents as a string for Python printing.")

        // Operators / overloaded methods
        .def("__matmul__", &ibatensor::Tensor::operator%,
             "Matrix multiplication (mapped from operator%).")
        .def("elem_wise_mult", &ibatensor::Tensor::operator*,
             "Element-wise multiplication (operator*).")
        .def("elem_wise_add", &ibatensor::Tensor::operator+,
             "Element-wise addition (operator+).")
        .def("elem_wise_sub", &ibatensor::Tensor::operator-,
             "Element-wise subtraction (operator-).")

        // Additional operations
        .def("add_bias_for_conv2d", &ibatensor::Tensor::add_bias_for_conv2d,
             "Add a bias Tensor for conv2d.")
        .def("relu", &ibatensor::Tensor::ReLu,
             "Apply ReLU activation.")
        .def("softmax", &ibatensor::Tensor::softmax, 
             "Compute the softmax along the last dimension.")
        .def("conv2d", &ibatensor::Tensor::conv2d,
             py::arg("kernel"), py::arg("padding"), py::arg("stride"),
             "Perform a 2D convolution with the given kernel, padding, and stride.")
        .def("avg_pool", &ibatensor::Tensor::avg_pool,
             py::arg("K"), py::arg("padding"), py::arg("stride"),
             "Perform average pooling with window size K, padding, and stride.")
        .def("max_pool", &ibatensor::Tensor::max_pool,
             py::arg("K"), py::arg("padding"), py::arg("stride"),
             "Perform max pooling and return a MaxPoolResult object.")
        .def("mat_transpose", &ibatensor::Tensor::mat_transpose,
             "Transpose the matrix portion of the Tensor.");

    // ---------------------------------------------------------------------
    // Binding free functions
    // ---------------------------------------------------------------------

    m.def("conv2d_backward_wr_kernel",
          &ibatensor::conv2d_backward_wr_kernel,
          py::arg("input"), py::arg("sigma"), py::arg("kernel"),
          py::arg("padding"), py::arg("stride"),
          "Compute the gradient with respect to the kernel in a 2D convolution.");

    m.def("conv2d_backward_wr_input",
          &ibatensor::conv2d_backward_wr_input,
          py::arg("input"), py::arg("sigma"), py::arg("kernel"),
          py::arg("padding"), py::arg("stride"),
          "Compute the gradient with respect to the input in a 2D convolution.");

    // For max_pool_backward_wr_input, accept the GPU pointer as an integer.
    m.def("max_pool_backward_wr_input",
          [](const ibatensor::Tensor &input,
             const ibatensor::Tensor &sigma,
             uintptr_t max_inds_ptr,  // GPU pointer passed as integer
             int K, int padding, int stride) {
              // Cast the integer back to a GPU pointer (const int*)
              const int* ptr = reinterpret_cast<const int*>(max_inds_ptr);
              return ibatensor::max_pool_backward_wr_input(input, sigma, ptr, K, padding, stride);
          },
          py::arg("input"), py::arg("sigma"), py::arg("max_inds_ptr"),
          py::arg("K"), py::arg("padding"), py::arg("stride"),
          "Compute the gradient for max pooling using the GPU pointer for max_inds.");

    m.def("avg_pool_backward_wr_input",
          &ibatensor::avg_pool_backward_wr_input,
          py::arg("input"), py::arg("sigma"),
          py::arg("K"), py::arg("padding"), py::arg("stride"),
          "Compute the gradient with respect to the input for average pooling.");

    m.def("conv2d_backwards_bias_wr_sigma",
          &ibatensor::conv2d_backwards_bias_wr_sigma,
          py::arg("sigma"),
          "Compute the gradient with respect to the bias from conv2d sigma.");

    m.def("relu_backwards",
          &ibatensor::relu_backwards,
          py::arg("sigma"), py::arg("in"),
          "Compute the gradient through the ReLU activation.");

    m.def("bias_backwards",
          &ibatensor::bias_backwards,
          py::arg("sigma"),
          "Compute the gradient with respect to the bias.");
}
