### Neural network

This module implements a neural network class from scratch. A custom tensor library has been created under `~/IBATensor` in C++.

To build the C++ tensor library as a python module:
```bash
# from root directory
python setup.py build_ext --inplace

```

Then, the tensor library can be imported in python files using `from IBALearn.neural_network import ibatensor`

Run the tensor example:
```bash
# from root directory
python -m IBALearn.neural_network.tensor_example
```