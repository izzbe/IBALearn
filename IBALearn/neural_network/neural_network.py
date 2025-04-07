import numpy as np
import pandas as pd
import math
from abc import ABC, abstractmethod
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")
import ibatensor

if __name__ == "__main__":
    main()

CUDA = 1
learning_rate = ibatensor.Tensor(np.full((1,1,1,1), 0.1, dtype=np.float32), 1)
mew = ibatensor.Tensor(np.full((1,1,1,1), 0.9, dtype=np.float32), 1)
tensor_neg_one = ibatensor.Tensor(np.full((1,1,1,1), -1, dtype=np.float32), 1)

class Layer(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

class Linear(Layer):
    def __init__(self, indim : int, outdim : int):
        weights_initializer = np.random.rand(outdim, indim)
        bias_initializer = np.random.rand(outdim)

        weights_initializer.resize((1,1, indim, outdim))
        self.weights = ibatensor.Tensor(weights_initializer, CUDA)

        bias_initializer.resize((1,1,1, outdim))
        self.bias = ibatensor.Tensor(bias_initializer, CUDA)

        self.prev_update_weights = None
        self.prev_update_bias = None
        self.prev_input = None


    def forward(self, X):
        self.prev_input = X

        return (X @ self.weights).elem_wise_add(self.bias)

    def backward(self, sigma):
        grad_w =  self.prev_input.mat_transpose() @ sigma
        grad_b = ibatensor.bias_backwards(sigma)

        if(self.prev_update_weights == None):
            update_weights = grad_w.elem_wise_mult(learning_rate).elem_wise_mult(tensor_neg_one)
            self.weights = self.weights.elem_wise_sub(update_weights)
            update_bias = grad_b.elem_wise_mult(learning_rate).elem_wise_mult(learning_rate)
            self.bias = self.bias.elem_wise_sub(update_bias)

            self.prev_update_weights = update_weights
            self.prev_update_bias = update_bias
        else:
            update_weights = self.prev_update_weights.elem_wise_mult(mew).elem_wise_sub(grad_w.elem_wise_mult(learning_rate).elem_wise_mult(tensor_neg_one))
            self.weights = self.weights.elem_wise_sub(update_weights)
            update_bias = self.prev_update_weights.elem_wise_mult(mew).elem_wise_sub(grad_b.elem_wise_mult(learning_rate).elem_wise_mult(learning_rate))
            self.bias = self.bias.elem_wise_sub(update_bias)

            self.prev_update_weights = update_weights
            self.prev_update_bias = update_bias

        return sigma @ self.weights.mat_transpose()

class ReLU(Layer):
    def __init__(self):
        self.prev_output = None
        return

    def forward(self, X):
        out = X.relu()
        self.prev_output = out
        return out


    def backward(self, sigma):
        return ibatensor.relu_backwards(sigma, self.prev_output)

class output_layer(Layer):
    def __init__(self):
        self.prev_output = None
        return
    def forward(self, X):
        result = X.softmax()
        self.prev_output = result
        return X.softmax()

    def backward(self, Y):
        return self.prev_output.elem_wise_sub(Y)

def main():
    X = np.random.uniform(-1000, 1000, 1000)
    Y_1 = X <= 0
    Y_2 = X > 0