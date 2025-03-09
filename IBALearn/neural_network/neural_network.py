import numpy as np
import pandas as pd
import math
from abc import ABC, abstractmethod

class Tensor:
    def __init__(self, *args):
        return

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
        self.weights = np.random.rand(outdim, indim)
        self.bias = np.random.rand(outdim)

    def forward(self, X):
        return self.weights.T @ X + self.bias

    def backward(self):
        return

class ReLU(Layer):
    def __init__(self):
        return

    def forward(self, X):
        return

    def backward(self):
        return

