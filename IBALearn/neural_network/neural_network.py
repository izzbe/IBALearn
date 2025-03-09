import numpy as np
import pandas as pd
import math
from abc import ABC, abstractmethod

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
    def __init__(self, indim, outdim):
        weights = None # should be (n, indim) (indim, outdim)
        bias = None

    def forward(self, X):
        return self.weights @ X + self.bias

    def backward(self):
        return None;

class ReLU(Layer):
    def __init__(self):
        return

    def forward(self, X):
        return

    def backward(self):
        return

