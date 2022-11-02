from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, z):
        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        raise NotImplementedError

    @abstractmethod
    def grad(self, z, **kwargs):
        raise NotImplementedError


class Sigmoid(Activation):
    def __init__(self):
        """A logistic sigmoid activation function."""
        super().__init__()

    def __str__(self):
        return "Sigmoid"

    def fn(self, z):
        return 1 / (1 + np.exp(-z))

    def grad(self, z):
        fn_z = self.fn(z)
        return fn_z * (1 - fn_z)

class Relu(Activation):
    def __init__(self):
        """A rectified linear activation function."""
        super().__init__()

    def __str__(self):
        return "ReLU"

    def fn(self, z):
        return np.clip(z, 0, np.inf)

    def grad(self, z):
        return (z > 0).astype(int)

class Kelu(Activation):
    def __init__(self, alpha=0.01):
        """A Modified Gelu which allows for gradient in negative values"""
        self.alpha = alpha
        super().__init__()

    def __str__(self):
        return f"KeLU (alpha={self.alpha})"

    def fn(self, z):
        e = np.e
        return (z / 1 + z**(-e*z)) + self.alpha*z
    
    def grad(self, z):
        e = np.e
        return (z*e**(1-e*z))/((e**(-e*z) + 1)**2)+1/(e**(-e*z) + 1)+self.alpha 
