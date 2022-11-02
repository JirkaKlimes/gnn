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
        pi, sqrt, tanh = np.pi, np.sqrt, np.tanh

        # return 0.5 * z * (1 + tanh(sqrt(2 / pi) * (z + 0.044715 * z ** 3)))

    def grad(self, z):
        pi, sqrt, tanh = np.pi, np.sqrt, np.tanh


class Tanh(Activation):
    def __init__(self):
        """A hyperbolic tangent activation function."""
        super().__init__()

    def __str__(self):
        return "Tanh"

    def fn(self, z):
        return np.tanh(z)

    def grad(self, x):
        return 1 - np.tanh(x) ** 2
