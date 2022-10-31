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