from abc import ABC, abstractmethod
from math import tanh

class Activation(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, z):
        return self.fn(z)

    @abstractmethod
    def fn(self, z) -> float:
        raise NotImplementedError

    @abstractmethod
    def grad(self, z) -> float:
        raise NotImplementedError

    @abstractmethod
    def cuda_fn(z) -> callable:
        raise NotImplementedError

    @abstractmethod
    def cuda_grad(z) -> callable:
        raise NotImplementedError

class Relu(Activation):
    def __init__(self):
        """A rectified linear activation function."""
        super().__init__()

    def __str__(self):
        return "ReLU"

    def fn(self, z):
        return max(0, z)

    def grad(self, z):
        return 0 if z < 0 else 1

    def cuda_fn(self):
        def fn(z):
            return max(0, z)
        return fn

    def cuda_grad(self):
        def fn(z):
            return 0 if z < 0 else 1
        return fn

class Identity(Activation):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Identity"

    def fn(self, z):
        return z

    def grad(self, z):
        return 1

    def cuda_fn(z):
        def fn(z):
            return z
        return fn
    
    def cuda_grad(z):
        def fn(z):
            return 1
        return fn


class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Tanh"

    def fn(self, z):
        return tanh(z)

    def grad(self, z):
        return 1 - tanh(z)**2

    def cuda_fn(z):
        def fn(z):
            return tanh(z)
        return fn
    
    def cuda_grad(z):
        def fn(z):
            return 1 - tanh(z)**2
        return fn

