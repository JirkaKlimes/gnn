from abc import ABC, abstractmethod


class Activation(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def fn(self, z: float) -> float:
        raise NotImplementedError

    __call__ = fn

    @abstractmethod
    def grad(self, z: float) -> float:
        raise NotImplementedError


class Linear(Activation):
    def fn(self, z: float) -> float:
        return z

    def grad(self, z: float) -> float:
        return 1


class ReLU(Activation):
    def fn(self, z: float) -> float:
        return max(0, z)

    def grad(self, z: float) -> float:
        return float(z > 0)
