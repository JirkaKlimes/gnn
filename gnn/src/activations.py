from abc import ABC, abstractstaticmethod


class Activation(ABC):

    @abstractstaticmethod
    def fn(z: float) -> float:
        raise NotImplementedError

    @abstractstaticmethod
    def grad(z: float) -> float:
        raise NotImplementedError


class Linear(Activation):
    def fn(z: float) -> float:
        return z

    def grad(z: float) -> float:
        return 1


class ReLU(Activation):
    def fn(z: float) -> float:
        return max(0, z)

    def grad(z: float) -> float:
        return float(z > 0)
