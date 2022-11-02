from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        return self.fn(y_true, y_pred)

    @abstractmethod
    def fn(self, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def grad(self, y_true, y_pred, **kwargs):
        raise NotImplementedError


class MeanSquaredError(Loss):
    def __init__(self):
        """A Mean-Squared-Error function"""
        super().__init__()

    def __str__(self):
        return "Mean-Squared-Error"

    def fn(self, y_true, y_pred):
        # multiply the squared error by 1/2 so when
        # we take the derivative the 2 in the power gets cancelled with the 1/2 multiplier
        # calculation is cleaner this way
        return 0.5 * np.linalg.norm(y_pred - y_true) ** 2

    def grad(self, y_true, y_pred):
        return y_pred - y_true