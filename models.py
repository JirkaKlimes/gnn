import numpy as np
import random

from abc import ABC, abstractmethod
from gnn import Gnn
from activations import Activation
from losses import Loss


class Model(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def build(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, trainX, trainY, **kwargs):
        raise NotImplementedError

class UnnamedModel1(Model):
    def __init__(self, gnn: Gnn):
        """
        A <insert some unique and cool name here> Model that:
            - adds new neurons and connections when loss stops droping
            - initializes weights diferently to ones that exists so they can learn new features
            - adds momentum to gradient for lower chance of getting stuck in local minima
        """

        self.gnn = gnn
        self._isBuilt = False
        self._initialize_hyperparameters()

        super().__init__()

    def __str__(self):
        return "Unnamed Model 1"

    def _initialize_hyperparameters(self):

        # activation functions that will be used
        self.activation_functions = []

        # loss function
        self.loss_fn = None

    def add_activation_functions(self, *funcs: Activation):
        for function in funcs:
            """
            Adds funtion to a list of activations that can/will be used
            """
            if self._isBuilt:
                raise Exception(f"Cannot add activation function to an already build network.")

            if str(function) in list(map(str, self.activation_functions)):
                raise Exception(f"Trainer already has \"{function}\" activation.")

            self.activation_functions.append(function)

    def set_loss_function(self, function: Loss):
        """
        Sets a function that will be used to computer network error and gradient
        """
        self.loss_fn = function

    def _fully_connect(self):
        """
        connected all inputs to all output neurons
        like in regular dense layer
        """
        for o in range(self.gnn.N_inputs, self.gnn.N_inputs+self.gnn.N_outputs):
            for i in range(self.gnn.N_inputs):
                weight_index = self.gnn.add_connection(i, o)
                self.gnn.weights[o, weight_index] = np.random.normal(size=(1))

            self.gnn.activation_functions_ids[o] = random.randint(0, len(self.activation_functions)-1)

    def build(self):
        if self.loss_fn is None:
            raise Exception("Cannot build, model is missing loss function.")
        self.gnn.loss_fn = self.loss_fn


        if not self.activation_functions:
            raise Exception("Cannot build, model doesn't have a single activation function.")
        self.gnn.activation_functions = tuple(self.activation_functions)

        self._fully_connect()
        self._isBuilt = True

    def train(self, trainX, trainY, target_loss: float = 0.01):
        """
        Trains the network on dataset
        """


