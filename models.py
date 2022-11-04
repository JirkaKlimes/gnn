import numpy as np
import random

from abc import ABC, abstractmethod
from gnn import Gnn
from activations import Activation
from losses import Loss


class Model(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self._initialize_hyperparameters()

    def _initialize_hyperparameters(self):

        # activation functions for hidden and output neurons
        self.hidden_act_fn = None
        self.output_act_fn = None

        # loss function
        self.loss_fn = None

    def set_hidden_act_function(self, func: Activation):
        """
        Sets atctivation that will be used for hiddne neurons
        """

        if self._isBuilt:
            raise Exception(f"Cannot set activation function for an already build network.")

        if self.hidden_act_fn is not None:
            raise Exception(f"Hidden activation function was already set.")

        self.hidden_act_fn = func

    def set_output_act_function(self, func: Activation):
        """
        Sets atctivation that will be used for output neurons
        """

        if self._isBuilt:
            raise Exception(f"Cannot set activation function for an already build network.")
        
        if self.output_act_fn is not None:
            raise Exception(f"Output activation function was already set.")

        self.output_act_fn = func

    def set_loss_function(self, function: Loss):
        """
        Sets a function that will be used to computer network error and gradient
        """
        self.loss_fn = function

    def _set_gnn_parameters(self):
        if self.loss_fn is None:
            raise Exception("Cannot build, model is missing loss function.")
        self.gnn.loss_fn = self.loss_fn

        if self.hidden_act_fn is None:
            raise Exception("Cannot build, model doesn't have a hidden activation function.")
        if self.output_act_fn is None:
            raise Exception("Cannot build, model doesn't have a output activation function.")

        self.gnn.hidden_act_fn = self.hidden_act_fn
        self.gnn.output_act_fn = self.output_act_fn

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

    def _fully_connect(self):
        """
        connected all inputs to all output neurons
        like in regular dense layer
        """
        for o in range(self.gnn.N_inputs, self.gnn.N_inputs+self.gnn.N_outputs):
            for i in range(self.gnn.N_inputs):
                weight_index = self.gnn.add_connection(i, o)
                self.gnn.weights[o, weight_index] = np.random.normal(size=(1))

    def _add_neuron_randomly(self, new_order_chance: float = 0.1, memory_chance: float = 0):
        # decides if new order will be created for new neuron
        if np.random.random((1)) < new_order_chance:
            
            # removes -1 (input layer) and adds 0
            posible_order_values = list(set(self.gnn.order_values + [0]) - {-1})
            
            # choses random value except the last element
            random_index = np.random.randint(0, len(posible_order_values)-1)

            # get the value between the one picked and next one
            order_value = (posible_order_values[random_index] + posible_order_values[random_index+1])/2

        else:

            # removes -1 (input layer), 1 (output layer) and adds 0
            posible_order_values = list(set(self.gnn.order_values + [0]) - {-1, 1})

            # chooses one value at random
            order_value = np.random.choice(posible_order_values)

        # decides if neuron will be of memory type (will be using old activations for new calculations)
        if np.random.random((1)) < memory_chance:
            # get indicies of all neurons with higher or same order value
            posibble_from_neurons = np.where(self.gnn.order >= order_value)[0]

            # get indicies of all neurons with lower or same order value excluding -1s (input neurons)
            posibble_to_neurons = np.where((self.gnn.order <= order_value) & (self.gnn.order != -1))[0]

            # chooses 2 neurons that the new one will connect
            from_neuron = np.random.choice(posibble_from_neurons)
            to_neuron = np.random.choice(posibble_to_neurons)

            # add the new neuron to gnn
            self.gnn.add_neuron(from_neuron, to_neuron, order_value)

        else:
            # get indicies of all neurons with lower order value
            posibble_from_neurons = np.where(self.gnn.order < order_value)[0]

            # get indicies of all neurons with higher order value
            posibble_to_neurons = np.where(self.gnn.order > order_value)[0]

            # chooses 2 neurons that the new one will connect
            from_neuron = np.random.choice(posibble_from_neurons)
            to_neuron = np.random.choice(posibble_to_neurons)

            # add the new neuron to gnn
            self.gnn.add_neuron(from_neuron, to_neuron, order_value)

    def _add_connection_randomly(self, memory_chance: float = 0):
        pass

    def build(self):
        self._set_gnn_parameters()
        self._fully_connect()
        self._isBuilt = True

    def train(self, trainX, trainY, target_loss: float = 0.01):
        """
        Trains the network on dataset
        """


