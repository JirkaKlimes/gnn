import numpy as np
from abc import ABC, abstractmethod
from activations import Activation
from losses import Loss

from gnn import Gnn

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

            # returns false if neuron can't be created
            if posibble_from_neurons.shape[0] == 0 or posibble_from_neurons.shape[0] == 0:
                return False

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

        return True

    # this is only prototype method
    def _add_connection_randomly(self, memory_chance: float = 0):
        # decides if connection will be of memory type (will be using old activations for new calculations)
        if np.random.random((1)) < memory_chance:
            pass

        else:

            # neurons that have free input
            free_inputs = np.unique(np.argwhere(self.gnn.digraph == -1)[:, 0])[self.gnn.N_inputs:]

            # neurons that have higher order value than 0
            possible_order_values = np.where(self.gnn.order > 0)[0]

            # neurons to which connection can be created
            possible_to_neurons = np.intersect1d(free_inputs, possible_order_values)

            if not possible_to_neurons.shape[0]:
                return False

            to_neuron = np.random.choice(possible_to_neurons)
            to_neuron_order = self.gnn.order[to_neuron]

            possible_from_neurons = np.where(self.gnn.order < to_neuron_order)[0]
            if not possible_from_neurons.shape[0]:
                return False

            from_neuron = np.random.choice(possible_from_neurons)

            if from_neuron in self.gnn.digraph[to_neuron]:
                return False

            self.gnn.add_connection(from_neuron, to_neuron)

        return True

    def build(self):
        self._set_gnn_parameters()
        self._fully_connect()
        self.gnn.weights += np.random.normal(size=self.gnn.weights.shape)
        self._isBuilt = True

    def get_batch(self, size):
        train_x, train_y = self.dataset
        dataset_lenght = train_x.shape[0]
        size = size if size <= dataset_lenght else dataset_lenght

        data_indicies = np.arange(dataset_lenght)
        picked_indicies = np.random.choice(data_indicies, size, replace=False)

        batch_x = train_x[picked_indicies]
        batch_y = train_y[picked_indicies]


    def train(self, dataset, batch_size: int = None, target_loss: float = 0, epochs: int = np.Infinity, validation_frequency: int = 10, learning_rate: float = 0.01, callbacks: list = []):
        """
        Trains the network on dataset
        """

        self.dataset = dataset

        [callback.create() for callback in callbacks]

        training_data = {
            "gnn": self.gnn,
            "val_iters": [],
            "loss": [],
            "number_of_neurons": self.gnn.number_of_neurons,
            "new_neurons_iters": [],
            "new_neurons_loss": []
        }

        current_loss = np.Infinity
        current_iter = 0
        while current_iter < epochs or current_loss < target_loss:
            current_iter += 1

            batch = self.get_batch(batch_size)

            loss = np.random.randint(current_iter, size=(1))[0]

            if current_iter % validation_frequency == 0:


                if np.random.random((1)) < 0.2:
                    training_data["new_neurons_iters"].append(current_iter)
                    training_data["new_neurons_loss"].append(loss)

                training_data["val_iters"].append(current_iter)
                training_data["loss"].append(loss)
                training_data["number_of_neurons"] = self.gnn.number_of_neurons

                [callback(training_data) for callback in callbacks]



if __name__ == "__main__":
    from activations import Relu, Identity
    from losses import MeanSquaredError
    from callbacks import PlotCallback

    gnn = Gnn(1, 1)

    model = UnnamedModel1(gnn)
    model.set_hidden_act_function(Relu())
    model.set_output_act_function(Identity())
    model.set_loss_function(MeanSquaredError())

    model.build()

    train_x = np.linspace(0, np.pi, 5)
    train_y = np.sin(train_x)

    dataset = (train_x.reshape(-1, 1, 1), train_y.reshape(-1, 1, 1))

    model.train(dataset, 5, callbacks=[PlotCallback()], epochs=10)