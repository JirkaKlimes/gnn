import numpy as np
from abc import ABC, abstractmethod
from math import ceil

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

        # memory neurons can't grow by default
        self.allow_memory = False

    def allow_memory(self):
        self.allow_memory = True

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

    def _add_neuron_randomly(self, new_order_probability: float = 0.1, memory_probability: float = 0):
        # decides if new order will be created for new neuron
        if np.random.random((1)) < new_order_probability:
            
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
        if np.random.random((1)) < memory_probability:

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

    # prototype method, memory can't form
    def _add_connection_randomly(self, memory_probability: float = 0):
        # decides if connection will be of memory type (will be using old activations for new calculations)
        if np.random.random((1)) < memory_probability:
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
        self.gnn.create_backprop_kernel()
        self.gnn.create_push_kernel()
        self._isBuilt = True

    def _prepare_dataset(self, dataset):
        train_x, train_y = dataset

        # gets lengths of all sequences and maximum sequence length
        self.sequence_lengths = np.array(list(map(len, train_x)))
        max_seq_len = self.sequence_lengths.max()

        # fills short sequences with zeros
        symetric_train_x = []
        symetric_train_y = []
        for seq_len, x_seq, y_seq in zip(self.sequence_lengths, train_x, train_y):
            sym_x_seq = np.zeros((max_seq_len, self.gnn.N_inputs))
            sym_y_seq = np.zeros((max_seq_len, self.gnn.N_outputs))
            sym_x_seq[:seq_len, :] = x_seq
            sym_y_seq[:seq_len, :] = y_seq
            symetric_train_x.append(sym_x_seq)
            symetric_train_y.append(sym_y_seq)

        self.train_x = np.array(symetric_train_x)
        self.train_y = np.array(symetric_train_y)

        self.dataset_length = self.train_x.shape[0]

    def _get_batch(self, size):
        # randomly picks "size" indicies
        b_size = size if size <= self.dataset_length else self.dataset_length
        data_indicies = np.arange(self.dataset_length)
        picked_indicies = np.random.choice(data_indicies, b_size, replace=False)

        # returns x, y and sequence lenghts on those indicies
        x = self.train_x[picked_indicies]
        y = self.train_y[picked_indicies]
        seq_lengths = self.sequence_lengths[picked_indicies]
        return x, y, seq_lengths

    # prototype (gradient momentum not implemented)
    def _update_batch(self):
        # gets gradients and updates weights and biases
        x_batch, y_batch, seq_lenghts = self._get_batch(self._batch_size)

        b_grad, w_grad = self.gnn.backprop_GPU(x_batch, y_batch, seq_lenghts)

        self.gnn.biases -= b_grad * self._learning_rate
        self.gnn.weights -= w_grad * self._learning_rate

    def _validate(self):
        pred_y = self.gnn.push_GPU(self.train_x, self.sequence_lengths)
        return self.loss_fn(self.train_y, pred_y)

    def _grow_network(self, grow_ratio, memory_probability, new_order_probability):
        n_neurons = self.gnn.order.shape[0] - self.gnn.N_inputs
        n_new_neurons = ceil(n_neurons * grow_ratio)
        for _ in range(n_new_neurons):
            self._add_neuron_randomly(new_order_probability, memory_probability)
        
        n_connections = np.count_nonzero(self.gnn.digraph.flatten() + 1)
        n_new_connections = ceil(n_connections * grow_ratio)
        for _ in range(n_new_connections):
            self._add_connection_randomly(memory_probability)

    def train(self, dataset, batch_size: int = None, target_loss: float = 0, epochs: int = np.Infinity,
                    lr: float = 0.01, vf: int = 10, ls: float = -0.005, lbs: float = 5, gr: float = 0.01,
                    gd: int = 10, mp: float = 0.05, nop: float = 0.2, callbacks: list = []):
        """
        Trains the network on dataset
        
        Parameters
        ----------
        dataset:
            tuple of x and y lists of sequences
        batch_size:
            size of batch backpropagating at once
        target_loss:
            training will stop when this loss is reached
        epochs:
            training will stop after this many epochs
        lr:
            learning rate
        vf:
            validation frequency
        ls:
            maximum slope in loss for growing network
        lbs:
            how many validations to look back for calculating loss slope
        gr:
            how much neurons should be added
        gd:
            how many validations to wait before new neurons can grow
        mp:
            probability that new neuron / connection will be memory
        nop:
            probability that new neuron will create new order value
        callbacks:
            list of callbacks that will be updated when validating the network
        """

        self._learning_rate = lr
        self._batch_size = batch_size

        self._prepare_dataset(dataset)
        self.gnn.create_backprop_kernel()
        self.gnn.create_push_kernel()
        self.gnn.load_GPU_data()

        [callback.create() for callback in callbacks]

        training_data = {
            "gnn": self.gnn,
            "epochs": [],
            "loss": [],
            "number_of_neurons": self.gnn.number_of_neurons,
            "new_neurons_epochs": [],
            "new_neurons_loss": []
        }

        current_loss = np.Infinity
        current_epoch = 0
        last_grow = 0
        while current_epoch < epochs or current_loss < target_loss:
            netwok_grew = False
            current_epoch += 1

            self._update_batch()

            loss = self._validate()

            if current_epoch % vf == 0:

                if len(training_data["loss"]) > lbs:
                    slope = (training_data["loss"][-1] - training_data["loss"][-lbs]) / vf * lbs

                    current_validation = current_epoch / vf
                    if slope > ls and current_validation > last_grow + gd:
                        training_data["new_neurons_epochs"].append(current_epoch)
                        training_data["new_neurons_loss"].append(loss)

                        last_grow = current_validation
                        self._grow_network(gr, mp, nop)
                        netwok_grew = True

                training_data["epochs"].append(current_epoch)
                training_data["loss"].append(loss)
                training_data["number_of_neurons"] = self.gnn.number_of_neurons

                [callback(training_data) for callback in callbacks]

            self.gnn.load_GPU_data(not netwok_grew)


if __name__ == "__main__":
    from activations import Relu, Identity
    from losses import MeanSquaredError
    from callbacks import PlotCallback, StdOutCallback

    from sequence_datasets import addToSum, test1
    from pprint import pprint

    gnn = Gnn(1, 1)

    model = UnnamedModel1(gnn)
    model.set_hidden_act_function(Relu())
    model.set_output_act_function(Identity())
    model.set_loss_function(MeanSquaredError())
    model.allow_memory()

    model.build()

    gnn.add_neuron(0, 1, 0.75)
    gnn.add_neuron(2, 1, 0.5)

    # gnn.add_neuron(0, 1, 0.9)
    # gnn.add_neuron(2, 1, 0.1)
    # gnn.add_neuron(0, 1, 0.2)

    # gnn.add_connection(0, 3)
    # gnn.add_connection(4, 3)

    # gnn.add_neuron(3, 1, 0.5)

    # gnn.add_connection(5, 4)
    # gnn.add_connection(3, 2)
    # gnn.add_connection(0, 5)

    gnn.weights += np.random.normal(size=gnn.weights.shape)


    # train_x = np.linspace(0, np.pi, 5).reshape(-1, 1)
    # train_y = np.sin(train_x)

    # dataset = (train_x, train_y)

    # dataset = addToSum()
    dataset = test1()

    model.train(dataset, 1, epochs = 10000, lr = 0.005, ls = -0.005, gr = 0.01, gd = 10, callbacks=[PlotCallback(), StdOutCallback()])

    x, _ = dataset
    seq = x[0]

    for x in seq:
        print(gnn.push(x))
