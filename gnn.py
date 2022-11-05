import numpy as np

class Gnn:
    def __init__(self, N_inputs: int, N_outputs: int):
        """
        Growing Neural Network
        
        Parameters
        ----------
        N_inputs: int
            numer of inputs to the neural network
        N_outputs: int
            number of outputs of the neural network
        """

        self.N_inputs = N_inputs
        self.N_outputs = N_outputs

        self._initialize_network()

    def _initialize_network(self):
        """
        Initializes network parameters
        """

        # weighted sum of inputs (initialized as 0s)
        # self.z = np.zeros((self.N_inputs+self.N_outputs, 1))

        # z passed through activation function (initialized as 0s)
        # self.activations = np.zeros((self.N_inputs+self.N_outputs, 1))

        # indicates order of network computation
        self.order = np.zeros((self.N_inputs+self.N_outputs))

        # neurons with order -1 will never be calculated (input neurons)
        self.order[:self.N_inputs] = -1

        # neurons with order 0 will be calculated first
        pass

        # neurons with order 1 will be calculated last (output neurons)
        self.order[self.N_inputs:] = 1

        # indicies of connected neurons
        # input -1 is not connected (inputs of input neurons should always be -1)
        self.digraph = np.zeros((self.N_inputs+self.N_outputs, 1), dtype=np.int32) - 1

        # biases for the neurons
        self.biases = np.zeros((self.N_inputs+self.N_outputs, 1))

        # weights for the neurons
        self.weights = np.zeros((self.N_inputs+self.N_outputs, 1))

        # stores activation functions
        self.hidden_act_fn = None
        self.output_act_fn = None

        # stores loss function
        self.loss_fn = None

    @property
    def order_values(self):
        return sorted(set(self.order))

    def _expand_inputs(self, n: int = 1):

        # dds row of -1s to digraph
        new_inputs = np.zeros((self.digraph.shape[0], n)) - 1
        self.digraph = np.hstack([self.digraph, new_inputs])
        
        # dds row of 0s to weights
        new_weights = np.zeros((self.weights.shape[0], n))
        self.weights = np.hstack([self.weights, new_weights])

    def _new_input_index(self, neuron_idx: int):
        """
        returns index of first "-1" in neuron inputs
        if there isn't one inputs get expanded
        """
        negative_ones = np.where(self.digraph[neuron_idx] == -1)[0]
        if negative_ones.shape[0]:
            return negative_ones[0]
        self._expand_inputs()
        return self.digraph.shape[1] - 1

    def add_connection(self, fromN: int, toN: int):
        """
        adds connection between two neurons to the digraph
        """
        if fromN > self.digraph.shape[0]:
            raise Exception(f"Number of neurons is {self.digraph.shape[0]}, connection from neuron {fromN} is not possible.")
        if toN < self.N_inputs:
            raise Exception(f"You can't add connection to input neuron. Number of inputs is {self.N_inputs}, connection to neuron {toN} is not possible.")
        if toN > self.digraph.shape[0]:
            raise Exception(f"Number of neurons is {self.digraph.shape[0]}, connection to neuron {toN} is not possible.")

        if fromN in self.digraph[toN]:
            raise Exception(f"Connection already exists from {fromN} to {toN}.")

        input_index = self._new_input_index(toN)
        self.digraph[toN][input_index] = fromN

        return input_index

    def add_neuron(self, fromN: int, toN: int, order: float):
        """
        adds new neuron to the digraph
        """
        if fromN > self.digraph.shape[0]:
            raise Exception(f"Number of neurons is {self.digraph.shape[0]}, connection from neuron {fromN} is not possible.")
        if toN < self.N_inputs:
            raise Exception(f"You can't add connection to input neuron. Number of inputs is {self.N_inputs}, connection to neuron {toN} is not possible.")
        if toN > self.digraph.shape[0]:
            raise Exception(f"Number of neurons is {self.digraph.shape[0]}, connection to neuron {toN} is not possible.")

        if (order < 0) or (order >= 1):
            raise Exception(f"Order {order} is not in interval <0, 1)")

        self.order = np.append(self.order, order)
        new_neuron_inputs = np.zeros((1, self.digraph.shape[1])) - 1
        self.digraph = np.vstack([self.digraph, new_neuron_inputs])
        self.biases = np.append(self.biases, 0)
        new_neuron_weights = np.zeros((1, self.weights.shape[1]))
        self.weights = np.vstack([self.weights, new_neuron_weights])

        new_neuron_index = self.digraph.shape[0] - 1

        self.add_connection(fromN, new_neuron_index)
        self.add_connection(new_neuron_index, toN)

    def remove_neuron(self, index):
        if index < self.N_inputs:
            raise Exception("Can't remove input neuron.")
        if index < self.N_inputs+self.N_outputs:
            raise Exception("Can't remove output neuron.")

        raise NotImplementedError

    def push(self, x):
        """
        Propagates single inputs through the network
        """

        # store neurons activations and z
        self.activations = np.zeros((self.order.shape[0]))
        self.z = np.zeros((self.order.shape[0]))

        # sets activation for input neurons
        self.activations[:self.N_inputs] = x

        # loops over all order values
        for order in self.order_values[1:]:

            # loops over all neurons with current order value
            curent_neuron_indicies = np.where(self.order == order)[0]
            for neuron_index in curent_neuron_indicies:

                # gets indicies of all inputs to current neuron and their values
                input_indicies = self.digraph[neuron_index, np.where(self.digraph[neuron_index] != -1)[0]].astype(int)
                input_values = self.activations[input_indicies]

                # weights assigned to theese inputs
                weights = self.weights[neuron_index, :input_values.shape[0]]

                # calculates weighted sum of inputs and adds the bias term
                z = input_values @ weights + self.biases[neuron_index]

                # passes z throung activation function
                if order == 1:
                    activation = self.output_act_fn(z)
                else:
                    activation = self.hidden_act_fn(z)

                # updates values in digraph
                self.z[neuron_index] = z
                self.activations[neuron_index] = activation

        return self.activations[self.N_inputs:self.N_inputs+self.N_outputs]

    def backprop(self, x, y):
        """
        Find gradient of weights and biases for single input
        """

        # forward pass
        self.push(x)

        # list to store weights and biases gradients
        self.biases_grad = np.zeros_like(self.biases)
        self.weights_grad = np.zeros_like(self.weights)

        # loops over order values and sorts them backwards + removes -1
        for order in self.order_values[::-1][:-1]:
            
            # loops over all neurons with current order value
            curent_neuron_indicies = np.where(self.order == order)[0]
            for neuron_index in curent_neuron_indicies:

                # if neuron is output neuron
                if order == 1:

                    # gets gradient of the loss function
                    output_index = neuron_index-self.N_inputs
                    loss_grad = self.loss_fn.grad(y[output_index], self.activations[neuron_index])

                    # multiplies loss gradient by gradient of activation function
                    b_grad = self.output_act_fn.grad(self.z[neuron_index]) * loss_grad

                    # stores bias gradient
                    self.biases_grad[neuron_index] = b_grad

                    # loops over all inputs to current neuron
                    for input_index, input_neuron_index in enumerate(self.digraph[neuron_index]):
                        if input_neuron_index == -1:
                            continue
                        input_neuron_index = int(input_neuron_index)
                        
                        # multiplies activation of input neuron and bias gradient to get weight gradient
                        activation = self.activations[input_neuron_index]
                        w_grad = activation * b_grad

                        # stores weight gradient
                        self.weights_grad[neuron_index, input_index] = w_grad

                else:
                    delta_sum = 0

                    # finds all neurons that use output of current neuron
                    for next_neuron_index, next_neuron in enumerate(self.digraph):
                        if neuron_index in next_neuron:

                            # gets weight of the connection
                            weight_index = np.where(next_neuron == neuron_index)[0][0]
                            weight = self.weights[next_neuron_index, weight_index]

                            # multiples weight by the gradient and adds to sum
                            delta = weight * self.biases_grad[neuron_index]
                            delta_sum += delta

                    # multiplies the sum of gradients by gradient of the activation function
                    b_grad = delta_sum * self.hidden_act_fn.grad(self.z[neuron_index])

                    # stores bias gradient
                    self.biases_grad[neuron_index] = b_grad

                    # loops over all inputs to current neuron
                    for input_index, input_neuron_index in enumerate(self.digraph[neuron_index]):
                        if input_neuron_index == -1:
                            continue
                        input_neuron_index = int(input_neuron_index)

                        # multiplies activation of input neuron and bias gradient to get weight gradient
                        activation = self.activations[input_neuron_index]
                        w_grad = activation * b_grad

                        # stores weight gradient
                        self.weights_grad[neuron_index, input_index] = w_grad

        return self.biases_grad, self.weights_grad
