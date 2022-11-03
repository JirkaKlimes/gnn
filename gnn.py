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

        # types of activation functions neurons use
        self.activation_functions_ids = np.zeros((self.N_inputs+self.N_outputs), np.uint8)

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

        # stores all used activation function
        self.activations = []
        self.loss_fn = None

    def _expand_inputs(self, n: int = 1):
        """
        Adds:
            - collumn of -1s to digraph
            - collumn of 0s to weights
        """
        new_inputs = np.zeros((self.digraph.shape[0], n)) - 1
        self.digraph = np.hstack([self.digraph, new_inputs])
        
        new_weights = np.zeros((self.weights.shape[0], n))
        self.weights = np.hstack([self.weights, new_weights])

    def _new_input_index(self, neuron_idx: int):
        """
        returns index of first -1 in neuron inputs
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

    def add_neuron(self, fromN: int, toN: int, order: float, activation_function: int):
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

        self.activation_functions_ids = np.append(self.activation_functions_ids, activation_function)
        self.order = np.append(self.order, order)
        new_neuron_inputs = np.zeros((1, self.digraph.shape[1])) - 1
        self.digraph = np.vstack([self.digraph, new_neuron_inputs])
        self.biases = np.append(self.biases, 0)
        new_neuron_weights = np.zeros((1, self.weights.shape[1]))
        self.weights = np.vstack([self.weights, new_neuron_weights])

        new_neuron_index = self.digraph.shape[0] - 1

        self.add_connection(fromN, new_neuron_index)
        self.add_connection(new_neuron_index, toN)

    def add_order_value(self, value: float):
        if (value <= 0) or (value > 1):
            raise Exception(f"Order for hidden neuron must be in interval (0, 1>, it was {value}.")

        if value in self.order_values:
            return

        self.order_values = sorted(self.order_values + [value])

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

        # get unique order values and sorts them + removes -1
        order_values = sorted(set(self.order))[1:]

        # loops over all order values
        for order in order_values:

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
                activation = self.activation_functions[self.activation_functions_ids[neuron_index]](z)

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

        # get unique order values and sorts them backwards + removes -1
        order_values = sorted(set(self.order), reverse=True)[:-1]

        # loops over all order values
        for order in order_values:
            
            # loops over all neurons with current order value
            curent_neuron_indicies = np.where(self.order == order)[0]
            for neuron_index in curent_neuron_indicies:
                
                # if neuron is output neuron
                if order == 1:
                    
                    # gets gradient of the loss function
                    output_index = neuron_index-self.N_inputs
                    loss_grad = self.loss_fn.grad(y[output_index], self.activations[neuron_index])
                    
                    # multiplies loss gradient by gradient of activation function
                    b_grad = self.activation_functions[self.activation_functions_ids[neuron_index]].grad(self.z[neuron_index]) * loss_grad

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
                    b_grad = delta_sum * self.activation_functions[self.activation_functions_ids[neuron_index]].grad(self.z[neuron_index])
                    
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

if __name__ == "__main__":
    pass




































# @property
# def table(self):
#     """
#     string representation of directed graph
#     """

#     rounded_activations = np.round(self.activations[0], 2)
#     z = np.round(self.z[0], 2)

#     activation_len = max(len(str(np.max(np.abs(rounded_activations)))), len('activation'))
#     z_len = max(len(str(np.max(np.abs(z)))), len('z')) + 1
#     order_len = max(len(str(np.max(np.abs((self.order))))), len('order'))
#     N_inputs = max((self.digraph.shape[1])*3-2, len('inputs'))
#     inputsSize = len(str(int(np.max(self.digraph.flatten()))))
#     table = f" {'activation'.center(activation_len, ' ')} | {'z'.center(z_len, ' ')} | {'order'.center(order_len, ' ')} | {'inputs'.center(N_inputs, ' ')} \n"
#     table+= f"{'-'*(activation_len+2)}|{'-'*(z_len+2)}|{'-'*(order_len+2)}|{'-'*(N_inputs+2)}\n"
#     for a, z, order, inputs in zip(rounded_activations, np.round(self.z, 2), self.order, self.digraph):
#         inputs = list(filter(lambda i: '-1' not in i, [str(int(i)).rjust(inputsSize+1) for i in inputs]))
#         inputs = " ".join(inputs)
        
#         table += f" {str(a).rjust(activation_len, ' ')} | {str(z).rjust(z_len, ' ')} | {str(order).rjust(order_len, ' ')} | {inputs.ljust(N_inputs, ' ')}\n"

#     return table
