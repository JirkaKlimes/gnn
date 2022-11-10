import numpy as np
from numba import cuda
import numba
from collections import Counter
import os
# os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

np.random.seed(1)

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
        self.z = np.zeros((self.N_inputs+self.N_outputs, 1))

        # z passed through activation function (initialized as 0s)
        self.activations = np.zeros((self.N_inputs+self.N_outputs, 1))
        self.many_activations = None

        # indicates order of network computation
        self.order = np.zeros((self.N_inputs+self.N_outputs), np.float32)

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
        self.biases = np.zeros((self.N_inputs+self.N_outputs), np.float64)

        # weights for the neurons
        self.weights = np.zeros((self.N_inputs+self.N_outputs, 1), np.float64)

        # stores activation functions
        self.hidden_act_fn = None
        self.output_act_fn = None

        # stores loss function
        self.loss_fn = None

        self.gpu_data = {
            "digraph": None,
            "thread_order": None,
            "weights": None,
            "biases": None
                        }

    @property
    def order_values(self):
        return sorted(set(self.order))[1:]

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

    # memory not working for now, need to change this
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
        for order in self.order_values:

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

        # loops over order values and sorts them backwards
        for order in self.order_values[::-1]:
            
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
                            delta = weight * self.biases_grad[next_neuron_index]
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
 
    def create_push_kernel(self):
        """
        Creates GPU kernel for computing multiple inputs in parallel
        (useful for validating network)
        """
        # activation functions callable from kernel
        hidden_act_fn = cuda.jit(self.hidden_act_fn.cuda_fn(), device=True)
        output_act_fn = cuda.jit(self.output_act_fn.cuda_fn(), device=True)

        @cuda.jit
        def kernel(digraph, thread_order, weights, biases, order, many_activations):
            # gets activations of current block
            activations = many_activations[cuda.blockIdx.x]

            # loops through neurons with same order value (forward pass)
            for neuron_indicies in thread_order:

                # finds neurons index for current thread
                neuron_index = int(neuron_indicies[cuda.threadIdx.x])

                # if neuron exists
                if neuron_index != -1:

                    # sets z to bias
                    z = biases[neuron_index]

                    # loops through all inputs of current neuron
                    for i in range(digraph.shape[1]):

                        # gets index of current input neuron
                        input_index = digraph[neuron_index, i]
                        input_index = int(input_index)

                        # stops if inputs index is "-1" (last input was calculated)
                        if input_index == -1: break

                        # weight of current neuron assigned to input "i"
                        weight = weights[neuron_index, i]

                        # adds weighted activation to z
                        z += activations[input_index] * weight

                    # passes z through activation function
                    activations[neuron_index] = output_act_fn(z) if order[neuron_index] == 1 else hidden_act_fn(z)

                # waits for all threads to finish current order value
                cuda.syncthreads()

        self._push_kernel = kernel

    def push_GPU(self, x: np.ndarray):
        """
        Propagetes multiple sets of inputs in parallel using GPU
        """
        # number of parallel inputs (CUDA grid size)
        batch_size = x.shape[0]

        # initializes activations to 0
        if self.many_activations is None:
            self.many_activations = np.zeros((batch_size, self.order.shape[0]))

        # sets input activations
        self.many_activations[:, :self.N_inputs] = x

        # moves activations to gpu
        gpu_many_actiavtions = cuda.to_device(self.many_activations)

        # block size is maximum number of neurons with same order
        block_size = self.gpu_data["thread_order"].shape[1]

        # computes activations using GPU
        self._push_kernel[batch_size, block_size](self.gpu_data["digraph"],
                                                 self.gpu_data["thread_order"],
                                                 self.gpu_data["weights"],
                                                 self.gpu_data["biases"],
                                                 self.gpu_data["order"],
                                                 gpu_many_actiavtions)
        
        # copies activations on GPU back to host
        self.many_activations = gpu_many_actiavtions.copy_to_host()

        # returns activations of output neurons
        y = self.many_activations[:, -self.N_outputs:]
        return y

    def load_push_data_to_GPU(self):
        # maximum number of neurons with same order value
        max_count = Counter(self.order).most_common(1)[0][1]

        # each row contains all neurons with same order (can be computed at the same time)
        thread_order = np.zeros((len(self.order_values), max_count)) - 1
        for i, value in enumerate(self.order_values):
            indicies = np.where(self.order == value)[0]
            thread_order[i, :indicies.shape[0]] = indicies

        # copies the data to GPU memory
        self.gpu_data['digraph'] = cuda.to_device(self.digraph)
        self.gpu_data['thread_order'] = cuda.to_device(thread_order)
        self.gpu_data['weights'] = cuda.to_device(self.weights)
        self.gpu_data['biases'] = cuda.to_device(self.biases)
        self.gpu_data['order'] = cuda.to_device(self.order)

    def create_backprop_kernel(self):
        """
        Creates CUDA kernel for computing multiple gradients in parallel
        """
        # activation functions and their gradients callable from kernel
        hidden_act_fn = cuda.jit(self.hidden_act_fn.cuda_fn(), device=True)
        output_act_fn = cuda.jit(self.output_act_fn.cuda_fn(), device=True)
        hidden_act_fn_grad = cuda.jit(self.hidden_act_fn.cuda_grad(), device=True)
        output_act_fn_grad = cuda.jit(self.output_act_fn.cuda_grad(), device=True)

        # loss function callable from kernel
        loss_fn_grad = cuda.jit(self.loss_fn.cuda_grad(), device=True)

        @cuda.jit
        def kernel(digraph, transposed_digraph, thread_order, weights, biases, order, many_activations, many_z, many_biases_grad, many_weights_grad, many_y, N_inputs):
            # gets data of current block
            activations = many_activations[cuda.blockIdx.x]
            z = many_z[cuda.blockIdx.x]
            biases_grad = many_biases_grad[cuda.blockIdx.x]
            weights_grad = many_weights_grad[cuda.blockIdx.x]
            y = many_y[cuda.blockIdx.x]

            # loops through neurons with same order value (forward pass)
            for neuron_indicies in thread_order:

                # finds neurons index for current thread
                neuron_index = int(neuron_indicies[cuda.threadIdx.x])

                # if neuron exists
                if neuron_index != -1:

                    # sets z to bias
                    z[neuron_index] = biases[neuron_index]

                    # loops through all inputs of current neuron
                    for i in range(digraph.shape[1]):

                        # gets index of current input neuron
                        input_index = digraph[neuron_index, i]
                        input_index = int(input_index)

                        # stops if inputs index is "-1" (last input was calculated)
                        if input_index == -1: break

                        # weight of current neuron assigned to input "i"
                        weight = weights[neuron_index, i]

                        # adds weighted activation to z
                        z[neuron_index] += activations[input_index] * weight

                    # passes z through activation function
                    activations[neuron_index] = output_act_fn(z[neuron_index]) if order[neuron_index] == 1 else hidden_act_fn(z[neuron_index])

                # waits for all threads to finish current order value
                cuda.syncthreads()

            # loops through neurons with same order value (backward pass)
            for neuron_indicies in thread_order[::-1]:

                # finds neurons index for current thread
                neuron_index = int(neuron_indicies[cuda.threadIdx.x])

                # neuron is output neuron
                if order[neuron_index] == 1:

                    # gets gradient of the loss function
                    output_index = neuron_index-N_inputs
                    loss_grad = loss_fn_grad(y[output_index], activations[neuron_index])

                    # multiplies loss gradient by gradient of activation function
                    b_grad = output_act_fn_grad(z[neuron_index]) * loss_grad

                    # stores bias gradient
                    biases_grad[neuron_index] = b_grad

                    # loops through all inputs of current neuron
                    for input_index in range(digraph.shape[1]):

                        # gets index of current input neuron
                        input_neuron_index = digraph[neuron_index, input_index]
                        input_neuron_index = int(input_neuron_index)

                        # multiplies activation of input neuron and bias gradient to get weight gradient
                        activation = activations[input_neuron_index]
                        w_grad = activation * b_grad

                        # stores weight gradient
                        weights_grad[neuron_index, input_index] = w_grad

                else:
                    delta_sum = 0

        self._backprop_kernel = kernel

    def backprop_GPU(self, x: np.ndarray, y: np.ndarray):
        """
        Find gradient of weights and biases for multiple inputs in parallel
        """
        # number of parallel inputs (CUDA grid size)
        batch_size = x.shape[0]

        # initializes activations and z to 0
        if self.many_activations is None:
            self.many_activations = np.zeros((batch_size, self.order.shape[0]))
            self.gpu_data["many_z"] = cuda.to_device(np.zeros((batch_size, self.order.shape[0])))

        # moves outputs to gpu
        gpu_many_y = cuda.to_device(y)

        # sets input activations
        self.many_activations[:, :self.N_inputs] = x

        # moves activations to gpu
        gpu_parallel_actiavtions = cuda.to_device(self.many_activations)

        # stores gradients for biases and weights on GPU
        many_bias_grads = cuda.to_device(np.zeros((batch_size, *self.biases.shape)))
        many_weight_grads = cuda.to_device(np.zeros((batch_size, *self.weights.shape)))

        # block size is maximum number of neurons with same order
        block_size = self.gpu_data["thread_order"].shape[1]

        # computes gradients using GPU
        self._backprop_kernel[batch_size, block_size](self.gpu_data["digraph"],
                                                      self.gpu_data["transposed_digraph"],
                                                      self.gpu_data["thread_order"],
                                                      self.gpu_data["weights"],
                                                      self.gpu_data["biases"],
                                                      self.gpu_data["order"],
                                                      gpu_parallel_actiavtions,
                                                      self.gpu_data["many_z"],
                                                      many_bias_grads,
                                                      many_weight_grads,
                                                      gpu_many_y,
                                                      self.N_inputs)
        
        # copies gradients back to host
        many_bias_grads = many_bias_grads.copy_to_host()
        many_weight_grads = many_weight_grads.copy_to_host()

        return many_bias_grads, many_weight_grads

    def _transpose_digraph(self):
        """
        Returns transpose of "self.digraph"
        """
        # finds width of new graph
        flattened = self.digraph.flatten()
        flattened = flattened[flattened != -1]
        width = Counter(flattened).most_common(1)[0][1]

        # new empty graph
        digraph = np.zeros((self.digraph.shape[0], width)) - 1

        # loops over index of every neuron
        for neuron in range(self.digraph.shape[0]):
            
            # finds all neurons that have neuron in it's inputs
            outputs = np.where(self.digraph == neuron)[0]
            
            # updates new digraph
            digraph[neuron, :outputs.shape[0]] = outputs
        
        return digraph

    def load_backprop_data_to_GPU(self):
        # loads push data to GPU
        self.load_push_data_to_GPU()

        # loads transposed digraph to GPU
        self.gpu_data["transposed_digraph"] = cuda.to_device(self._transpose_digraph())


if __name__ == "__main__":
    from activations import Relu, Identity
    from losses import MeanSquaredError

    np.random.seed(1)

    gnn = Gnn(2, 2)

    gnn.hidden_act_fn = Identity()
    gnn.output_act_fn = Identity()
    gnn.loss_fn = MeanSquaredError()
    gnn.add_connection(0, 3)
    gnn.add_connection(1, 3)
    gnn.add_connection(0, 2)
    gnn.add_connection(1, 2)
    gnn.add_neuron(0, 3, 0.25)
    gnn.add_neuron(1, 2, 0.25)
    gnn.add_neuron(4, 3, 0.5)
    gnn.add_neuron(5, 3, 0.75)


    gnn.weights += np.random.normal(size=gnn.weights.shape)

    gnn.create_backprop_kernel()

    gnn.load_backprop_data_to_GPU()

    x = np.array([[1, 2]])
    y = np.array([[3, 4]])

    np.set_printoptions(3)

    b_grad, w_grad = gnn.backprop(x[0], y[0])

    gpu_b_grad, gpu_w_grad = gnn.backprop_GPU(x, y)

    print(b_grad)
    print(w_grad)
    print()

    print(gpu_b_grad)
    print(gpu_w_grad)







