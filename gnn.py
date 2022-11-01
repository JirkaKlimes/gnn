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

        self._create_digraph()
        self._create_weights()

    def _create_digraph(self):
        """
        Represents connection of neurons
        
        example of directed graph

        activation | state | order |   inputs  
        -----------|-------|-------|---|---|---
                 1 |     1 |    -1 | -1| -1| -1
                 1 |     1 |    -1 | -1| -1| -1
                 0 |     0 |    -1 | -1| -1| -1
              0.35 |    12 |     1 |  0|  1|  2
              0.78 |     2 |     1 |  0|  1|  2

        activation: float
            state passed through activation function
        state: float
            weighted sum of all inputs
        order: float
            indicated in which "pseudo" layer is the neuron located
            neurons with order -1 will never be calculated (input neurons)
            neurons with order 0 will be calculated first
            neurons with order 1 will be calculated last (output neurons)
        inputs: int
            indicies of connected neurons
            input -1 is not connected (inputs of input neurons should always be -1)
            input 0, 1, 2, means that state of neuron will be calculated
                based on activation of neurons with indices 0, 1, 2
        """

        digraph = np.zeros((self.N_inputs + self.N_outputs, 4))
        digraph[:self.N_inputs, 2] = -1 # order -1 for inputs
        digraph[-self.N_outputs:, 2] = 1 # order 1 for outputs
        digraph[:, 3] = -1 # order 1 for outputs
        self.digraph = digraph

        """
        Array of unique order values for faster computation
        """
        self.order_values = np.arange(2)

    @property
    def table(self):
        """
        string representation of directed graph
        """

        rounded_graph = np.round(self.digraph, 2)
        activation_len = max(len(str(np.max(np.abs((rounded_graph[:, 0]))))), len('activation'))
        state_len = max(len(str(np.max(np.abs((rounded_graph[:, 1]))))), len('state'))
        order_len = max(len(str(np.max(np.abs((rounded_graph[:, 2]))))), len('order'))
        N_inputs = max((rounded_graph.shape[1] - 3)*3-2, len('inputs'))
        inputsSize = len(str(int(np.max(rounded_graph[:, 3:].flatten()))))
        table = f" {'activation'.center(activation_len, ' ')} | {'state'.center(state_len, ' ')} | {'order'.center(order_len, ' ')} | {'inputs'.center(N_inputs, ' ')} \n"
        table+= f"{'-'*(activation_len+2)}|{'-'*(state_len+2)}|{'-'*(order_len+2)}|{'-'*(N_inputs+2)}\n"
        for row in rounded_graph:
            
            inputs = list(filter(lambda i: '-1' not in i, [str(int(i)).rjust(inputsSize+1) for i in row[3:]]))
            inputs = " ".join(inputs)
            
            table += f" {str(row[0]).rjust(activation_len, ' ')} | {str(row[1]).rjust(state_len, ' ')} | {str(row[2]).rjust(order_len, ' ')} | {inputs.ljust(N_inputs, ' ')}\n"

        return table

    def _create_weights(self):
        """
        first weight is the bias term
        we create the digraph without any connections
            so we only need biases for output neurons
        """
        weights = np.zeros((self.N_outputs, 1))
        self.weights = weights

    def _expand_inputs(self, n: int = 1):
        """
        Adds collumn of -1s to digraph 
        """
        new_inputs = np.zeros((self.digraph.shape[0], n)) - 1
        self.digraph = np.hstack([self.digraph, new_inputs])

    def _new_input_index(self, neuron_idx: int):
        """
        returns index of first -1 in neuron inputs
        if there isn't one inputs get expanded
        """
        negative_ones = np.where(self.digraph[neuron_idx, 3:] == -1)[0]
        if negative_ones.shape[0]:
            # +3 because we sliced only neuron inputs
            return negative_ones[0] + 3
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

        if fromN in self.digraph[toN, 3:]:
            raise Exception(f"Connection already exists from {fromN} to {toN}.")

        input_index = self._new_input_index(toN)
        self.digraph[toN][input_index] = fromN

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
        if (order <= 0) or (order > 1):
            raise Exception(f"Order for hidden neuron must be in interval (0, 1>, it was {order}.")


        new_neuron = np.zeros((1, self.digraph.shape[1]))
        new_neuron[0, 2] = order
        new_neuron[0, 3:] = -1
        self.digraph = np.vstack([self.digraph, new_neuron])
        new_neuron_index = self.digraph.shape[0] - 1

        self.add_connection(fromN, new_neuron_index)
        self.add_connection(new_neuron_index, toN)

    def fully_connect(self):
        for i in range(self.N_inputs):
            for o in range(self.N_outputs):
                self.add_connection(i, o+self.N_inputs)

if __name__ == "__main__":
    gnn = Gnn(2, 2)
    gnn._create_digraph()

    gnn.fully_connect()
    print(gnn.table)