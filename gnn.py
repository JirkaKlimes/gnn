import numpy as np
from pyparsing import IndentedBlock


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
        rounded_graph = np.round(self.digraph, 2)
        activation_len = max(len(str(np.max(np.abs((rounded_graph[:, 0]))))), len('activation'))
        state_len = max(len(str(np.max(np.abs((rounded_graph[:, 1]))))), len('state'))
        order_len = max(len(str(np.max(np.abs((rounded_graph[:, 2]))))), len('order'))
        N_inputs = max((rounded_graph.shape[1] - 3)*3-2, len('inputs'))
        table = f" {'activation'.center(activation_len, ' ')} | {'state'.center(state_len, ' ')} | {'order'.center(order_len, ' ')} | {'inputs'.center(N_inputs, ' ')} \n"
        table+= f"{'-'*(activation_len+2)}|{'-'*(state_len+2)}|{'-'*(order_len+2)}|{'-'*(N_inputs+2)}\n"
        for row in rounded_graph:
            inputs = f"{str([i for i in row[3:].astype(int)])[1:-1]}"
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

    def add_connection(self, fromN: int, toN: int):
        if fromN > self.digraph.shape[0]:
            raise Exception(f"Number of neurons is {self.digraph.shape[0]}, connection from neuron {fromN} is not possible.")
        if toN < self.N_inputs:
            raise Exception(f"You can't add connection to input neuron. Number of inputs is {self.N_inputs}, connection to neuron {toN} is not possible.")
        if toN > self.digraph.shape[0]:
            raise Exception(f"Number of neurons is {self.digraph.shape[0]}, connection to neuron {toN} is not possible.")

        """
        checks if there is space for new input
        if not add collumn of -1s to entire digraph
        """

        if self.digraph[toN][-1] != -1:
            new_inputs = np.zeros((self.digraph.shape[0], 1)) - 1
            self.digraph = np.hstack([self.digraph, new_inputs])

        self.digraph[toN][-1] = fromN


    def add_neuron(self, fromN: int, toN: int, order: float):
        pass

if __name__ == "__main__":
    gnn = Gnn(4, 2)
    gnn._create_digraph()
    
    print(gnn.table)
    gnn.add_connection(2, 4)
    print(gnn.table)
    gnn.add_connection(2, 4)
    print(gnn.table)
