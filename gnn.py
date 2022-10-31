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

        digraph = np.zeros((self.N_inputs + self.N_outputs, 3))
        digraph[:self.N_inputs, 2] = -1 # order -1 for inputs
        digraph[-self.N_outputs:, 2] = 1 # order 1 for outputs
        self.digraph = digraph

        """
        Array of unique order values
        """
        self.order_values = np.arange(2)

    def _create_weights(self):
        """
        first weight is the bias term
        we create the digraph without any connections
            so we only need biases for output neurons
        """
        weights = np.zeros((self.N_outputs, 1))
        self.weights = weights

if __name__ == "__main__":
    gnn = Gnn(4, 2)
    gnn._create_digraph()

    print(gnn.digraph)
    print(gnn.weights)