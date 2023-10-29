import unittest

import numpy as np
from gnn import GNN, ReLU, Linear


class GNNTest(unittest.TestCase):

    def test_create(self):
        GNN(4, [ReLU, Linear])

    def test_adding_connections(self):
        gnn = GNN(4, [ReLU, Linear])
        gnn.add_connection(0, 4, 0.1)
        gnn.add_connection(2, 4, 0.2)
        gnn.add_connection(1, 5, -0.4)
        gnn.add_connection(3, 5, 0.5)
        assert (gnn.neuron_order == gnn.index_dtype(
            [4294967295, 4294967295])).all()
        assert (gnn.neuron_functions == np.byte([1, 0])).all()
        assert (gnn.neuron_biases == gnn.value_dtype([0, 0])).all()
        assert (gnn.multiplicative == np.array([False, False])).all()
        assert (gnn.weight_pointers == gnn.index_dtype([0, 2])).all()
        assert (gnn.weight_counts == gnn.index_dtype([2, 2])).all()
        assert (gnn.weights == gnn.value_dtype([0.1, 0.2, -0.4, 0.5])).all()
        assert (gnn.connections == gnn.index_dtype([0, 2, 1, 3])).all()

    def test_adding_neurons(self):
        gnn = GNN(4, [ReLU, Linear])
        gnn.add_neuron(0, 5, gnn.max_index // 2, 0.3, 0.7, 0, ReLU, True)

        assert (gnn.neuron_order == gnn.index_dtype(
            [4294967295, 4294967295, 2147483647])).all()
        assert (gnn.neuron_functions == np.byte([1, 0, 1])).all()
        assert (gnn.neuron_biases == gnn.value_dtype([0, 0, 0])).all()
        assert (gnn.multiplicative == np.array([False, False, True])).all()
        assert (gnn.weight_pointers == gnn.index_dtype([0, 0, 1])).all()
        assert (gnn.weight_counts == gnn.index_dtype([0, 1, 1])).all()
        assert (gnn.weights == gnn.value_dtype([0.7, 0.3])).all()
        assert (gnn.connections == gnn.index_dtype([6, 0])).all()


if __name__ == "__main__":
    unittest.main()
