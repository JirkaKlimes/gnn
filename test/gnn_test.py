import unittest

import numpy as np
from gnn import GNN, ReLU, Linear, CycleConnection, InputConnection


class GNNTest(unittest.TestCase):
    def test_creating_gnn(self):
        gnn = GNN(4, [ReLU, Linear])

    def test_adding_connections(self):
        gnn = GNN(4, [ReLU, Linear])
        gnn.add_connection(0, 4, 0.5)
        gnn.add_connection(1, 4, -0.5)
        gnn.add_connection(2, 4, 0.5)
        gnn.add_connection(3, 4, -0.5)
        gnn.add_connection(0, 5, 0.5)
        gnn.add_connection(1, 5, -0.5)
        gnn.add_connection(2, 5, 0.5)
        gnn.add_connection(3, 5, -0.5)
        assert (gnn.neuron_functions == np.uint8([1, 0])).all()
        assert (gnn.neuron_biases == np.float32([0, 0])).all()
        assert (gnn.multiplicative == np.array([0, 0], bool)).all()
        assert (gnn.layer_sizes == np.uint32([2])).all()
        assert (gnn.layer_pointers == np.uint32([0])).all()
        assert (gnn.neuron_indices == np.uint32([0, 1])).all()
        assert (gnn.neuron_layers == np.uint32([0, 0])).all()
        assert (gnn.conn_pointers == np.uint32([0, 4])).all()
        assert (gnn.conn_counts == np.uint32([4, 4])).all()
        assert (gnn.conn_weights == np.float32(
            [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5])).all()
        assert (gnn.conn_indices == np.uint32([0, 1, 2, 3, 0, 1, 2, 3])).all()
        assert (gnn.conn_recurrent == np.array(
            [0, 0, 0, 0, 0, 0, 0, 0], bool)).all()

    def test_adding_neurons(self):
        gnn = GNN(4, [ReLU, Linear])
        gnn.add_neuron(0, 0.2, 4, -0.3, ReLU, 1, True)
        gnn.add_neuron(0, 0.7, 4, 1.1, ReLU, 1, False)

        assert (gnn.neuron_functions == np.uint8([1, 0, 1, 1])).all()
        assert (gnn.neuron_biases == np.float32([0, 0, 1, 1])).all()
        assert (gnn.multiplicative == np.array([0, 0, 1, 0], bool)).all()
        assert (gnn.layer_sizes == np.uint32([2, 2])).all()
        assert (gnn.layer_pointers == np.uint32([0, 2])).all()
        assert (gnn.neuron_indices == np.uint32([2, 3, 0, 1])).all()
        assert (gnn.neuron_layers == np.uint32([1, 1, 0, 0])).all()
        assert (gnn.conn_pointers == np.uint32([0, 2, 2, 3])).all()
        assert (gnn.conn_counts == np.uint32([2, 0, 1, 1])).all()
        assert (gnn.conn_weights == np.float32([-0.3, 1.1, 0.2, 0.7])).all()
        assert (gnn.conn_indices == np.uint32([6, 7, 0, 0])).all()
        assert (gnn.conn_recurrent == np.array(
            [0, 0, 0, 0], bool)).all()

    def test_input_connections(self):
        gnn = GNN(4, [ReLU, Linear])
        try:
            gnn.add_neuron(0, 0, 0, 0, ReLU, 1, True)
            raise Exception()
        except InputConnection:
            pass

    def test_cycle_connection(self):
        gnn = GNN(4, [ReLU, Linear])
        gnn.add_neuron(0, 0, 4, 0, ReLU, 1, True)
        gnn.add_neuron(0, 0, 4, 0, ReLU, 1, True)
        try:
            gnn.add_connection(6, 7, 0)
            gnn.add_connection(7, 6, 0)
            raise Exception()
        except CycleConnection:
            pass


if __name__ == "__main__":
    unittest.main()
