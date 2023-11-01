import unittest

import numpy as np
import random
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
            gnn.add_neuron(0, 0, 1, 0, ReLU, 1, True)
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

    def test_adding_lot_of_neurons(self):

        gnn = GNN(4, [ReLU, Linear])
        gnn.add_neuron(0, 1, 4, 1, ReLU, 1, False)
        gnn.add_neuron(0, 1, 4, 1, ReLU, 1, False)
        gnn.add_neuron(6, 1, 7, 1, ReLU, 1, False)

        gnn = GNN(4, [ReLU, Linear])
        gnn.add_neuron(0, 1, 5, 1, ReLU, 1, False)
        gnn.add_neuron(6, 1, 5, 1, ReLU, 1, False)

        gnn = GNN(4, [ReLU, Linear])
        gnn.add_neuron(1, 1, 4, 1, ReLU, 1, False)
        gnn.add_neuron(0, 1, 5, 1, ReLU, 1, False)
        gnn.add_neuron(0, 1, 5, 1, ReLU, 1, False)
        gnn.add_neuron(6, 1, 8, 1, ReLU, 1, False)

        gnn = GNN(4, [ReLU, Linear])
        gnn.add_neuron(1, 1, 4, 1, ReLU, 1, False)
        gnn.add_neuron(2, 1, 5, 1, ReLU, 1, False)
        gnn.add_neuron(3, 1, 5, 1, ReLU, 1, False)
        gnn.add_neuron(6, 1, 5, 1, ReLU, 1, False)
        gnn.add_neuron(7, 1, 5, 1, ReLU, 1, False)
        gnn.add_neuron(7, 1, 8, 1, ReLU, 1, False)
        gnn.add_neuron(1, 1, 9, 1, ReLU, 1, False)
        gnn.add_neuron(7, 1, 11, 1, ReLU, 1, False)

        for _ in range(20):
            gnn = GNN(4, [ReLU, Linear])
            for _ in range(100):
                from_indices = [
                    *range(gnn.n_in), *(gnn.neuron_indices[:-int(gnn.layer_sizes[-1])] + gnn.n_in)]
                to_indices = [*(gnn.neuron_indices + gnn.n_in)]

                fi = random.choice(from_indices)
                ti = random.choice(to_indices)
                try:
                    gnn.add_neuron(fi, 1, ti, 1, ReLU, 1, False)
                except CycleConnection:
                    pass

    def test_forward_propagation_cpu(self):
        gnn = GNN(4, [ReLU, Linear])
        gnn.add_neuron(0, 0.2, 4, -0.3, ReLU, 1.2, True)
        gnn.add_neuron(0, 0.7, 4, 1.1, ReLU, 0.5, False)
        gnn.add_connection(7, 6, 0.5)
        assert (gnn.push([1, 2, 3, 4]) == np.float32([0.58559996, 0])).all()


if __name__ == "__main__":
    unittest.main()
