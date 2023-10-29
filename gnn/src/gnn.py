import numpy as np
from typing import List
from gnn.src.activations import Activation, Linear


class GNN:
    def __init__(
        self,
        n_inputs: int,
        outputs: List[Activation],
        value_type: np.dtype = np.float32,
        index_type: np.dtype = np.uint32
    ):
        self.n_inputs = n_inputs
        self.n_outputs = len(outputs)
        self.value_type = value_type
        self.index_type = index_type

        self._max_index = np.iinfo(self.index_type).max

        self.gnn_funcs = {Linear: 0}
        list(map(self.add_activation, outputs))

        self.neuron_order = np.full(
            self.n_outputs, self._max_index, dtype=self.index_type
        )

        self.neuron_functions = np.uint8([self.gnn_funcs[f] for f in outputs])
        self.neuron_biases = np.zeros(self.n_outputs, self.value_type)
        self.multiplicative = np.zeros(self.n_outputs, bool)

        self.weights = np.empty(0, dtype=self.value_type)
        self.weight_pointers = np.empty(0, dtype=self.index_type)
        self.weight_counts = np.empty(0, dtype=self.index_type)

    def add_activation(self, func: Activation):
        if func not in self.gnn_funcs:
            self.gnn_funcs[func] = len(self.gnn_funcs)
