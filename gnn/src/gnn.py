import numpy as np
from typing import List
from gnn.src.activations import Activation, Linear


class SameOrderException(Exception):
    def __init__(self, from_idx: int, to_idx: int, order: int):
        message = f"Cannot connect 2 neurons ({from_idx} -> {to_idx}) with same order ({order})"
        super().__init__(message)


class GNN:
    VALUE_DTYPE = np.float32
    INDEX_DTYPE = np.uint32

    def __init__(
        self,
        n_inputs: int,
        outputs: List[Activation],
        value_dtype: np.dtype = VALUE_DTYPE,
        index_dtype: np.dtype = INDEX_DTYPE
    ):
        self.n_inputs = n_inputs
        self.n_outputs = len(outputs)
        self.value_dtype = value_dtype
        self.index_dtype = index_dtype

        self.max_index = np.iinfo(self.index_dtype).max

        self.gnn_funcs = {Linear: 0}
        list(map(self.add_activation, outputs))

        self.neuron_order = np.full(
            self.n_outputs, self.max_index, dtype=self.index_dtype
        )

        self.neuron_functions = np.uint8([self.gnn_funcs[f] for f in outputs])
        self.neuron_biases = np.zeros(self.n_outputs, self.value_dtype)
        self.multiplicative = np.zeros(self.n_outputs, bool)

        self.weight_pointers = np.zeros(self.n_outputs, dtype=self.index_dtype)
        self.weight_counts = np.zeros(self.n_outputs, dtype=self.index_dtype)
        self.weights = np.empty(0, dtype=self.value_dtype)
        self.connections = np.empty(0, dtype=self.index_dtype)

    def add_activation(self, func: Activation):
        if func not in self.gnn_funcs:
            self.gnn_funcs[func] = len(self.gnn_funcs)

    def add_connection(self, from_idx: INDEX_DTYPE, to_idx: INDEX_DTYPE, weight: VALUE_DTYPE):
        # offset of `n_inputs` since we don't store anything for input neurons
        wit = to_idx - self.n_inputs
        wif = from_idx - self.n_inputs

        if from_idx >= self.n_inputs and self.neuron_order[wit] == self.neuron_order[wif]:
            raise SameOrderException(wif, wit, self.neuron_order[wit])

        self.weight_pointers[wit + 1:] += 1
        p = self.weight_pointers[wit]
        self.weights = np.insert(
            self.weights, p + self.weight_counts[wit], weight)
        self.connections = np.insert(
            self.connections, p + self.weight_counts[wit], from_idx)
        self.weight_counts[wit] += 1

    def add_neuron(
        self,
        from_idx: INDEX_DTYPE, to_idx: INDEX_DTYPE, order: INDEX_DTYPE,
        wf: VALUE_DTYPE, wt: VALUE_DTYPE, b: VALUE_DTYPE,
        func: Activation, m: bool = False
    ):

        self.add_activation(func)

        self.neuron_order = np.append(self.neuron_order, order)
        self.neuron_functions = np.append(
            self.neuron_functions, self.gnn_funcs[func])
        self.neuron_biases = np.append(self.neuron_biases, b)
        self.multiplicative = np.append(self.multiplicative, m)
        self.weight_pointers = np.append(
            self.weight_pointers, self.weights.size)
        self.weight_counts = np.append(self.weight_counts, 0)

        middle_idx = self.n_inputs + self.neuron_order.size - 1
        self.add_connection(from_idx, middle_idx, wf)
        self.add_connection(middle_idx, to_idx, wt)
