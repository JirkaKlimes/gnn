import numpy as np
from typing import List, Optional
from gnn.src.activations import Activation, Linear


class CycleConnection(Exception):
    def __init__(self, from_idx, to_idx) -> None:
        msg = f"Cannot connect {from_idx} -> {to_idx} (would result in cycle)"
        super().__init__(msg)


class InputConnection(Exception):
    def __init__(self, from_idx, to_idx) -> None:
        msg = f"Cannot connect {from_idx} -> {to_idx} (is input)"
        super().__init__(msg)


class GNN:
    VALUE_DTYPE = np.float32
    INDEX_DTYPE = np.uint32

    def __init__(self,
                 n_inputs: int,
                 outputs: List[Activation],
                 value_dtype: np.dtype = VALUE_DTYPE,
                 index_dtype: np.dtype = INDEX_DTYPE
                 ):
        self.n_in = n_inputs
        self.n_out = len(outputs)
        self.valueT = value_dtype
        self.indexT = index_dtype

        self.max_index = np.iinfo(self.indexT).max

        self.funcs = {Linear: 0}
        list(map(self.add_activation, outputs))

        self.neuron_functions = np.uint8([self.funcs[f] for f in outputs])
        self.neuron_biases = np.zeros(self.n_out, self.valueT)
        self.multiplicative = np.zeros(self.n_out, bool)

        self.layer_sizes = np.full(1, self.n_out, self.indexT)
        self.layer_pointers = np.zeros(1, self.indexT)
        self.neuron_indices = np.arange(self.n_out, dtype=self.indexT)
        self.neuron_layers = np.zeros(self.n_out, dtype=self.indexT)

        self.conn_counts = np.zeros(self.n_out, dtype=self.indexT)
        self.conn_pointers = np.zeros(self.n_out, dtype=self.indexT)

        self.conn_weights = np.empty(0, dtype=self.valueT)
        self.conn_indices = np.empty(0, dtype=self.indexT)
        self.conn_recurrent = np.empty(0, bool)

    def __len__(self):
        return self.multiplicative.size

    @property
    def n_neurons(self):
        return len(self)

    @property
    def n_connections(self):
        return self.conn_weights.size

    def add_activation(self, func: Activation):
        if func not in self.funcs:
            self.funcs[func] = len(self.funcs)

    def __add_unconnected_neuron(self, function: Activation, bias: VALUE_DTYPE, multiplicative: bool):
        self.neuron_functions = np.append(
            self.neuron_functions, self.funcs[function])
        self.neuron_biases = np.append(self.neuron_biases, bias)
        self.multiplicative = np.append(self.multiplicative, multiplicative)

        self.conn_counts = np.append(self.conn_counts, 0)
        self.conn_pointers = np.append(self.conn_pointers, self.n_connections)

    def add_neuron(self,
                   from_idx: INDEX_DTYPE, from_weight: VALUE_DTYPE,
                   to_idx: INDEX_DTYPE, to_weight: VALUE_DTYPE,
                   function: Activation, bias: VALUE_DTYPE,
                   multiplicative: bool,
                   from_recurrent: Optional[bool] = False, to_recurrent: Optional[bool] = False,
                   layer: Optional[INDEX_DTYPE] = None
                   ):

        if from_idx >= self.n_in and self.neuron_layers[from_idx - self.n_in] > self.neuron_layers[to_idx - self.n_in]:
            raise CycleConnection(from_idx, to_idx)

        if to_idx < self.n_in:
            raise InputConnection(from_idx, to_idx)

        mid_idx = len(self)
        self.__add_unconnected_neuron(function, bias, multiplicative)
        # finds layers between the 2 connecting neurons where new one could be inserted
        lf = 0 if from_idx < self.n_in else \
            self.neuron_layers[from_idx - self.n_in] + 1
        lt = self.neuron_layers[to_idx - self.n_in] + 1
        sizes = self.layer_sizes[lf: min(lt, self.layer_sizes.size - 1)]
        if not sizes.size:
            # if no layer found, create new
            li = self.neuron_layers[to_idx - self.n_in]
            self.layer_sizes = np.insert(self.layer_sizes, li, 1)
            self.layer_pointers = np.insert(
                self.layer_pointers, li, self.layer_pointers[li])
            self.layer_pointers[li + 1:] += 1
            self.neuron_indices = np.insert(self.neuron_indices, li, mid_idx)
            self.neuron_layers = np.append(self.neuron_layers, li)
            self.neuron_layers[self.layer_pointers[li:]] += 1
        else:
            # if layers found, pick the smallest one
            li = np.argmin(sizes) + lf
            self.neuron_indices = np.insert(
                self.neuron_indices, self.layer_pointers[li] + self.layer_sizes[li], mid_idx)
            self.layer_sizes[li] += 1
            self.layer_pointers[li + 1:] += 1
            self.neuron_layers = np.append(self.neuron_layers, li)

        mid_idx += self.n_in
        self.add_connection(from_idx, mid_idx, from_weight, from_recurrent)
        self.add_connection(mid_idx, to_idx, to_weight, to_recurrent)

    def add_connection(self,
                       from_idx: INDEX_DTYPE, to_idx: INDEX_DTYPE,
                       weight: VALUE_DTYPE, recurrent: Optional[bool] = False
                       ):

        wit = to_idx - self.n_in
        wif = from_idx - self.n_in

        if from_idx >= self.n_in and self.neuron_layers[from_idx - self.n_in] > self.neuron_layers[to_idx - self.n_in]:
            raise CycleConnection(from_idx, to_idx)

        if to_idx < self.n_in:
            raise InputConnection(from_idx, to_idx)

        if from_idx >= self.n_in and self.neuron_layers[from_idx - self.n_in] == self.neuron_layers[to_idx - self.n_in]:
            li = self.neuron_layers[wit]
            self.layer_sizes[li] -= 1
            self.layer_sizes = np.insert(self.layer_sizes, li, 1)
            p = self.layer_pointers[li]
            self.layer_pointers = np.insert(
                self.layer_pointers, li, p)
            self.layer_pointers[li + 1] += 1
            layers = self.neuron_indices[p:self.layer_sizes[li] + 1]
            layers = np.delete(layers, np.where(layers == wif))
            layers = np.insert(layers, 0, wif)
            self.neuron_indices[p:self.layer_sizes[li] + 1] = layers
            self.neuron_layers[np.where(self.neuron_layers >= li)] += 1
            self.neuron_layers[wif] = li

        self.conn_pointers[wit + 1:] += 1
        p = self.conn_pointers[wit]
        self.conn_weights = np.insert(
            self.conn_weights, p + self.conn_counts[wit], weight)
        self.conn_indices = np.insert(
            self.conn_indices, p + self.conn_counts[wit], from_idx)
        self.conn_recurrent = np.insert(
            self.conn_recurrent, p + self.conn_counts[wit], recurrent)
        self.conn_counts[wit] += 1
