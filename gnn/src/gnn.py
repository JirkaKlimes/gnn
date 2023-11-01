import numpy as np
from typing import List, Optional, Union, Dict
from gnn.src.activations import Activation, Linear


class CycleConnection(Exception):
    def __init__(self, from_idx, to_idx) -> None:
        msg = f"Cannot connect {from_idx} -> {to_idx} (would result in cycle)"
        super().__init__(msg)


class InputConnection(Exception):
    def __init__(self, from_idx, to_idx) -> None:
        msg = f"Cannot connect {from_idx} -> {to_idx} (is input)"
        super().__init__(msg)


class NonExistingNeuron(Exception):
    def __init__(self, from_idx, to_idx) -> None:
        msg = f"Cannot connect {from_idx} -> {to_idx} (doesn't exist)"
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

        self.funcs: Dict[Activation: int] = {Linear: 0}
        self.__funcs_T: List[Activation] = []
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

    @property
    def is_recurrent(self):
        return np.any(self.conn_recurrent)

    @property
    def funcs_T(self):
        if not (self.__funcs_T is None or len(self.__funcs_T) == len(self.funcs)):
            self.__funcs_T = list(self.funcs.keys())
        return self.__funcs_T

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

        if from_idx == to_idx:
            raise CycleConnection(from_idx, to_idx)

        if from_idx >= len(self) + self.n_in or to_idx >= len(self) + self.n_in:
            raise NonExistingNeuron(from_idx, to_idx)

        if to_idx < self.n_in:
            raise InputConnection(from_idx, to_idx)

        if from_idx >= self.n_in and self.neuron_layers[from_idx - self.n_in] > self.neuron_layers[to_idx - self.n_in]:
            raise CycleConnection(from_idx, to_idx)

        mid_idx = len(self)
        self.__add_unconnected_neuron(function, bias, multiplicative)
        # finds layers between the 2 connecting neurons where new one could be inserted
        lf = 0 if from_idx < self.n_in else \
            self.neuron_layers[from_idx - self.n_in] + 1
        lt = self.neuron_layers[to_idx - self.n_in] + 1
        sizes = self.layer_sizes[lf: min(lt, self.layer_sizes.size - 1)]
        if lf == lt:
            li = self.neuron_layers[to_idx - self.n_in]
            self.neuron_indices = np.insert(
                self.neuron_indices, self.layer_pointers[li] + self.layer_sizes[li], mid_idx)
            self.layer_sizes[li] += 1
            self.layer_pointers[li + 1:] += 1
            self.neuron_layers = np.append(self.neuron_layers, li)
        elif not sizes.size:
            # if no layer found, create new
            li = self.neuron_layers[to_idx - self.n_in]
            self.layer_sizes = np.insert(self.layer_sizes, li, 1)
            self.layer_pointers = np.insert(
                self.layer_pointers, li, self.layer_pointers[li])
            self.layer_pointers[li + 1:] += 1
            self.neuron_indices = np.insert(
                self.neuron_indices, self.layer_pointers[li] + self.layer_sizes[li] - 1, mid_idx)
            self.neuron_layers = np.append(self.neuron_layers, li)
            self.neuron_layers[np.where(self.neuron_layers >= li)] += 1
            self.neuron_layers[mid_idx] = li
        else:
            # if layers found, and no specific set, pick the smallest one
            if layer is not None and layer in range(lf, lf + len(sizes)):
                li = layer
            else:
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

        # if neurons reside in the same layer, we have to split it
        if from_idx >= self.n_in and self.neuron_layers[from_idx - self.n_in] == self.neuron_layers[to_idx - self.n_in]:
            li = self.neuron_layers[wit]
            p = self.layer_pointers[li]
            layers = self.neuron_indices[p:p + self.layer_sizes[li]]
            layers = np.delete(layers, np.where(layers == wif))
            layers = np.insert(layers, 0, wif)
            self.neuron_indices[p:p + self.layer_sizes[li]] = layers
            self.neuron_layers[np.where(self.neuron_layers >= li)] += 1
            self.neuron_layers[wif] = li
            self.layer_sizes[li] -= 1
            self.layer_sizes = np.insert(self.layer_sizes, li, 1)
            self.layer_pointers = np.insert(
                self.layer_pointers, li, p)
            self.layer_pointers[li + 1] += 1

        self.conn_pointers[wit + 1:] += 1
        p = self.conn_pointers[wit]
        self.conn_weights = np.insert(
            self.conn_weights, p + self.conn_counts[wit], weight)
        self.conn_indices = np.insert(
            self.conn_indices, p + self.conn_counts[wit], from_idx)
        self.conn_recurrent = np.insert(
            self.conn_recurrent, p + self.conn_counts[wit], recurrent)
        self.conn_counts[wit] += 1

    def push(self, x: Union[np.ndarray, list], prev_states: Optional[Union[np.ndarray, list]] = None, return_states: Optional[bool] = False):
        if isinstance(x, list):
            x = np.array(x, self.valueT)

        assert x.size == self.n_in

        if prev_states is None and self.is_recurrent:
            prev_states = np.zeros(len(self), dtype=self.valueT)

        states = np.concatenate([x, np.empty(len(self), dtype=self.valueT)])
        for layer_s, layer_p in zip(self.layer_sizes, self.layer_pointers):
            for p in range(layer_p, layer_p + layer_s):
                n = self.neuron_indices[p]
                conn_c, conn_p = self.conn_counts[n], self.conn_pointers[n]
                f, b = self.neuron_functions[n], self.neuron_biases[n]
                if self.multiplicative[n]:
                    z = b
                    for c in range(conn_p, conn_p + conn_c):
                        r, w, i = self.conn_recurrent[c], self.conn_weights[c], self.conn_indices[c]
                        z *= ((prev_states if r else states)[i] + w)
                    states[n + self.n_in] = self.funcs_T[f].fn(z)
                else:
                    z = b
                    for c in range(conn_p, conn_p + conn_c):
                        r, w, i = self.conn_recurrent[c], self.conn_weights[c], self.conn_indices[c]
                        z += ((prev_states if r else states)[i] * w)
                    states[n + self.n_in] = self.funcs_T[f].fn(z)

        if return_states:
            return states

        return states[self.n_in:self.n_in + self.n_out]
