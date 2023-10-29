from gnn import GNN
from gnn import ReLU, Linear

n_inputs = 4
outputs = [ReLU, Linear]

gnn = GNN(n_inputs, outputs)
print(gnn.neuron_order)
print(gnn.neuron_functions)
print(gnn.neuron_biases)
print(gnn.multiplicative)
print(gnn.weights)
print(gnn.weight_pointers)
print(gnn.weight_counts)
