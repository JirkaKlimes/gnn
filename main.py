from gnn import GNN
from gnn import ReLU, Linear

n_inputs = 4
outputs = [ReLU, Linear]

gnn = GNN(n_inputs, outputs)


def print_gnn():
    print('neuron_order:', gnn.neuron_order)
    print('neuron_functions:', gnn.neuron_functions)
    print('neuron_biases:', gnn.neuron_biases)
    print('multiplicative:', gnn.multiplicative)
    print('weight_pointers:', gnn.weight_pointers)
    print('weight_counts:', gnn.weight_counts)
    print('weights:', gnn.weights)
    print('connections:', gnn.connections)


print_gnn()
gnn.add_connection(0, 4, 0.1)
print('=' * 10)
print_gnn()
gnn.add_connection(2, 4, 0.2)
print('=' * 10)
print_gnn()
gnn.add_connection(1, 5, -0.4)
print('=' * 10)
print_gnn()
gnn.add_connection(3, 5, 0.5)
print('=' * 10)
print_gnn()
gnn.add_neuron(0, 5, gnn.max_index // 2, 0.3, 0.7, 0, ReLU, True)
print('=' * 10)
print_gnn()
