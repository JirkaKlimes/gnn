from gnn import GNN
from gnn import ReLU, Linear
from gnn import CycleConnection, InputConnection, NonExistingNeuron
from pyvis.network import Network
import random


def print_gnn():
    print('=' * 20)
    print('neuron_functions:', gnn.neuron_functions)
    print('neuron_biases:', gnn.neuron_biases)
    print('multiplicative:', gnn.multiplicative)
    print('layer_sizes:', gnn.layer_sizes)
    print('layer_pointers:', gnn.layer_pointers)
    print('neuron_indices:', gnn.neuron_indices)
    print('neuron_layers:', gnn.neuron_layers)
    print('conn_pointers:', gnn.conn_pointers)
    print('conn_counts:', gnn.conn_counts)
    print('conn_weights:', gnn.conn_weights)
    print('conn_indices:', gnn.conn_indices)
    print('conn_recurrent:', gnn.conn_recurrent)


def export_gnn():
    net = Network(
        notebook=True, directed=True,
        neighborhood_highlight=True,
        width='100%', height='900',
        cdn_resources='in_line'
    )
    net.barnes_hut()

    for i in range(gnn.n_in):
        net.add_node(i, color='#57dc88', label=str(i))

    for i in range(gnn.n_in, gnn.n_in + gnn.n_out):
        net.add_node(i, color='#f45b5c', label=str(i))

    for i in range(gnn.n_in + gnn.n_out, gnn.n_in + len(gnn)):
        m = gnn.multiplicative[i - gnn.n_in]
        net.add_node(i, color='#4ad0df', label=f'(M) {i}' if m else str(i))

    for i, (p, c) in enumerate(zip(gnn.conn_pointers, gnn.conn_counts)):
        for j in range(p, p + c):
            f, w = gnn.conn_indices[j], gnn.conn_weights[j]

            net.add_edge(int(f), i + gnn.n_in, label=str(w))

    net.save_graph('graph.html')


n_inputs = 4
outputs = [ReLU, Linear]

gnn = GNN(n_inputs, outputs)

# conns = [(2, 5), (3, 6), (2, 6), (1, 5), (0, 5), (6, 5),
#          (7, 6), (12, 4), (13, 5), (12, 9), (7, 8)]

# for i, j in conns:
#     gnn.add_neuron(i, 1, j, 1, ReLU, 1, False)
#     print_gnn()
# export_gnn()

# gnn.add_neuron(1, 1, 5, 1, ReLU, 1, False)
# gnn.add_neuron(3, 1, 4, 1, ReLU, 1, False)
# gnn.add_neuron(2, 1, 4, 1, ReLU, 1, False)
# gnn.add_neuron(4, 1, 5, 1, ReLU, 1, False)
# print_gnn()

conns = []
for _ in range(1000):
    from_indices = [
        *range(gnn.n_in), *(gnn.neuron_indices[:-int(gnn.layer_sizes[-1])] + gnn.n_in)]
    to_indices = [*(gnn.neuron_indices + gnn.n_in)]

    fi = random.choice(from_indices)
    ti = random.choice(to_indices)
    # print_gnn()
    print(fi, ti)
    try:
        gnn.add_neuron(fi, 1, ti, 1, ReLU, 1, False)
        conns.append((fi, ti))
    except CycleConnection:
        pass
    print(conns)

print_gnn()
export_gnn()
