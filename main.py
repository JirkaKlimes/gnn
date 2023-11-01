from gnn import GNN
from gnn import ReLU, Linear
from gnn import CycleConnection, InputConnection, NonExistingNeuron, ConnectionAlreadyExists
from pyvis.network import Network
import random
import time


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


conns = []
NUM = 100
for i in range(NUM):
    from_indices = [
        *range(gnn.n_in), *(gnn.neuron_indices[:-int(gnn.layer_sizes[-1])] + gnn.n_in)]
    to_indices = [*(gnn.neuron_indices + gnn.n_in)]

    fi = random.choice(from_indices)
    ti = random.choice(to_indices)
    try:
        gnn.add_neuron(
            fi, random.normalvariate(),
            ti, random.normalvariate(),
            ReLU, random.normalvariate(),
            False,
        )
    except CycleConnection:
        pass
    print(f'Adding {NUM} neurons: {100*(i+1)/NUM:.2f}%', end='\r')

print(f'Adding {NUM} neurons finished!')
NUM = 10000
for i in range(NUM):
    from_indices = [
        *range(gnn.n_in), *(gnn.neuron_indices[:-int(gnn.layer_sizes[-1])] + gnn.n_in)]
    to_indices = [*(gnn.neuron_indices + gnn.n_in)]

    fi = random.choice(from_indices)
    ti = random.choice(to_indices)
    try:
        gnn.add_connection(fi, ti, random.normalvariate(), False)
    except CycleConnection:
        pass
    except ConnectionAlreadyExists:
        pass
    print(f'Adding {NUM} connections: {100*(i+1)/NUM:.2f}%', end='\r')
print(f'Adding {NUM} connections finished!')

print_gnn()
st = time.monotonic()
print(gnn.push([1, 2, 3, 4]))
et = time.monotonic()
print(f'Time taken: {(et-st)*1e3:.2f}ms')
# 41ms for 10k neurons and aprox. 100k connections
print(gnn)
export_gnn()
