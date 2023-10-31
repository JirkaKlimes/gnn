from gnn import GNN
from gnn import ReLU, Linear
from pyvis.network import Network


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

# print_gnn()
gnn.add_neuron(0, 0.2, 4, -0.3, ReLU, 1.2, True)
# print_gnn()
gnn.add_neuron(0, 0.7, 4, 1.1, ReLU, 0.5, False)
# print_gnn()
gnn.add_connection(7, 6, 0.5)
states = gnn.push([1, 2, 3, 4])
print_gnn()
print(states)
export_gnn()
