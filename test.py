from src.gnn import Gnn
from src.activations import Relu, Identity
from src.losses import MeanSquaredError
from src.models import LINC
from src.callbacks import StdOutCallback

import numpy as np
import time
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')


N_INPUTS = 5
N_OUTPUTS = 5
N_NEURONS = 500
N_CONNECTIONS = 5000
TARGET_LOSS = 0.001
SIZE = 50


print(f"Creating GNN with {N_INPUTS} inputs and {N_OUTPUTS} outputs...")
t1 = time.perf_counter()*1000
gnn = Gnn(5, 5)
gnn.hidden_act_fn = Relu()
gnn.output_act_fn = Identity()
gnn.loss_fn = MeanSquaredError()
print(f"Time taken: {(time.perf_counter()*1000 - t1):02f}ms ✅\n")


print("Building model...")
t1 = time.perf_counter()*1000
model = LINC(gnn)
model.build(True)
print(f"Time taken: {(time.perf_counter()*1000 - t1):02f}ms ✅\n")


print(f"Adding {N_NEURONS} neurons...")
t1 = time.perf_counter()*1000
for i in range(N_NEURONS):
    model._add_neuron_randomly()
print(f"Time taken: {(time.perf_counter()*1000 - t1):02f}ms ✅\n")


print(f"Adding {N_CONNECTIONS} connections...")
t1 = time.perf_counter()*1000
for i in range(N_CONNECTIONS):
    model._add_connection_randomly()
print(f"Time taken: {(time.perf_counter()*1000 - t1):02f}ms ✅\n")


print("Exporting gnn...")
t1 = time.perf_counter()*1000
gnn.export(overwrite=True)
print(f"Time taken: {(time.perf_counter()*1000 - t1):02f}ms ✅\n")


print("Loading gnn from file...")
t1 = time.perf_counter()*1000
loaded_gnn = Gnn.from_file()

print(f"\t{'Order:':<15}{'✅' if np.all(loaded_gnn.order == gnn.order) else '❌'}")
print(f"\t{'Digraph:':<15}{'✅' if np.all(loaded_gnn.digraph == gnn.digraph) else '❌'}")
print(f"\t{'Biases:':<15}{'✅' if np.all(loaded_gnn.biases == gnn.biases) else '❌'}")
print(f"\t{'Weights:':<15}{'✅' if np.all(loaded_gnn.weights == gnn.weights) else '❌'}")
print(f"\t{'Creation date:':<15}{'✅' if loaded_gnn._creation_date == gnn._creation_date else '❌'}")
print(f"\t{'Training date:':<15}{'✅' if loaded_gnn._last_training_date == gnn._last_training_date else '❌'}")
print(f"\t{'Hidden act fn:':<15}{'✅' if str(loaded_gnn.hidden_act_fn) == str(gnn.hidden_act_fn) else '❌'}")
print(f"\t{'Output act fn:':<15}{'✅' if str(loaded_gnn.output_act_fn) == str(gnn.output_act_fn) else '❌'}")
print(f"Time taken: {(time.perf_counter()*1000 - t1):02f}ms ✅\n")


gnn = Gnn(784, 10)
gnn.hidden_act_fn = Relu()
gnn.output_act_fn = Identity()
gnn.loss_fn = MeanSquaredError()
model = LINC(gnn)
model.build(True)

print(f"Loading MNIST dataset...")
(x_train, _y_train), (x_test, _y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)[:SIZE]
_y_train = _y_train[:SIZE]

y_train = np.zeros((_y_train.size, _y_train.max() + 1))
y_train[np.arange(_y_train.size), _y_train] = 1

dataset = model.convert_dataset((x_train.reshape(SIZE, -1), y_train))
print(f"Dataset laoded ✅")


print(f"Training on {SIZE} images for mse of {TARGET_LOSS}")
t1 = time.perf_counter()*1000
model.train(dataset, 20, target_loss = TARGET_LOSS, vf=20,
            lr = 0.2, ls = -0.005, gr = 0.01, gd = 10,
            nop = 0.05, callbacks=[StdOutCallback(False)])
print(f"Time taken: {(time.perf_counter()*1000 - t1):02f}ms ✅\n")

