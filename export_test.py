from src.activations import Relu, Identity
from src.losses import MeanSquaredError
from src.models import UnnamedModel1
from src.gnn import Gnn

import numpy as np

gnn = Gnn(5, 5)

model = UnnamedModel1(gnn)
model.set_hidden_act_function(Relu())
model.set_output_act_function(Identity())
model.set_loss_function(MeanSquaredError())
model.build()


for _ in range(200):
    model._add_neuron_randomly()

for _ in range(3000):
    model._add_connection_randomly()

gnn.weights += np.random.uniform(-10e10, 10e10, gnn.weights.shape)
gnn.biases += np.random.uniform(-10e10, 10e10, size=gnn.biases.shape)

gnn.export(overwrite=True)