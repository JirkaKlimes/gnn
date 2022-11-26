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


for i in range(5000):
    model._add_neuron_randomly()

for i in range(50000):
    model._add_connection_randomly()

gnn.weights += np.random.uniform(-10e10, 10e10, gnn.weights.shape)
gnn.biases += np.random.uniform(-10e10, 10e10, size=gnn.biases.shape)

gnn.export(overwrite=True)

"""Export size: 1.838 MB"""