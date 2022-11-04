import numpy as np

from activations import Relu, Identity
from models import UnnamedModel1
from losses import MeanSquaredError
from gnn import Gnn



gnn = Gnn(2, 2)

model = UnnamedModel1(gnn)

model.set_hidden_act_function(Relu())
model.set_output_act_function(Identity())

model.set_loss_function(MeanSquaredError())

model.build()
