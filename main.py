from activations import Sigmoid, Relu, Kelu
from trainers import UnnamedTrainer1
from losses import MeanSquaredError
from gnn import Gnn




inputs = 2
outputs = 2

gnn = Gnn(inputs, outputs)

trainer = UnnamedTrainer1(gnn)

trainer.add_activation_function(Relu())
trainer.add_activation_function(Sigmoid())


trainer.build()
