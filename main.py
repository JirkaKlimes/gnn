import numpy as np

from activations import Sigmoid, Relu, Kelu, Tanh
from models import UnnamedModel1
from losses import MeanSquaredError
from gnn import Gnn



# np.random.seed(1)

inputs = 2
outputs = 2

gnn = Gnn(inputs, outputs)

model = UnnamedModel1(gnn)

model.add_activation_functions(Sigmoid())

model.set_loss_function(MeanSquaredError())

model.build()


train_x = np.array([4, 3])
train_y = np.array([0.1, 0.3])


for i in range(10000):

    biases_grad, weights_grad = gnn.backprop(train_x, train_y)
    gnn.biases -= biases_grad * 0.1
    gnn.weights -= weights_grad * 0.1

    loss = MeanSquaredError().fn(train_y, gnn.push(train_x))
    print(loss)

    if str(loss) == 'nan':
        quit()

    if loss < 0.01:
        print(i)
        break