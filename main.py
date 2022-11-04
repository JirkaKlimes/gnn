import numpy as np

from activations import Sigmoid, Relu
from models import UnnamedModel1
from losses import MeanSquaredError
from gnn import Gnn

gnn = Gnn(2, 2)

model = UnnamedModel1(gnn)

model.set_hidden_act_function(Relu())
model.set_output_act_function(Sigmoid())

model.set_loss_function(MeanSquaredError())

model.build()

gnn.weights += np.random.normal(size=gnn.weights.shape)

train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_y = np.array([[0, 1], [1, 0], [1, 0], [1, 0]])
# train_y = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])


for i in range(100000):

    sum_biases_grad = np.zeros_like(gnn.biases)
    sum_weights_grad = np.zeros_like(gnn.weights)
    for x, y in zip(train_x, train_y):
        biases_grad, weights_grad = gnn.backprop(x, y)

        sum_biases_grad += biases_grad
        sum_weights_grad += weights_grad
    
    b_grad = sum_biases_grad / train_x.shape[0]
    w_grad = sum_weights_grad / train_x.shape[0]

    gnn.biases -= b_grad * 0.1
    gnn.weights -= w_grad * 0.1
    
    loss = 0
    for x, y in zip(train_x, train_y):
        loss += MeanSquaredError().fn(y, gnn.push(x))

    loss /= train_x.shape[0]

    print(loss)

    if loss < 0.01:
        print(i)
        break