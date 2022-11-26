from src.activations import Relu, Identity
from src.losses import MeanSquaredError
from src.callbacks import PlotCallback, StdOutCallback
from src.models import UnnamedModel1
from src.gnn import Gnn

import numpy as np
import tensorflow as tf

gnn = Gnn(784, 10)

model = UnnamedModel1(gnn)
model.set_hidden_act_function(Relu())
model.set_output_act_function(Identity())
model.set_loss_function(MeanSquaredError())
model.build()


SIZE = 160
(x_train, _y_train), (x_test, _y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)[:SIZE]
x_test = tf.keras.utils.normalize(x_test, axis=1)[:SIZE]
_y_train = _y_train[:SIZE]
_y_test = _y_test[:SIZE]

y_train = np.zeros((_y_train.size, _y_train.max() + 1))
y_train[np.arange(_y_train.size), _y_train] = 1

y_test = np.zeros((_y_test.size, _y_test.max() + 1))
y_test[np.arange(_y_test.size), _y_test] = 1


dataset = model.convert_dataset((x_train.reshape(SIZE, -1), y_train))


model.train(dataset, 80, target_loss = 0.05, vf=20,
            lr = 0.2, ls = -0.005, gr = 0.01, gd = 10,
            nop = 0.05,
            callbacks=[PlotCallback(), StdOutCallback()])
