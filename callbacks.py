from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import os

class TrainingCallback(ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        self.update(data)

    def create(self):
        pass

    @abstractmethod
    def update(self, data):
        raise NotImplementedError

class PlotCallback(TrainingCallback):

    def create(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def plot_loss(self):
        x = np.array(self.data["epochs"])
        y = np.array(self.data["loss"])
        y = np.log(y+1)

        self.ax.plot(x, y, "k-", label="ln(loss)", linewidth=1)

    def plot_new_neurons(self):
        x = np.array(self.data["new_neurons_epochs"])
        y = np.array(self.data["new_neurons_loss"])
        y = np.log(y+1)
        
        self.ax.plot(x, y, "g.", label="+Neurons")


    def update(self, data):
        self.data = data
        self.ax.cla()
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Loss")

        self.plot_loss()
        self.plot_new_neurons()

        self.ax.legend(loc="upper right")
        plt.draw()
        plt.pause(0.0001)

class StdOutCallback(TrainingCallback):
    def __init__(self, clear_stdout: bool = True):
        super().__init__()
        self.clear_stdout = clear_stdout

    def update(self, data):
        epoch = data["epochs"][-1]
        loss = data["loss"][-1]
        loss_fn = data['gnn'].loss_fn
        N_neurons = data['gnn'].order.shape[0] - data['gnn'].N_inputs
        N_layers = len(data['gnn'].order_values)

        if self.clear_stdout:
            os.system('cls')
            print(f"|=============================")
            print(f"| Epoch: {epoch}")
            print(f"| {loss_fn}: {loss}")
            print(f"| Number of neurons: {N_neurons}")
            print(f"| Number of layers: {N_layers}")
            print(f"|=============================")
        else:
            print(f"|===========================================================================")
            print(f"| Epoch: {epoch} | {loss_fn}: {round(loss, 5)} | Number of neurons: {N_neurons}")