from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class TrainingCallback(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        self.create()

    @abstractmethod
    def create(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, data):
        raise NotImplementedError

class PlotCallback(TrainingCallback):

    def create(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def plot_loss(self):
        x = np.array(self.data["val_iters"])
        y = np.array(self.data["loss"])
        y = np.log(y+1)

        self.ax.plot(x, y, "k-", label="ln(loss)", linewidth=1)

    def plot_new_neurons(self):
        x = np.array(self.data["new_neurons_iters"])
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