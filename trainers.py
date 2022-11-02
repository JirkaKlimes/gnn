from abc import ABC, abstractmethod
from gnn import Gnn
from activations import Activation
from losses import Loss

class Trainer(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def build(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, trainX, trainY, **kwargs):
        raise NotImplementedError

class UnnamedTrainer1(Trainer):
    def __init__(self, gnn: Gnn):
        """
        A <insert some unique and cool name here> Trainer that:
            - adds new neurons and connections when loss stops droping
            - initializes weights diferently to ones that exists so they can learn new features
            - adds momentum to gradient for lower chance of getting stuck in local minima
        """

        self.gnn = gnn
        self._isBuilt = False
        self._initialize_hyperparameters()

        super().__init__()


    def __str__(self):
        return "Unnamed Trainer 1"

    def _initialize_hyperparameters(self):

        # activation functions that will be used
        self.activation_functions = []

        # loss function
        self.loss_fn = None

    def add_activation_function(self, function: Activation):
        """
        Adds funtion to a list of activations that can/will be used
        """
        if self._isBuilt:
            print(f"Cannot add activation function to an already build network.")
            return

        if str(function) in list(map(str, self.activation_functions)):
            print(f"Trainer already has \"{function}\" activation.")
            return

        self.activation_functions.append(function)

    def build(self):
        self._isBuilt = True

    def train(self, trainX, trainY, target_loss: float = 0):
        """
        Trains the network on dataset
        """


