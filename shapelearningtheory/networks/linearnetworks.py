import torch
from typing import Callable
from .trainingwrapper import TrainingWrapper

class ShallowLinear(TrainingWrapper):
    """Single-layer linear network."""
    def __init__(self, num_inputs: int, num_outputs: int, loss_fun: Callable,
            metric: Callable, lr: float=0.01, weight_decay: float=1e-4,
            momentum: float=0.9, gamma: float=0.99):
        super().__init__(
            net = torch.nn.Linear(num_inputs, num_outputs),
            loss_fun=loss_fun, metric=metric, lr=lr, weight_decay=weight_decay,
            momentum=momentum, gamma=gamma
        )
    

class DeepLinear(TrainingWrapper):
    "Linear networks with multiple fully-connected layers."
    def __init__(self, num_inputs: int, num_hidden: int, num_layers: int,
            num_outputs: int, loss_fun: Callable, metric: Callable,
            lr: float=0.01, weight_decay: float=1e-4, momentum: float=0.9,
            gamma: float=0.99):
        # generate layers:
        layers = torch.nn.Sequential()
        layers.append(torch.nn.Linear(num_inputs, num_hidden))
        for _ in range(num_layers - 2):
            layers.append(torch.nn.Linear(num_hidden, num_hidden))
        layers.append(torch.nn.Linear(num_hidden, num_outputs))
        # set up model
        super().__init__(net=layers, loss_fun=loss_fun, metric=metric, lr=lr,
            weight_decay=weight_decay, momentum=momentum, gamma=gamma)