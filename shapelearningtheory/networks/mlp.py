import torch
from typing import Callable
from .trainingwrapper import TrainingWrapper

class MLP(TrainingWrapper):
    "Non-linear multi-layer perceptron"
    def __init__(self, num_inputs: int, num_hidden: int, num_layers: int,
            num_outputs: int, loss_fun: Callable, metric: Callable,
            lr: float=0.01, weight_decay: float=1e-4, momentum: float=0.9,
            gamma: float=0.99):
        # generate layers:
        layers = torch.nn.Sequential()
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(num_inputs, num_hidden))
        layers.append(torch.nn.LayerNorm(num_hidden, elementwise_affine=False))
        layers.append(torch.nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(torch.nn.Linear(num_hidden, num_hidden))
            layers.append(torch.nn.LayerNorm(num_hidden, elementwise_affine=False))
            layers.append(torch.nn.GELU())
        layers.append(torch.nn.Linear(num_hidden, num_outputs))
        # set up network
        super().__init__(net=layers, loss_fun=loss_fun, metric=metric, lr=lr,
            weight_decay=weight_decay, momentum=momentum, gamma=gamma)
    
    def layer_activations(self, x):
        outputs = {'image': x}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, torch.nn.Linear):
                outputs["layer {}".format(i)] = x
        return outputs