import torch
from typing import List, Callable
from .trainingwrapper import TrainingWrapper

class SimpleConvNet(TrainingWrapper):

    def __init__(self, channels_per_layer: List[int], kernel_sizes: List[int],
            in_channels: int, out_units: int, loss_fun: Callable, metric: Callable,
            lr: float=0.01, weight_decay: float=1e-4, momentum: float=0.9,
            gamma: float=0.99):
        # generate layers
        layers = self.build_layers(in_channels, channels_per_layer, kernel_sizes,
            out_units)
        super().__init__(net=layers, loss_fun=loss_fun, metric=metric, lr=lr,
                         weight_decay=weight_decay, momentum=momentum, gamma=gamma)

    def build_layers(self, in_channels, channels_per_layer, kernel_sizes,
            out_units):
        layers = torch.nn.Sequential()
        for c, k in zip(channels_per_layer, kernel_sizes):
            block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, c, k, padding="same"),
                torch.nn.GroupNorm(c, c, affine=False),
                torch.nn.GELU())
            layers.append(block)
            in_channels = c
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.LazyLinear(out_units))
        return layers


class RecurrentConvNet(SimpleConvNet):
    def __init__(self, channels_per_layer: List[int], kernel_sizes: List[int],
            in_channels: int, out_units: int, num_steps: int,
            loss_fun: Callable, metric: Callable,
            lr: float=0.01, weight_decay: float=1e-4, momentum: float=0.9,
            gamma: float=0.99):
        super().__init__(
            channels_per_layer=channels_per_layer,
            kernel_sizes=kernel_sizes,
            in_channels=in_channels,
            out_units=out_units,
            loss_fun=loss_fun,
            metric=metric,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            gamma=gamma)
        self.layers, self.lateral = self.build_layers(in_channels,
            channels_per_layer, kernel_sizes, out_units)
        self.num_steps = num_steps

    def build_layers(self, in_channels, channels_per_layer, kernel_sizes,
            out_units):
        # Generate self.layers as a Sequential like in SimpleConvNet.
        # These are the forward layers (i.e., the filters used to
        # pass information up between the recurrent units).
        # Additionally, we generate a list self.lateral of layers
        # which perform the recurrent filtering over time.
        layers = torch.nn.Sequential()
        lateral = torch.nn.ModuleList()
        for c, k in zip(channels_per_layer, kernel_sizes):
            # generate forward layer
            block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, c, k, padding="same"),
                torch.nn.GroupNorm(c, c, affine=False),
                torch.nn.GELU())
            layers.append(block)
            # generate lateral filters
            lateral_block = torch.nn.Sequential(
                torch.nn.Conv2d(c, c, k, padding="same"),
                torch.nn.GroupNorm(c, c, affine=False),
                torch.nn.GELU())
            lateral.append(lateral_block)
            in_channels = c
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.LazyLinear(out_units))
        return layers, lateral

    def forward(self, x):
        state = []
        for step in range(self.num_steps):
            o = x
            # iterate over conv layers (forward and lateral layer pairs)
            for (i, (forward, lateral)) in enumerate(zip(self.layers, self.lateral)):
                # on the first step, only run forward layers and record state
                if step == 0:
                    o = forward(o)
                    state.append(o)
                else:
                    h = forward(o)
                    r = lateral(state[i])
                    o = h + r # TODO: replace this by better formula
                    state[i] = o # store output for next iteration
        # after recurrent iterations finish, pass output of highest layer
        # on to fully connected part
        return self.layers[-2:](o)
