import torch
from typing import List, Callable
from .trainingwrapper import TrainingWrapper
from .solutionnetworks import RootLU

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
    
    def get_layers_of_interest(self):
        blocks = {f"block{i}": i for i in range(1, len(self.layers)-1)}
        blocks["output"] = len(self.layers)
        return blocks
    
    def __getitem__(self, idx):
        return self.layers[idx]

class RecurrentBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.forward_conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, 
            padding="same"
        )
        self.lateral_conv = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, 
            padding="same"
        )
        self.norm = torch.nn.GroupNorm(
            out_channels, out_channels, affine=False
        )
        self.activation = torch.nn.GELU()

    def forward(self, x, h_old):
        h = self.forward_conv(x)
        if h_old is not None:
            h += self.lateral_conv(h_old)
        return self.activation(self.norm(h))

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
        self.layers = self.build_layers(in_channels,
        self.layers = self.build_layers(in_channels,
            channels_per_layer, kernel_sizes, out_units)
        self.num_steps = num_steps

    def build_layers(self, in_channels, channels_per_layer, kernel_sizes,
            out_units):
        layers = torch.nn.Sequential()
        for c, k in zip(channels_per_layer, kernel_sizes):
            # generate forward layer
            block = RecurrentBlock(in_channels, c, k)
            block = RecurrentBlock(in_channels, c, k)
            layers.append(block)
            in_channels = c
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.LazyLinear(out_units))
        return layers
        return layers

    def forward(self, x):
        state = []
        for step in range(self.num_steps):
            forward_input = x
            for (i, layer) in enumerate(self.layers):
            forward_input = x
            for (i, layer) in enumerate(self.layers):
                if step == 0:
                    layer_output = layer(forward_input)
                    layer_output = layer(forward_input)
                else:
                    lateral_input = state[i]
                    layer_output = layer(forward_input, lateral_input)
                    state[i] = layer_output
                forward_input = layer_output
        return self.layers[-2:](layer_output)
                    lateral_input = state[i]
                    layer_output = layer(forward_input, lateral_input)
                    state[i] = layer_output
                forward_input = layer_output
        return self.layers[-2:](layer_output)


class RectangleLikeConvNet(TrainingWrapper):
    """Network with same architecture as CRectangleConvNet, but with trained weights."""
    def __init__(self, loss_fun: Callable, metric: Callable,
            lr: float=0.01, weight_decay: float=1e-4, momentum: float=0.9,
            gamma: float=0.99):
        layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 2, kernel_size=3), # Laplace
            RootLU(),
            torch.nn.Conv2d(2, 1, kernel_size=1), # SumChannels
            torch.nn.Conv2d(1, 4, kernel_size=3), # Sobel
            RootLU(coeff=0.5),
            torch.nn.Conv2d(4, 2, kernel_size=1), # Borders
            torch.nn.Conv2d(2, 4, kernel_size=13), # Distances
            torch.nn.Conv2d(4, 2, kernel_size=1),
            torch.nn.AdaptiveMaxPool2d(output_size=1),
            torch.nn.Flatten()
        )
        super().__init__(net=layers, loss_fun=loss_fun, metric=metric, lr=lr,
                         weight_decay=weight_decay, momentum=momentum, gamma=gamma)
        
    def forward(self, x):
        return self.layers(x)
    
class ColorLikeConvNet(TrainingWrapper):
    def __init__(self, loss_fun: Callable, metric: Callable,
            lr: float=0.01, weight_decay: float=1e-4, momentum: float=0.9,
            gamma: float=0.99):
        layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 2, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool2d(output_size=1),
            torch.nn.Flatten(),
            torch.nn.Linear(2, 2)
        )
        super().__init__(net=layers, loss_fun=loss_fun, metric=metric, lr=lr,
                         weight_decay=weight_decay, momentum=momentum, gamma=gamma)
        
    def forward(self, x):
        return self.layers(x)
    
class TextureLikeConvNet(TrainingWrapper):
    def __init__(self, loss_fun: Callable, metric: Callable,
            lr: float=0.01, weight_decay: float=1e-4, momentum: float=0.9,
            gamma: float=0.99):
        layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=21, padding="same"), # Gabor
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 2, kernel_size=1),
            torch.nn.AdaptiveMaxPool2d(output_size=1),
            torch.nn.Flatten()
        )
        super().__init__(net=layers, loss_fun=loss_fun, metric=metric, lr=lr,
                         weight_decay=weight_decay, momentum=momentum, gamma=gamma)
        
    def forward(self, x):
        return self.layers(x)

class LTLikeConvNet(TrainingWrapper):
    def __init__(self, loss_fun: Callable, metric: Callable,
            lr: float=0.01, weight_decay: float=1e-4, momentum: float=0.9,
            gamma: float=0.99):
        layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 2, kernel_size=3), # Laplace
            RootLU(),
            torch.nn.Conv2d(2, 1, kernel_size=1), # SumChannels
            torch.nn.Conv2d(1, 4, kernel_size=3), # Sobel
            RootLU(coeff=0.5),
            torch.nn.Conv2d(4, 1, kernel_size=1),
            torch.nn.Conv2d(1, 4, kernel_size=5), # End detectors
            RootLU(),
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            torch.nn.Conv2d(4, 1, kernel_size=1),
            torch.nn.Conv2d(1, 2, kernel_size=1),
            torch.nn.Flatten()
        )
        super().__init__(net=layers, loss_fun=loss_fun, metric=metric, lr=lr,
                         weight_decay=weight_decay, momentum=momentum, gamma=gamma)
        
    def forward(self, x):
        return self.layers(x)
