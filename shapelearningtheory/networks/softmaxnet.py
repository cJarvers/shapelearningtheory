import torch
import torchvision
from torch import Tensor
from typing import Callable, List, Literal
from .trainingwrapper import TrainingWrapper

class SpatialSoftmax2d(torch.nn.Module):
    """Non-linearity like layer that applies a softmax in a local neighborhood.
    
    Shape:
        - Input: `(N, C, H_{in}, W_{in})`
        - Output: `N, C, H_{out}, W_{out}`

    Args:
    - 
    """
    def __init__(self, in_channels: int, kernel_size: int, sigma: float, temperature: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.temperature = temperature
        # initialize weights to Gaussian
        weights = torch.zeros(in_channels, 1, kernel_size, kernel_size)
        weights[:, 0, kernel_size//2, kernel_size//2] = 1
        weights = torchvision.transforms.functional.gaussian_blur(weights, kernel_size, sigma)
        weights /= weights.max()
        self.register_buffer(
            name = "weight",
            tensor = weights
        )
        # padding dimensions
        self.padding = [kernel_size//2] * 4

    def forward(self, input: Tensor):
        # subtract max for numerical stability of softmax
        exp_input = torch.exp((input - input.max()) / self.temperature)
        # perform local sums to get denominator
        padded = torch.nn.functional.pad(exp_input, self.padding, "reflect")
        sums = torch.nn.functional.conv2d(padded, self.weight,
            groups=self.in_channels,
            padding="valid")
        return exp_input / sums
        


class SoftmaxConvNet(TrainingWrapper):
    """
    """
    def __init__(self, channels_per_layer: List[int], kernel_sizes: List[int],
            softmax_sizes: List[int],
            in_channels: int, out_units: int, loss_fun: Callable, metric: Callable,
            version: Literal["ccsl", "cscl", "clcs"],
            lr: float=0.01, weight_decay: float=1e-2, momentum: float=0.9,
            gamma: float=0.99):
        # generate layers
        layers = self.build_layers(in_channels, out_units, channels_per_layer,
            kernel_sizes, softmax_sizes, version)
        # set up network
        super().__init__(net=layers, loss_fun=loss_fun, metric=metric, lr=lr,
            weight_decay=weight_decay, momentum=momentum, gamma=gamma)
        

    def build_layers(self, in_channels, out_units, channels_per_layer,
            kernel_sizes, softmax_sizes, version):
        layers = torch.nn.Sequential()
        for c, k, s in zip(channels_per_layer, kernel_sizes, softmax_sizes):
            # Three alternative ways of ordering the operations.
            # Performance can be subtly different, so leaving all three versions
            # in for now.
            #
            # In all three cases, important elements are:
            # - using "replicate" padding in the spatial convolutions
            # - disabling bias
            # - without GroupNorm before spatial softmax, loss does not decrease
            # In contrast, the presence / parametrization of the LRN layer does
            # not seem to have a big effect
            if version == "ccsl":
                # Version 1: both convolutions, the spatial softmax, then lrn
                block = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, c, 1, bias=False),
                    torch.nn.Conv2d(c, c, k, groups=c, bias=False,
                        padding="same", padding_mode="replicate"),
                    torch.nn.GroupNorm(c, c, affine=False),
                    SpatialSoftmax2d(c, s, sigma=s/3, temperature=0.2),
                    torch.nn.LocalResponseNorm(c, alpha=1.0, k=1.0, beta=0.75)
                )
            elif version == "cscl":
                # Version 2: First spatial conv and spatial softmax, then feature conv and lrn
                block = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, in_channels, k, padding="same",
                        groups=in_channels, bias=False, padding_mode="replicate"),
                    torch.nn.GroupNorm(in_channels, in_channels, affine=False),
                    SpatialSoftmax2d(in_channels, s, sigma=s/3, temperature=0.2),
                    torch.nn.Conv2d(in_channels, c, 1, padding="same", bias=False),
                    torch.nn.LocalResponseNorm(c, alpha=1.0, k=1.0, beta=0.75)
                )
            elif version == "clcs":
                # Version 3: First feature conv and lrn, then spatial conv and spatial softmax
                block = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, c, 1, padding="same", bias=False),
                    torch.nn.LocalResponseNorm(c, alpha=1.0, k=1.0, beta=0.75),
                    torch.nn.Conv2d(c, c, k, padding="same", padding_mode="replicate",
                        groups=c, bias=False),
                    torch.nn.GroupNorm(c, c, affine=False),
                    SpatialSoftmax2d(c, s, sigma=s/3, temperature=0.2),
                )
            layers.append(block)
            in_channels = c
        #layers.append(torch.nn.AdaptiveAvgPool2d(15))
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.LazyLinear(out_units))
        return layers
    
    def get_layers_of_interest(self):
        blocks = {f"block{i}": i for i in range(1, len(self.layers)-1)}
        blocks["output"] = len(self.layers)
        return blocks