import torch
import torchvision
from torch import Tensor
import pytorch_lightning as pl
from typing import Any, Callable, List, Literal, Tuple

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
        


class SoftmaxConvNet(pl.LightningModule):
    """
    """
    def __init__(self, channels_per_layer: List[int], kernel_sizes: List[int],
            softmax_sizes: List[int],
            in_channels: int, out_units: int, loss_fun: Callable, metric: Callable,
            version: Literal["ccsl", "cscl", "clcs"],
            lr: float=0.01, weight_decay: float=1e-2, momentum: float=0.9,
            gamma: float=0.99):
        super().__init__()
        # store and log hyperparameters
        self.save_hyperparameters(ignore=["loss_fun", "metric"])
        self.loss_fun = loss_fun
        self.metric = metric
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.gamma = gamma
        # generate layers
        self.layers = torch.nn.Sequential()
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
            self.layers.append(block)
            in_channels = c
        #self.layers.append(torch.nn.AdaptiveAvgPool2d(15))
        self.layers.append(torch.nn.Flatten())
        self.layers.append(torch.nn.LazyLinear(out_units))

    def compute_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        p = self.forward(x)
        loss = self.loss_fun(p, y)
        with torch.no_grad():
            metric = self.metric(p, y)
        return loss, metric
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.SGD(self.layers.parameters(), lr=self.lr,
            momentum=self.momentum, weight_decay=self.weight_decay,
            nesterov=True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
            gamma=self.gamma)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.layers(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int) -> torch.Tensor:
        loss, metric = self.compute_loss(batch)
        self.log("train_loss", loss.detach())
        self.log("train_metric", metric)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int) -> torch.Tensor:
        loss, metric = self.compute_loss(batch)
        self.log("val_loss", loss)
        self.log("val_metric", metric)
        return loss

    @torch.no_grad()
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int) -> torch.Tensor:
        loss, metric = self.compute_loss(batch)
        self.log("test_loss", loss)
        self.log("test_metric", metric)
        return loss