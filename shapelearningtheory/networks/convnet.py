import torch
from typing import List, Tuple, Callable, Any
import pytorch_lightning as pl

class SimpleConvNet(pl.LightningModule):

    def __init__(self, channels_per_layer: List[int], kernel_sizes: List[int],
            in_channels: int, out_units: int, loss_fun: Callable, metric: Callable,
            lr: float=0.01, weight_decay: float=1e-4, momentum: float=0.9,
            gamma: float=0.99):
        super().__init__()
        self.save_hyperparameters(ignore=["metric", "loss_fun"])
        self.loss_fun = loss_fun
        self.metric = metric
        # generate layers
        self.build_layers(in_channels, channels_per_layer, kernel_sizes,
            out_units)

    def build_layers(self, in_channels, channels_per_layer, kernel_sizes,
            out_units):
        self.layers = torch.nn.Sequential()
        for c, k in zip(channels_per_layer, kernel_sizes):
            block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, c, k, padding="same"),
                torch.nn.GroupNorm(c, c, affine=False),
                torch.nn.GELU())
            self.layers.append(block)
            in_channels = c
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
        optimizer = torch.optim.SGD(self.layers.parameters(), lr=self.hparams.lr,
            momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay,
            nesterov=True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
            gamma=self.hparams.gamma)
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
        self.num_steps = num_steps

    def build_layers(self, in_channels, channels_per_layer, kernel_sizes,
            out_units):
        # Generate self.layers as a Sequential like in SimpleConvNet.
        # These are the forward layers (i.e., the filters used to
        # pass information up between the recurrent units).
        # Additionally, we generate a list self.lateral of layers
        # which perform the recurrent filtering over time.
        self.layers = torch.nn.Sequential()
        self.lateral = torch.nn.ModuleList()
        for c, k in zip(channels_per_layer, kernel_sizes):
            # generate forward layer
            block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, c, k, padding="same"),
                torch.nn.GroupNorm(c, c, affine=False),
                torch.nn.GELU())
            self.layers.append(block)
            # generate lateral filters
            lateral_block = torch.nn.Sequential(
                torch.nn.Conv2d(c, c, k, padding="same"),
                torch.nn.GroupNorm(c, c, affine=False),
                torch.nn.GELU())
            self.lateral.append(lateral_block)
            in_channels = c
        self.layers.append(torch.nn.Flatten())
        self.layers.append(torch.nn.LazyLinear(out_units))

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
