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
