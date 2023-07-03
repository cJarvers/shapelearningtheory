import torch
from typing import Tuple, Callable, Any
import pytorch_lightning as pl

class MLP(pl.LightningModule):
    "Non-linear multi-layer perceptron"
    
    def __init__(self, num_inputs: int, num_hidden: int, num_layers: int,
            num_outputs: int, loss_fun: Callable, metric: Callable,
            lr: float=0.01, weight_decay: float=1e-4, momentum: float=0.9,
            gamma: float=0.99):
        super().__init__()
        self.loss_fun = loss_fun
        self.metric = metric
        self.save_hyperparameters()
        # generate layers:
        self.layers = torch.nn.Sequential()
        self.layers.append(torch.nn.Linear(num_inputs, num_hidden))
        self.layers.append(torch.nn.LayerNorm(num_hidden, elementwise_affine=False))
        self.layers.append(torch.nn.GELU())
        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(num_hidden, num_hidden))
            self.layers.append(torch.nn.LayerNorm(num_hidden, elementwise_affine=False))
            self.layers.append(torch.nn.GELU())
        self.layers.append(torch.nn.Linear(num_hidden, num_outputs))

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
        x = torch.flatten(x, start_dim=1)
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