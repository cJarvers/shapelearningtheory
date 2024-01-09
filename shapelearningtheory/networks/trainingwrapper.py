import torch
import pytorch_lightning as pl
from typing import Any, Callable, Tuple

class TrainingWrapper(pl.LightningModule):
    def __init__(self, net: torch.nn.Module,
            loss_fun: Callable, metric: Callable,
            lr: float=0.01, weight_decay: float=1e-4,
            momentum: float=0.9, gamma: float=0.99):
        super().__init__()
        self.save_hyperparameters(ignore=["net", "loss_fun", "metric"])
        self.layers = net
        self.loss_fun = loss_fun
        self.metric = metric

    def __getitem__(self, idx):
        return self.layers[idx]

    def compute_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        p = self.forward(x)
        loss = self.loss_fun(p, y)
        with torch.no_grad():
            metric = self.metric(p, y)
        return loss, metric

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

    