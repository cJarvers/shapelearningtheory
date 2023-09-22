import torch
import torchvision
from typing import Tuple, Callable, Any
import pytorch_lightning as pl

class VisionTransformer(pl.LightningModule):
    "Wrapper around torchvisions VisionTransformer"
    
    def __init__(self, image_size: int, patch_size: int, num_layers:int,
            num_heads: int, hidden_dim: int, mlp_dim: int, num_classes: int,
            loss_fun: Callable, metric: Callable,
            lr: float=0.01, weight_decay: float=1e-4, momentum: float=0.9,
            gamma: float=0.99):
        super().__init__()
        self.loss_fun = loss_fun
        self.metric = metric
        self.save_hyperparameters() #ignore=["loss_fun", "metric"])
        # generate layers:
        self.layers = torchvision.models.VisionTransformer(
            image_size=image_size, patch_size=patch_size, num_layers=num_layers,
            num_heads=num_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim,
            num_classes=num_classes
        )

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
    
    def layer_activations(self, x):
        outputs = {'image': x}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, torch.nn.Linear):
                outputs["layer {}".format(i)] = x
        return outputs
    
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