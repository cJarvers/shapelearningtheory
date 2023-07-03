from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torchmetrics
from typing import Any, List, Optional, Tuple
import pytorch_lightning as pl

class AutoEncoder(pl.LightningModule):
    """
    Trains several layers via an autoencoder objective. In addition,
    trains a linear layer for classification from the hidden representation.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int],
            representation_dim: int, num_classes: int,
            lr: float=0.01, weight_decay: float=1e-4, momentum: float=0.9,
            gamma: float=0.99):
        super().__init__()
        self.save_hyperparameters()
        # initial batchnorm; the normed input is reconstructed
        self.norm = torch.nn.LayerNorm(input_dim, elementwise_affine=False, eps=0.01)
        # generate encoder layers
        self.encoder = torch.nn.Sequential()
        previous_dim = input_dim
        for d in hidden_dims:
            self.encoder.append(torch.nn.Linear(previous_dim, d))
            self.encoder.append(torch.nn.LayerNorm(d, elementwise_affine=False))
            previous_dim = d
            self.encoder.append(torch.nn.GELU())
        self.encoder.append(torch.nn.Linear(previous_dim, representation_dim))
        # generate classifier layer
        self.classifier = torch.nn.Linear(representation_dim, num_classes)
        # generate decoder layers
        self.decoder = torch.nn.Sequential()
        previous_dim = representation_dim
        for d in reversed(hidden_dims):
            self.decoder.append(torch.nn.Linear(previous_dim, d))
            self.decoder.append(torch.nn.LayerNorm(d, elementwise_affine=False))
            previous_dim = d
            self.decoder.append(torch.nn.GELU())
        self.decoder.append(torch.nn.Linear(previous_dim, input_dim))
        # set up loss functions
        self.reconstruction_loss = torch.nn.MSELoss()
        self.classification_loss = torch.nn.functional.cross_entropy
        self.classification_metric = torchmetrics.Accuracy('multiclass', num_classes=num_classes)
        # use manual optimization to use the two optimizers
        self.automatic_optimization = False

    def configure_optimizers(self) -> Any:
        autoencode_optimizer = torch.optim.SGD([
                {"params": self.encoder.parameters()},
                {"params": self.decoder.parameters()}
            ], lr=self.hparams.lr, momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay, nesterov=True)
        classify_optimizer = torch.optim.SGD(self.classifier.parameters(),
            lr = self.hparams.lr, momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay, nesterov=True)
        autoencode_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            autoencode_optimizer, gamma=self.hparams.gamma
        )
        classify_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            classify_optimizer, gamma=self.hparams.gamma
        )
        return [autoencode_optimizer, classify_optimizer], [autoencode_scheduler, classify_scheduler]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        representation = self.encoder(x)
        prediction = self.classifier(representation.detach())
        reconstruction = self.decoder(representation)
        return reconstruction, prediction
    
    def on_train_epoch_end(self) -> None:
        autoencode_scheduler, classify_scheduler = self.lr_schedulers()
        autoencode_scheduler.step()
        classify_scheduler.step()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x_flat = torch.flatten(x, start_dim=1)
        x_norm = self.norm(x_flat)
        autoencode_optimizer, classify_optimizer = self.optimizers()
        # apply network and calculate losses
        reconstruction, prediction = self.forward(x_norm)
        reconstruction_loss = self.reconstruction_loss(reconstruction, x_norm)
        classification_loss = self.classification_loss(prediction, y)
        # apply optimizers
        autoencode_optimizer.zero_grad()
        self.manual_backward(reconstruction_loss)
        autoencode_optimizer.step()
        classify_optimizer.zero_grad()
        self.manual_backward(classification_loss)
        classify_optimizer.step()
        # log metrics
        with torch.no_grad():
            self.log("reconstruction_loss_train", reconstruction_loss, prog_bar=True)
            self.log("train_loss", classification_loss, prog_bar=True)
            accuracy = self.classification_metric(prediction, y)
            self.log("train_metric", accuracy)
            if batch_idx == 0:
                tensorboard = self.logger.experiment
                tensorboard.add_image("input", x[0], global_step=self.global_step, dataformats="CHW")
                tensorboard.add_image("input_normed", x_norm.reshape(x.size())[0], global_step=self.global_step, dataformats="CHW")
                tensorboard.add_image("reconstruction", reconstruction.reshape(x.size())[0], global_step=self.global_step, dataformats="CHW")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = torch.flatten(x, start_dim=1)
        x_norm = self.norm(x)
        reconstruction, prediction = self.forward(x_norm)
        reconstruction_loss = self.reconstruction_loss(reconstruction, x_norm)
        self.log("reconstruction_loss_val", reconstruction_loss)
        classification_loss = self.classification_loss(prediction, y)
        self.log("val_loss", classification_loss)
        accuracy = self.classification_metric(prediction, y)
        self.log("val_metric", accuracy)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = torch.flatten(x, start_dim=1)
        x_norm = self.norm(x)
        reconstruction, prediction = self.forward(x_norm)
        reconstruction_loss = self.reconstruction_loss(reconstruction, x_norm)
        self.log("reconstruction_loss_test", reconstruction_loss)
        classification_loss = self.classification_loss(prediction, y)
        self.log("test_loss", classification_loss)
        accuracy = self.classification_metric(prediction, y)
        self.log("test_metric", accuracy)
