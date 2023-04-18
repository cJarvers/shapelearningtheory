import torch
from torchmetrics import Accuracy
import pytorch_lightning as pl
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.linedataset import LineDataModule
from shapelearningtheory.linearnetworks import ShallowLinear, DeepLinear

# get data
data = LineDataModule(15, 15, range(3, 11))

# define models
shallow_model = ShallowLinear(15 * 15, 2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))
deep_model = DeepLinear(num_inputs=15 * 15, num_hidden=1000, num_layers=3,
    num_outputs=2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))

# initialize trainers
shallow_trainer = pl.Trainer(max_epochs=100)
deep_trainer = pl.Trainer(max_epochs=100)

shallow_trainer.fit(shallow_model, data)
deep_trainer.fit(deep_model, data)