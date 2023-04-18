import torch
from torchmetrics import Accuracy
import pytorch_lightning as pl
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.linedataset import LineDataModule
from shapelearningtheory.linearnetworks import ShallowLinear, DeepLinear
from shapelearningtheory.mlp import MLP

# get data
traindata = LineDataModule(15, 15, range(5, 11))
shorttest = LineDataModule(15, 15, [3])
longtest = LineDataModule(15, 15, [13])

# define models
shallow_model = ShallowLinear(15 * 15, 2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))
deep_model = DeepLinear(num_inputs=15 * 15, num_hidden=1000, num_layers=3,
    num_outputs=2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))
mlp_model = MLP(num_inputs=15 * 15, num_hidden=1000, num_layers=3,
    num_outputs=2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))

# initialize trainers
shallow_trainer = pl.Trainer(max_epochs=100)
deep_trainer = pl.Trainer(max_epochs=100)
mlp_trainer = pl.Trainer(max_epochs=100)

# train
shallow_trainer.fit(shallow_model, traindata)
deep_trainer.fit(deep_model, traindata)
mlp_trainer.fit(mlp_model, traindata)

# test generalization with shorter and longer lines
shallow_trainer.test(shallow_model, shorttest)
shallow_trainer.test(shallow_model, longtest)
deep_trainer.test(deep_model, shorttest)
deep_trainer.test(deep_model, longtest)
mlp_trainer.test(mlp_model, shorttest)
mlp_trainer.test(mlp_model, longtest)
