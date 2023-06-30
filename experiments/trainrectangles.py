import torch
from torchmetrics import Accuracy
import pytorch_lightning as pl
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.rectangledataset import RectangleDataModule
from shapelearningtheory.squaredataset import SquaresDataModule
from shapelearningtheory.linearnetworks import ShallowLinear, DeepLinear
from shapelearningtheory.mlp import MLP
from shapelearningtheory.autoencoder import AutoEncoder
from shapelearningtheory.convnet import SimpleConvNet
from shapelearningtheory.colors import Grey, RedXORBlue, NotRedXORBlue, RandomRed, RandomBlue

# hyper-parameters for the task
pattern1 = RedXORBlue # RandomRed # 
pattern2 = NotRedXORBlue # RandomBlue # 
imgsize = 36
lengths=[4, 6, 9, 12]
widths=[3, 4, 6, 9]
oversample = 3
# hyper-parameters for the networks
num_layers = 3
num_hidden = 1000
# hyper-parameters for training
epochs = 100

# get data:
# training dataset
traindata = RectangleDataModule(imgsize, imgsize, lengths, widths, pattern1=pattern1, pattern2=pattern2, oversampling_factor=oversample)
#
# test datasets - parametrized slightly differently to test generalization
test_sets = {
    "traindata": traindata,
    "color only": SquaresDataModule(imgsize, imgsize, widths, pattern1=pattern1, pattern2=pattern2), # correct color, but squares instead of rectangles (cannot classify by shape)
    "shape only": RectangleDataModule(imgsize, imgsize, lengths, widths, pattern1=Grey, pattern2=Grey), # same rectangles but no color
    "conflict": RectangleDataModule(imgsize, imgsize, lengths, widths, pattern1=pattern2, pattern2=pattern1) # same rectangles, incorrect color
}

# define models
shallow_model = ShallowLinear(imgsize * imgsize * 3, 2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))
deep_model = DeepLinear(num_inputs=imgsize * imgsize * 3, num_hidden=num_hidden, num_layers=num_layers,
    num_outputs=2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))
mlp_model = MLP(num_inputs=imgsize * imgsize * 3, num_hidden=num_hidden, num_layers=num_layers,
    num_outputs=2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))
simpleconv_model = SimpleConvNet(channels_per_layer=[16, 32, 64], kernel_sizes=[3,3,3],
    in_channels=3, out_units=2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))
autoencoder = AutoEncoder(input_dim=imgsize * imgsize * 3, hidden_dims=[num_hidden, num_hidden],
    representation_dim=100, num_classes=2)
models = {
    "shallow_linear": shallow_model,
    "deep_linear": deep_model,
    "mlp": mlp_model,
    "conv": simpleconv_model,
    "autoencoder": autoencoder
}

# initialize trainers
trainers = {name: pl.Trainer(max_epochs=epochs) for name in models.keys()}

# train
for name, model in models.items():
    trainers[name].fit(model, traindata)


# test generalization with shorter and longer lines
test_results = {}
for name, model in models.items():
    results = {}
    for testname, testset in test_sets.items():
        results[testname] = trainers[name].test(model, testset, verbose=False)
    test_results[name] = results
# Print test results as table
cellwidth = 15
print("Test results:")
print("| " + "Network".ljust(cellwidth-1), end="|")
for testname in test_sets.keys():
    print(" " + testname.ljust(cellwidth-1), end="|")
print("\n" + ("|" + "-" * cellwidth) * (len(test_sets)+1) + "|")
for model, results in test_results.items():
    print("| " + model.ljust(cellwidth-1), end="")
    for testname, result in results.items():
        r = round(result[0]["test_metric"], ndigits=3)
        print("|" + f"{r}".rjust(cellwidth-2), end="  ")
    print("|")