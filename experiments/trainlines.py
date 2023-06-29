# This script trains several network variants to classify horizontal vs. vertical lines.
# Notably, the lines can be distinguished by their color as well as their orientation.
# The key question is whether the network relies on orientation (shape) or color to classify the lines,
# similar to the experiments in Malhotra et al. (2022) PLOS Computational Biology 15(5), e1009572.
#
# To test which features the networks use to classify, we test them on modified datasets that
# change either the shape or color information:
# - short: shorter lines in the correct color (check that the network did not overfit to line length)
# - long: longer lines in the correct color (check that the network did not overfit to line length)
# - coloronly: squares in the correct color; if the network classifies these correctly, it uses color information
# - grey: lines without color; if the network classifies these correctly, it must use shape information
# - conflict: lines with opposite color mapping; if network classifies these correctly, it prefers shape to color
#
# We train / test the following networks:
# - shallow_linear: a one-layer linear network (currently disabled since color/shape classes are not linearly seperable)
# - deep_linear: a multi-layer linear network (currently disabled since color/shape classes are not linearly seperable)
# - mlp: a multi-layer perceptron; number of layers is controlled by variable num_layers
# - conv: a simple convolutional network with three convolutional layers, global average pooling, and one fc layer
#
#
# Observations:
#
# - The linear networks perform at chance level since the classes are not linearly seperable.
# - Whether the MLP prefers shape or color depends on the types of colors used.
#    - If the two color classes are linearly seperable (e.g., RandomRed and RandomBlue) then the
#      network relies on color. It performs at 100% for color-only stimuli and at 0% for conflict
#      stimuli.
#    - If the two color classes are not linearly seperable (e.g., RedXORBlue and NotRedXORBlue),
#      even though the network can learn to distinguish them (when trained on squares), it learns
#      a mix of both features (with a preference for shape?). It performs at ~70% for color-only
#      stimuli and at ~80% for conflict stimuli.

import torch
from torchmetrics import Accuracy
import pytorch_lightning as pl
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.linedataset import LineDataModule
from shapelearningtheory.squaredataset import SquaresDataModule
from shapelearningtheory.linearnetworks import ShallowLinear, DeepLinear
from shapelearningtheory.mlp import MLP
from shapelearningtheory.autoencoder import AutoEncoder
from shapelearningtheory.convnet import SimpleConvNet
from shapelearningtheory.colors import Grey, RedXORBlue, NotRedXORBlue, RandomRed, RandomBlue

# hyper-parameters for the task
color1 = RedXORBlue # RandomRed # 
color2 = NotRedXORBlue # RandomBlue # 
imgsize = 15
short = 3
long = 13
# hyper-parameters for the networks
num_layers = 3
num_hidden = 1000
# hyper-parameters for training
epochs = 100

# get data:
# training dataset
traindata = LineDataModule(imgsize, imgsize, range(short+2, long-2), horizontalcolor=color1, verticalcolor=color2)
# alternative version: use squares to check that networks really can learn to distinguish the colors
#traindata = SquaresDataModule(imgsize, imgsize, range(short, (short+long)//2), color1=color1, color2=color2) 
#
# test datasets - parametrized slightly differently to test generalization
test_sets = {
    #"short": LineDataModule(imgsize, imgsize, [short], horizontalcolor=color1, verticalcolor=color2), # shorter lines, correct color
    #"long": LineDataModule(imgsize, imgsize, [long], horizontalcolor=color1, verticalcolor=color2), # longer lines, correct color
    "traindata": traindata,
    "color only": SquaresDataModule(imgsize, imgsize, range(short+2, long-2), pattern1=color1, pattern2=color2), # correct color, but squares instead of lines (cannot classify by shape)
    "shape only": LineDataModule(imgsize, imgsize, range(short+2, long-2), horizontalcolor=Grey, verticalcolor=Grey), # medium length lines, no color
    "conflict": LineDataModule(imgsize, imgsize, range(short+2, long-2), horizontalcolor=color2, verticalcolor=color1) # medium length lines, incorrect color
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