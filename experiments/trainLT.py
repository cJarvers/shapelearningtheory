import torch
from torchmetrics import Accuracy
import pytorch_lightning as pl
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.datasets import LTDataModule, SquaresDataModule
from shapelearningtheory.networks import MLP, AutoEncoder, SimpleConvNet, SoftmaxConvNet
from shapelearningtheory.colors import Grey, RedXORBlue, NotRedXORBlue, RandomGrey
from shapelearningtheory.textures import HorizontalGrating, VerticalGrating

# hyper-parameters for the task
patternL = RedXORBlue
patternT = NotRedXORBlue
background = RandomGrey
nopattern = Grey
channels = 3
imgsize = 18
heights=range(8, 12)
strengths=range(1, 3)
# hyper-parameters for the networks
num_layers = 3
num_hidden = 1000
# hyper-parameters for training
epochs = 100
batchsize = 128

# get data:
# training dataset
traindata = LTDataModule(imgsize, imgsize, heights=heights, widths=heights, strengths=strengths,
    patternL=patternL, patternT=patternT, background_pattern=background, batch_size=batchsize)
# validation data uses same parameters, but due to randomization the images will be slightly different
valdata = LTDataModule(imgsize, imgsize, heights=heights, widths=heights, strengths=strengths,
    patternL=patternL, patternT=patternT, background_pattern=background, batch_size=batchsize)
#
# test datasets - parametrized slightly differently to test generalization
test_sets = {
    "traindata": traindata,
    "validation": valdata,
    "color only": SquaresDataModule(imgsize, imgsize, heights, pattern1=patternL, # correct color, squares (no clear shape)
        pattern2=patternT, background_pattern=background, batch_size=batchsize, oversampling_factor=5),
    "shape only": LTDataModule(imgsize, imgsize, heights=heights, widths=heights, # same shapes but no color / texture
        strengths=strengths, patternL=nopattern, patternT=nopattern,
        background_pattern=background, batch_size=batchsize),
    "conflict": LTDataModule(imgsize, imgsize, heights=heights, widths=heights, # same shapes, colors/textures swapped
        strengths=strengths, patternL=patternT, patternT=patternL,
        background_pattern=background, batch_size=batchsize)
}

# define models
mlp_model = MLP(num_inputs=imgsize * imgsize * channels, num_hidden=num_hidden, num_layers=num_layers,
    num_outputs=2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))
simpleconv_model = SimpleConvNet(channels_per_layer=[16, 32, 64], kernel_sizes=[3,3,3],
    in_channels=channels, out_units=2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))
softmaxconv_model = SoftmaxConvNet(
    channels_per_layer=[16, 32, 64],
    kernel_sizes=[5,5,5],
    softmax_sizes=[9,9,9],
    version="cscl",
    in_channels=channels, out_units=2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))
autoencoder = AutoEncoder(input_dim=imgsize * imgsize * channels, hidden_dims=[num_hidden] * num_layers,
    representation_dim=500, num_classes=2)
models = {
    "mlp": mlp_model,
    "conv": simpleconv_model,
    "softmaxconv": softmaxconv_model,
    "autoencoder": autoencoder
}

# initialize trainers
trainers = {name: pl.Trainer(max_epochs=epochs, accelerator="gpu") for name in models.keys()}

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