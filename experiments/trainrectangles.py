import torch
from torchmetrics import Accuracy
import pytorch_lightning as pl
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.datasets import RectangleDataModule, SquaresDataModule
from shapelearningtheory.networks import ShallowLinear, DeepLinear, MLP, AutoEncoder, SimpleConvNet
from shapelearningtheory.colors import Grey, GreySingleChannel, RedXORBlue, NotRedXORBlue, RandomGrey, RandomGreySingleChannel
from shapelearningtheory.textures import HorizontalGrating, VerticalGrating

# hyper-parameters for the task
use_color = False
if use_color:
    pattern1 = RedXORBlue
    pattern2 = NotRedXORBlue
    background = RandomGrey
    nopattern = Grey
    channels = 3
else:
    pattern1 = VerticalGrating
    pattern2 = HorizontalGrating
    background = RandomGreySingleChannel
    nopattern = GreySingleChannel
    channels = 1
imgsize = 36
lengths=[4, 6, 9, 12]
widths=[3, 4, 6, 9]
oversample = 5
# hyper-parameters for the networks
num_layers = 3
num_hidden = 1000
# hyper-parameters for training
epochs = 500
batchsize = 128

# get data:
# training dataset
traindata = RectangleDataModule(imgsize, imgsize, lengths, widths, pattern1=pattern1, pattern2=pattern2, background_pattern=background, oversampling_factor=oversample, batch_size=batchsize)
#
# test datasets - parametrized slightly differently to test generalization
test_sets = {
    "traindata": traindata,
    "color only": SquaresDataModule(imgsize, imgsize, widths, pattern1=pattern1, pattern2=pattern2, background_pattern=background, batch_size=batchsize), # correct color, but squares instead of rectangles (cannot classify by shape)
    "shape only": RectangleDataModule(imgsize, imgsize, lengths, widths, pattern1=nopattern, pattern2=nopattern, background_pattern=background, batch_size=batchsize), # same rectangles but no color
    "conflict": RectangleDataModule(imgsize, imgsize, lengths, widths, pattern1=pattern2, pattern2=pattern1, background_pattern=background, batch_size=batchsize) # same rectangles, incorrect color
}

# define models
shallow_model = ShallowLinear(imgsize * imgsize * channels, 2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))
deep_model = DeepLinear(num_inputs=imgsize * imgsize * channels, num_hidden=num_hidden, num_layers=num_layers,
    num_outputs=2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))
mlp_model = MLP(num_inputs=imgsize * imgsize * channels, num_hidden=num_hidden, num_layers=num_layers,
    num_outputs=2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))
simpleconv_model = SimpleConvNet(channels_per_layer=[16, 32, 64], kernel_sizes=[3,3,3],
    in_channels=channels, out_units=2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))
autoencoder = AutoEncoder(input_dim=imgsize * imgsize * channels, hidden_dims=[num_hidden] * num_layers,
    representation_dim=500, num_classes=2)
models = {
    "shallow_linear": shallow_model,
    "deep_linear": deep_model,
    "mlp": mlp_model,
    "conv": simpleconv_model,
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