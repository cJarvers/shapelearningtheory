import torch
from torchmetrics import Accuracy
import pytorch_lightning as pl
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.datasets import make_rectangles_color, make_rectangles_texture, make_rectangles_coloronly, \
    make_rectangles_textureonly, make_rectangles_shapeonly, make_rectangles_wrong_color, make_rectangles_wrong_texture
from shapelearningtheory.networks import make_mlp_small, make_convnet_small, make_rconvnet_small, \
    make_softmaxconv_small, make_ViT_small, make_AE_small

# hyper-parameters for the task
use_color = False
# hyper-parameters for training
epochs = 100

# get data:
# training dataset
if use_color:
    traindata = make_rectangles_color()
    # validation data uses same parameters, but due to randomization the images will be slightly different
    valdata = make_rectangles_color()
else:
    traindata = make_rectangles_texture()
    valdata = make_rectangles_texture()
#
# test datasets - parametrized slightly differently to test generalization
test_sets = {
    "traindata": traindata,
    "validation": valdata,
    "pattern only": make_rectangles_coloronly() if use_color else make_rectangles_textureonly(),
    "shape only": make_rectangles_shapeonly(),
    "conflict": make_rectangles_wrong_color() if use_color else make_rectangles_wrong_texture()
}

# hyperparameters from dataset
traindata.prepare_data()
imgheight = traindata.dataset.imgheight
imgwidth = traindata.dataset.imgwidth
channels = 3
classes = 2

# define models
models = {
    "mlp": make_mlp_small(num_inputs=imgheight * imgwidth * channels, num_outputs=classes),
    "conv": make_convnet_small(channels=channels, classes=classes),
    "rconv": make_rconvnet_small(channels=channels, classes=classes),
    "softmaxconv": make_softmaxconv_small(channels=channels, classes=classes),
    "ViT": make_ViT_small(imgsize=imgheight, classes=classes),
    "autoencoder": make_AE_small(num_inputs=imgheight * imgwidth * channels, classes=classes)
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