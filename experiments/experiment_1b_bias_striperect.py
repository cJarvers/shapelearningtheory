import pytorch_lightning as pl
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.datasets import make_rectangles_texture, make_rectangles_textureonly, \
    make_rectangles_shapeonly, make_rectangles_wrong_texture
from shapelearningtheory.networks import make_mlp_small, make_convnet_small, make_rconvnet_small, \
    make_softmaxconv_small, make_ViT_small, make_AE_small
from helpers import print_table, train_and_validate

# hyper-parameters for training
epochs = 100

# get data:
# training dataset
traindata = make_rectangles_texture()
#
# test datasets - parametrized slightly differently to test generalization
test_sets = {
    "traindata": traindata,
    "validation": make_rectangles_texture(),
    "pattern only": make_rectangles_textureonly(),
    "shape only": make_rectangles_shapeonly(),
    "conflict": make_rectangles_wrong_texture()
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

# train and test
test_results = {}
for name, model in models.items():
    test_results[name] = train_and_validate(
        model, traindata, test_sets, repetitions=2, epochs=10
    )

# Print test results as table
print_table(test_sets.keys(), test_results, cellwidth=15)
