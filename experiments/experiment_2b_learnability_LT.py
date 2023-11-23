from matplotlib import pyplot as plt
import seaborn as sns
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.datasets import make_LT_wrong_color, make_LT_shapeonly
from shapelearningtheory.networks import make_mlp_small, make_convnet_small, make_rconvnet_small, \
    make_softmaxconv_small, make_ViT_small, make_AE_small
from helpers import print_table, train_and_validate, unpack_results

# hyper-parameters for training
epochs = 100

# get data:
# training dataset
traindata = make_LT_shapeonly()
#
# test datasets - parametrized slightly differently to test generalization
test_sets = {
    "traindata": traindata,
    "validation": make_LT_shapeonly(),
    "conflict": make_LT_wrong_color()
}

# hyperparameters from dataset
traindata.prepare_data()
imgheight = traindata.dataset.imgheight
imgwidth = traindata.dataset.imgwidth
channels = 3
classes = 2

# define models
models = {
    "mlp": lambda: make_mlp_small(num_inputs=imgheight * imgwidth * channels, num_outputs=classes),
    "conv": lambda: make_convnet_small(channels=channels, classes=classes),
    "rconv": lambda: make_rconvnet_small(channels=channels, classes=classes),
    "softmaxconv": lambda: make_softmaxconv_small(channels=channels, classes=classes),
    "ViT": lambda: make_ViT_small(imgsize=imgheight, classes=classes),
    "autoencoder": lambda: make_AE_small(num_inputs=imgheight * imgwidth * channels, classes=classes)
}

# train and test
test_results = {}
for name, model in models.items():
    test_results[name] = train_and_validate(
        model, traindata, test_sets
    )

# Print test results as table
print_table(test_sets.keys(), test_results, cellwidth=15)
# Plot results as bar plot
df = unpack_results(test_results)
fig, ax = plt.subplots()
sns.barplot(df, x="dataset", y="metric", hue="model", ax=ax)
fig.suptitle("Accuracy on shape-only LvT")
plt.savefig("figures/exp2b_barplot.png")