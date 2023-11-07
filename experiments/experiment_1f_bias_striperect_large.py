from matplotlib import pyplot as plt
import seaborn as sns
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.datasets import make_rectangles_texture_large, make_rectangles_textureonly_large, \
    make_rectangles_shapeonly_large, make_rectangles_wrong_texture_large
from shapelearningtheory.networks import make_resnet50, make_vit_b_16
from helpers import print_table, train_and_validate, unpack_results

# hyper-parameters for training
epochs = 100
batch_size = 8

# get data:
# training dataset
traindata = make_rectangles_texture_large(batchsize=batch_size)
#
# test datasets - parametrized slightly differently to test generalization
test_sets = {
    "traindata": traindata,
    "validation": make_rectangles_texture_large(batchsize=batch_size),
    "pattern only": make_rectangles_textureonly_large(batchsize=batch_size),
    "shape only": make_rectangles_shapeonly_large(batchsize=batch_size),
    "conflict": make_rectangles_wrong_texture_large(batchsize=batch_size)
}

# hyperparameters from dataset
traindata.prepare_data()
imgheight = traindata.dataset.imgheight
imgwidth = traindata.dataset.imgwidth
channels = 3
classes = 2

# define models
models = {
    "resnet": lambda: make_resnet50(classes),
    "vit_b_16": lambda: make_vit_b_16(imgheight, classes)
}

# train and test
test_results = {}
for name, model in models.items():
    test_results[name] = train_and_validate(
        model, traindata, test_sets, epochs=20, repetitions=5
    )

# Print test results as table
print_table(test_sets.keys(), test_results, cellwidth=15)
# Plot results as bar plot
df = unpack_results(test_results)
fig, ax = plt.subplots()
sns.barplot(df, x="dataset", y="metric", hue="model", ax=ax)
fig.suptitle("Accuracy on striped rectangles")
plt.savefig("figures/exp1f_barplot.png")