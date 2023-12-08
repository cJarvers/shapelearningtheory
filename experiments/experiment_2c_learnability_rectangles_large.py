from matplotlib import pyplot as plt
import seaborn as sns
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.datasets import make_rectangles_color_large, make_rectangles_shapeonly_large, make_rectangles_texture_large
from shapelearningtheory.networks import make_resnet50, make_vit_b_16
from helpers import print_table, train_and_validate, unpack_results

# hyper-parameters for training
epochs = 20

# get data:
# training dataset
traindata = make_rectangles_shapeonly_large()
#
# test datasets - parametrized slightly differently to test generalization
test_sets = {
    "traindata": traindata,
    "validation": make_rectangles_shapeonly_large(),
    "with color": make_rectangles_color_large(),
    "with texture": make_rectangles_texture_large()
}

# hyperparameters from dataset
traindata.prepare_data()
imgheight = traindata.train.imgheight
imgwidth = traindata.train.imgwidth
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
        model, traindata, test_sets,
        epochs=epochs
    )

# Print test results as table
print_table(test_sets.keys(), test_results, cellwidth=15)
# Plot results as bar plot
df = unpack_results(test_results)
fig, ax = plt.subplots()
sns.barplot(df, x="dataset", y="metric", hue="model", ax=ax)
fig.suptitle("Accuracy on shape-only rectangles")
plt.savefig("figures/exp2c_barplot.png")