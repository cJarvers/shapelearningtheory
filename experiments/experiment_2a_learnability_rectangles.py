from matplotlib import pyplot as plt
import seaborn as sns
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.datasets import make_rectangles_color, make_rectangles_shapeonly, make_rectangles_texture
from helpers import print_table, train_and_validate, unpack_results, get_basic_networks

# hyper-parameters for training
epochs = 100
repetitions = 5

# get data:
# training dataset
traindata = make_rectangles_shapeonly()
#
# test datasets - parametrized slightly differently to test generalization
test_sets = {
    "traindata": traindata,
    "validation": make_rectangles_shapeonly(),
    "with color": make_rectangles_color(),
    "with texture": make_rectangles_texture()
}

# hyperparameters from dataset
traindata.prepare_data()
imagesize = traindata.train.imgheight
channels = 3
classes = 2

# define models
models = get_basic_networks(classes, imagesize, channels)

# train and test
test_results = {}
for name, model in models.items():
    test_results[name] = train_and_validate(
        model, traindata, test_sets,
        epochs=epochs, repetitions=repetitions
    )

# Print test results as table
print_table(test_sets.keys(), test_results, cellwidth=15)
# Plot results as bar plot
df = unpack_results(test_results)
fig, ax = plt.subplots()
sns.barplot(df, x="dataset", y="metric", hue="model", ax=ax)
fig.suptitle("Accuracy on shape-only rectangles")
plt.savefig("figures/exp2a_barplot.png")