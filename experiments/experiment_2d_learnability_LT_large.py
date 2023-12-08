from matplotlib import pyplot as plt
import seaborn as sns
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.datasets import make_LT_color_large, make_LT_shapeonly_large, make_LT_texture_large
from helpers import print_table, train_and_validate, unpack_results, get_standard_networks

# hyper-parameters for training
epochs = 20

# get data:
# training dataset
traindata = make_LT_shapeonly_large()
#
# test datasets - parametrized slightly differently to test generalization
test_sets = {
    "traindata": traindata,
    "validation": make_LT_shapeonly_large(),
    "with color": make_LT_color_large(),
    "with texture": make_LT_texture_large()
}

# hyperparameters from dataset
traindata.prepare_data()
imagesize = traindata.train.imgheight
classes = 2

# define models
models = get_standard_networks(classes, imagesize)

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
fig.suptitle("Accuracy on shape-only LvT")
plt.savefig("figures/exp2d_barplot.png")