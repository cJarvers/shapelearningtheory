from matplotlib import pyplot as plt
import seaborn as sns
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.datasets import make_dataset
from helpers import print_table, train_and_validate, unpack_results, get_standard_networks

# hyper-parameters for training
epochs = 10
repetitions = 2
batch_size = 4

# get data:
# training dataset
traindata = make_dataset("LvT", "color", "large", "standard", batchsize=batch_size)
#
# test datasets - parametrized slightly differently to test generalization
test_sets = {
    "traindata": traindata,
    "validation": make_dataset("LvT", "color", "large", "standard", batchsize=batch_size),
    "pattern only": make_dataset("LvT", "color", "large", "patternonly", batchsize=batch_size),
    "shape only": make_dataset("LvT", "color", "large", "shapeonly", batchsize=batch_size),
    "conflict": make_dataset("LvT", "color", "large", "conflict", batchsize=batch_size)
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
        epochs=epochs, repetitions=repetitions
    )

# Print test results as table
print_table(test_sets.keys(), test_results, cellwidth=15)
# Plot results as bar plot
df = unpack_results(test_results)
fig, ax = plt.subplots()
sns.barplot(df, x="dataset", y="metric", hue="model", ax=ax)
fig.suptitle("Accuracy on color LvT large")
plt.savefig("figures/exp1g_barplot.png")