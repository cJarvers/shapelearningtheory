from matplotlib import pyplot as plt
import seaborn as sns
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.datasets import make_dataset
from helpers import format_table, train_and_validate, unpack_results, get_basic_networks

# hyper-parameters for training
epochs = 20
repetitions = 5
batch_size = 128

# get data:
# training dataset
traindata = make_dataset("rectangles", "stripes", "small", "standard", batchsize=batch_size)
#
# test datasets - parametrized slightly differently to test generalization
test_sets = {
    "traindata": traindata,
    "validation": make_dataset("rectangles", "stripes", "small", "standard", batchsize=batch_size),
    "pattern only": make_dataset("rectangles", "stripes", "small", "patternonly", batchsize=batch_size),
    "shape only": make_dataset("rectangles", "stripes", "small", "shapeonly", batchsize=batch_size),
    "conflict": make_dataset("rectangles", "stripes", "small", "conflict", batchsize=batch_size)
}

# hyperparameters from dataset
traindata.prepare_data()
imagesize = traindata.train.imgheight
channels = 3
classes = 2

# define models
models = get_basic_networks(classes, channels, imagesize)

# train and test
test_results = {}
for name, model in models.items():
    test_results[name] = train_and_validate(
        model, traindata, test_sets,
        epochs=epochs, repetitions=repetitions
    )

# Print test results as table
table = format_table(test_sets.keys(), test_results, cellwidth=15)
print(table)
with open("figures/exp1b_table.txt", "w") as f:
    f.write(table)
# Plot results as bar plot
df = unpack_results(test_results)
fig, ax = plt.subplots()
sns.barplot(df, x="dataset", y="metric", hue="model", ax=ax, legend=False)
fig.suptitle("Accuracy on striped rectangles")
plt.savefig("figures/exp1b_barplot.png")
