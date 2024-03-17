import argparse
from matplotlib import pyplot as plt
import pandas as pd
import pytorch_lightning as L
import seaborn as sns
import sys
from statsmodels.stats.descriptivestats import sign_test
# local imports
sys.path.append("..")
from shapelearningtheory.datasets import make_dataset
from helpers import format_table, train_and_validate, unpack_results, \
    get_basic_networks, get_standard_networks, create_save_path

parser = argparse.ArgumentParser("Expermient 1: are neural networks biased to shape or color/texture?")
parser.add_argument("--shape", type=str, default="rectangles", choices=["rectangles", "LvT"])
parser.add_argument("--pattern", type=str, default="color", choices=["color", "striped"])
parser.add_argument("--imgsize", type=str, default="small", choices=["small", "large"])
parser.add_argument("--repetitions", type=int, default=10)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batchsize", type=int, default=128)
parser.add_argument("--showlegend", action="store_true")
parser.add_argument("--random_seed", type=int, default=0)

def run_sign_test(data):
    models = data["model"].unique()
    datasets = data["dataset"].unique()
    test_statistics = {"model": [], "dataset": [], "M": [], "p": [], "accuracy": []}
    for model in models:
        for dataset in datasets:
            accuracies = data[(data["model"] == model) & (data["dataset"] == dataset)].loc[:, "accuracy"]
            try:
                M, p = sign_test(accuracies, 0.5)
            except ValueError: # can arise if all values are exactly 0.5
                M = 0
                p = 1.0
            test_statistics["model"].append(model)
            test_statistics["dataset"].append(dataset)
            test_statistics["M"].append(M)
            test_statistics["p"].append(p)
            test_statistics["accuracy"].append(accuracies.mean())
    return pd.DataFrame(test_statistics)

def mark_significance(ax, test_statistics):
    num_bars = len(test_statistics["model"].unique())
    for i, model in enumerate(test_statistics["model"].unique()):
        for j, dataset in enumerate(test_statistics["dataset"].unique()):
            idx = (test_statistics["model"] == model) & (test_statistics["dataset"] == dataset)
            M = test_statistics.loc[idx, "M"].squeeze()
            p = test_statistics.loc[idx, "p"].squeeze()
            accuracy = test_statistics.loc[idx, "accuracy"].squeeze()
            if p < 0.05:
                x = j + (i - (num_bars - 1) / 2) / (num_bars + 1)
                y = round(accuracy, 1) + 0.1
                if y > 0.8: # better alignment for ceiling performance
                    y = 1.1
                if M > 0:
                    ax.plot(x, y, "*", color="black")
                else:
                    ax.plot(x, y, "o", color="black")


if __name__ == "__main__":
    args = parser.parse_args()
    L.seed_everything(args.random_seed)
    patternonly = "color only" if args.pattern == "color" else "texture only"
    figpath = create_save_path("figures", "experiment_1", args.imgsize, args.shape, args.pattern)
    # get data:
    # training dataset
    traindata = make_dataset(args.shape, args.pattern, args.imgsize, "standard", batchsize=args.batchsize)
    #
    # test datasets - parametrized slightly differently to test generalization
    test_sets = {
        "traindata": traindata,
        #"validation": make_dataset(args.shape, args.pattern, args.imgsize, "standard", batchsize=args.batchsize),
        patternonly: make_dataset(args.shape, args.pattern, args.imgsize, "patternonly", batchsize=args.batchsize),
        "shape only": make_dataset(args.shape, args.pattern, args.imgsize, "shapeonly", batchsize=args.batchsize),
        "conflict": make_dataset(args.shape, args.pattern, args.imgsize, "conflict", batchsize=args.batchsize)
    }

    # hyperparameters from dataset
    traindata.prepare_data()
    imagesize = traindata.train.imgheight
    channels = 3
    classes = 2

    # define models
    if args.imgsize == "small":
        models = get_basic_networks(classes, channels, imagesize)
    else:
        models = get_standard_networks(classes, imagesize)

    # train and test
    test_results = {}
    for name, model in models.items():
        test_results[name] = train_and_validate(
            model, traindata, test_sets,
            epochs=args.epochs, repetitions=args.repetitions
        )

    # Print test results as table
    table = format_table(test_sets.keys(), test_results, cellwidth=15)
    print(table)
    with open(figpath + "/table.txt", "w") as f:
        f.write(table)
    # Plot results as bar plot
    df = unpack_results(test_results)
    df.to_csv(figpath + "/results.csv")
    fig, ax = plt.subplots()
    sns.barplot(df, x="dataset", y="accuracy", hue="model", ax=ax)
    if args.showlegend:
        ax.legend(loc="lower left")
    fig.suptitle(f"Accuracy on {args.pattern} {args.shape}")
    # perform hypothesis tests
    test_statistics = run_sign_test(df)
    test_statistics.to_csv(figpath + "/sign_tests.csv")
    mark_significance(ax, test_statistics)
    plt.savefig(figpath + "/accuracy_barplot.png")


            