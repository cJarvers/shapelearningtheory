import argparse
from matplotlib import pyplot as plt
import pytorch_lightning as L
import seaborn as sns
import sys
from statsmodels.stats.descriptivestats import sign_test
# local imports
sys.path.append("..")
from shapelearningtheory.datasets import make_dataset
from helpers import format_table, train_and_validate, unpack_results, get_basic_networks, \
    create_save_path, get_standard_networks
from experiment1_bias import run_sign_test, mark_significance

parser = argparse.ArgumentParser("Expermient 1: are neural networks biased to shape or color/texture?")
parser.add_argument("--shape", type=str, default="rectangles", choices=["rectangles", "LvT"])
parser.add_argument("--imgsize", type=str, default="small", choices=["small", "large"])
parser.add_argument("--repetitions", type=int, default=10)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batchsize", type=int, default=128)
parser.add_argument("--showlegend", action="store_true")
parser.add_argument("--random_seed", type=int, default=0)

if __name__ == "__main__":
    args = parser.parse_args()
    L.seed_everything(args.random_seed)
    figpath = create_save_path("figures", "experiment_2", args.imgsize, args.shape)
    # get data:
    # training dataset
    traindata = make_dataset(args.shape, "color", args.imgsize, "shapeonly", batchsize=args.batchsize)
    #
    # test datasets - parametrized slightly differently to test generalization
    test_sets = {
        "traindata": traindata,
        "validation": make_dataset(args.shape, "color", args.imgsize, "shapeonly", batchsize=args.batchsize),
        "with color": make_dataset(args.shape, "color", args.imgsize, "standard", batchsize=args.batchsize),
        "with texture": make_dataset(args.shape, "striped", args.imgsize, "standard", batchsize=args.batchsize)
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
    fig.suptitle(f"Accuracy on shape-only {args.shape}")
    # perform hypothesis tests
    test_statistics = run_sign_test(df)
    test_statistics.to_csv(figpath + "/sign_tests.csv")
    mark_significance(ax, test_statistics)
    plt.savefig(figpath + "/accuracy_barplot.png")