import argparse
from matplotlib import pyplot as plt
import pytorch_lightning as L
import seaborn as sns
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.datasets import make_dataset
from shapelearningtheory.networks import make_vit_b_16
from helpers import format_table, train_and_validate, unpack_results, \
    create_save_path, get_standard_networks
from experiment1_bias import run_sign_test, mark_significance

parser = argparse.ArgumentParser("Expermient 7: Do transformers learn better with different hyperparameters?")
parser.add_argument("--shape", type=str, default="rectangles", choices=["rectangles", "LvT"])
parser.add_argument("--repetitions", type=int, default=10)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batchsize", type=int, default=128)
parser.add_argument("--showlegend", action="store_true")
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--pretrained", action="store_true", help="For large images, choose whether to use pretrained weights from torchvision")

if __name__ == "__main__":
    args = parser.parse_args()
    L.seed_everything(args.random_seed)
    figpath = create_save_path("figures", "experiment_7", args.shape)
    # get data:
    # training dataset
    traindata = make_dataset(args.shape, "color", "large", "shapeonly", batchsize=args.batchsize)
    #
    # test datasets - parametrized slightly differently to test generalization
    test_sets = {
        "traindata": traindata,
        "validation": make_dataset(args.shape, "color", "large", "shapeonly", batchsize=args.batchsize),
        "with color": make_dataset(args.shape, "color", "large", "standard", batchsize=args.batchsize),
        "with texture": make_dataset(args.shape, "striped", "large", "standard", batchsize=args.batchsize)
    }

    # hyperparameters from dataset
    traindata.prepare_data()
    imagesize = traindata.train.imgheight
    channels = 3
    classes = 2

    # define models
    models = {
        "lr=0.01": lambda: make_vit_b_16(imagesize, classes, False, lr=0.01),
        "lr=0.001": lambda: make_vit_b_16(imagesize, classes, False, lr=0.001),
        "lr=0.0001": lambda: make_vit_b_16(imagesize, classes, False, lr=0.0001),
    }

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