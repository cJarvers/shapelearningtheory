import argparse
from matplotlib import pyplot as plt
import pandas as pd
import pytorch_lightning as L
import seaborn as sns
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.datasets.multi_shape_dataset import *
from helpers import train_and_validate, unpack_results, \
    create_save_path, get_standard_networks
from experiment1_bias import run_sign_test, mark_significance

parser = argparse.ArgumentParser("Expermient 6: learnability on multi-shape dataset")
parser.add_argument("--dataset", type=str, choices=["random_color", "random_shape", "ignoring_orientation", "ignoring_shape_type"])
parser.add_argument("--repetitions", type=int, default=10)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batchsize", type=int, default=128)
parser.add_argument("--showlegend", action="store_true")
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--image_size", type=int, default=112)
parser.add_argument("--images_per_class", type=int, default=1000)

if __name__ == "__main__":
    args = parser.parse_args()
    L.seed_everything(args.random_seed)
    figpath = create_save_path("figures", "experiment_6")
    # get data:
    # training dataset
    if args.dataset == "random_color":
        dataset = MultiShapeDataModuleWithRandomColors(image_size=args.image_size, images_per_class=args.images_per_class)
    elif args.dataset == "random_shape":
        dataset = MultiShapeDataModuleWithRandomShape(image_size=args.image_size, images_per_class=args.images_per_class)
    elif args.dataset == "ignoring_orientation":
        dataset = MultiShapeDataModuleWithRandomColorIgnoringOrientation(image_size=args.image_size, images_per_class=args.images_per_class)
    elif args.dataset == "ignoring_shape_type":
        dataset = MultiShapeDataModuleWithRandomColorIgnoringShapeType(image_size=args.image_size, images_per_class=args.images_per_class)
    else:
        raise ValueError("Unknown dataset option: " + args.dataset)
    dataset.prepare_data()

    # hyperparameters from dataset
    imagesize = dataset.image_size
    num_classes = len(dataset.train.shape_classes)
    channels = 3

    # train and test
    test_results = {}
    models = get_standard_networks(num_classes, imagesize)
    for model_name, model in models.items():
        test_results[model_name] = train_and_validate(
            model, dataset.train_dataloader(), {args.dataset: dataset.test_dataloader()},
            epochs=args.epochs, repetitions=args.repetitions
        )

    # Print test results as table
    df = unpack_results(test_results)
    df.to_csv(figpath + "/results_" + args.dataset + ".csv")
    print(df)
    # Plot results as bar plot
    fig, ax = plt.subplots()
    sns.barplot(df, x="dataset", y="accuracy", hue="model", ax=ax)
    if args.showlegend:
        ax.legend(loc="lower left")
    # perform hypothesis tests
    chance_level = 1 / len(dataset.train.shape_classes)
    test_statistics = run_sign_test(df, chance_level=chance_level)
    test_statistics.to_csv(figpath + "/sign_tests_" + args.dataset + ".csv")
    mark_significance(ax, test_statistics)
    plt.savefig(figpath + f"/accuracy_barplot_{args.dataset}.png")
