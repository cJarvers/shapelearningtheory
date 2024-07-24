import argparse
from matplotlib import pyplot as plt
import pytorch_lightning as L
import seaborn as sns
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.datasets.multi_shape_dataset import MultiShapeDataModule
from helpers import format_table, train_and_validate, unpack_results, \
    get_standard_networks, create_save_path
from experiment1_bias import run_sign_test, mark_significance

parser = argparse.ArgumentParser("Expermient 5: are neural networks biased to shape or color (multiple shapes)?")
parser.add_argument("--repetitions", type=int, default=10)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batchsize", type=int, default=128)
parser.add_argument("--showlegend", action="store_true")
parser.add_argument("--random_seed", type=int, default=0)

if __name__ == "__main__":
    args = parser.parse_args()
    L.seed_everything(args.random_seed)
    figpath = create_save_path("figures", "experiment_5")
    # get data:
    dataset = MultiShapeDataModule(image_size=112, images_per_class=1000)
    dataset.prepare_data()
    # training dataset
    traindata = dataset.train_dataloader()
    #
    # test datasets - parametrized slightly differently to test generalization
    test_sets = {
        "validation": dataset.test_dataloader(),
        "random shape": dataset.random_shape_dataloader(),
        "random color": dataset.random_color_dataloader()
    }

    # hyperparameters from dataset
    imagesize = dataset.image_size
    channels = 3
    classes = 10

    # define models
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
    fig.suptitle(f"Accuracy on multishape dataset")
    # perform hypothesis tests
    chance_level = 1 / len(dataset.train.shape_classes)
    print("Chance level is: ", chance_level)
    test_statistics = run_sign_test(df, chance_level=1 / len(dataset.train.shape_classes))
    test_statistics.to_csv(figpath + "/sign_tests.csv")
    mark_significance(ax, test_statistics)
    plt.savefig(figpath + "/accuracy_barplot.png")


            