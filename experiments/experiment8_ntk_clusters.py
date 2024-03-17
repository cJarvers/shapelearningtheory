import argparse
from matplotlib import pyplot as plt
import os
import sys
import pandas as pd
import pytorch_lightning as L
from pytorch_lightning import Trainer
import seaborn as sns
from sklearn import cluster, metrics
from statsmodels.stats.nonparametric import rank_compare_2indep
import torch
from typing import Any, Literal
# local imports
sys.path.append("..")
from shapelearningtheory.datasets import make_dataset
from shapelearningtheory.networks import make_convnet_small, make_softmaxconv_small
from shapelearningtheory.analysis.ntk import *
from helpers import create_save_path

parser = argparse.ArgumentParser("Run Neural Tangent Kernel experiments.")
parser.add_argument("--nettype", type=str, default="ConvNet", choices=["ConvNet", "spcConvNet"])
parser.add_argument("--shape", type=str, default="rectangles", choices=["rectangles", "LvT"])
parser.add_argument("--pattern", type=str, default="color", choices=["color", "striped"])
parser.add_argument("--eval_variant", type=str, default="standard", choices=["standard", "random"])
parser.add_argument("--repetitions", type=int, default=10)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--random_seed", type=int, default=0)

def run_cluster_analysis(args, variant, eval_data):
    net = set_up_network(args.nettype)
    data = make_dataset(args.shape, args.pattern, "small", variant)
    net = train_reference_net(net, data, args.epochs)
    eval_x, eval_y = next(iter(eval_data.test_dataloader()))
    ntk = get_ntk(net, eval_x)
    affinity_matrix = affinity_matrix_from_NTK(ntk)
    agreement = compare_clusters_to_labels(affinity_matrix, eval_y)
    return agreement


def set_up_network(nettype: Literal["ConvNet", "spcConvNet"]):
    if nettype == "ConvNet":
        net = make_convnet_small(channels=3, classes=2)
    else:
        net = make_softmaxconv_small(channels=3, classes=2)
    return net

def train_reference_net(net, data, epochs):
    trainer = Trainer(max_epochs=epochs, accelerator="gpu", logger=False, enable_checkpointing=False,
                      check_val_every_n_epoch=epochs+1)
    trainer.fit(net, data)
    return net

def compare_clusters_to_labels(affinity_matrix, labels, n_cluster_range=range(2, 50)):
    agreement = []
    for n_clusters in n_cluster_range:
        cluster_model = cluster.SpectralClustering(n_clusters=n_clusters, affinity="precomputed", assign_labels="cluster_qr")
        cluster_model.fit(affinity_matrix)
        agreement.append(metrics.adjusted_rand_score(labels, cluster_model.labels_))
    return agreement

def affinity_matrix_from_NTK(ntk):
    return torch.exp(ntk / ntk.std()).nan_to_num()

def compare_maxima(data, args):
    maxima = data.groupby(["NTK type", "repetition"]).max().reset_index()
    shape_agreement = maxima[maxima["NTK type"] == "shape"].loc[:, "agreement"]
    pattern_agreement = maxima[maxima["NTK type"] == args.patternname].loc[:, "agreement"]
    result = rank_compare_2indep(pattern_agreement, shape_agreement)
    return result



if __name__ == "__main__":
    args = parser.parse_args()
    L.seed_everything(args.random_seed)
    args.patternname = "color" if args.pattern == "color" else "texture"
    figpath = create_save_path("figures", "experiment_8", args.nettype, args.shape, args.pattern)

    # clustering analysis for NTKs 
    n_cluster_range=range(2, 50)
    clustering_results = []
    for rep in range(args.repetitions):
        # set up evaluation data
        eval_data = make_dataset(args.shape, args.pattern, "eval", "standard", batchsize=4096)
        # train networks
        shape_agreement = run_cluster_analysis(args, "shapeonly", eval_data)
        clustering_results.append(
            pd.DataFrame(zip(shape_agreement, ["shape"] * len(shape_agreement), [rep] * len(shape_agreement)),
                         columns=["agreement", "NTK type", "repetition"])
        )
        pattern_agreement = run_cluster_analysis(args, "patternonly", eval_data)
        clustering_results.append(
            pd.DataFrame(zip(pattern_agreement, [args.patternname] * len(pattern_agreement), [rep] * len(pattern_agreement)),
                         columns=["agreement", "NTK type", "repetition"])
        )
    clustering_results = pd.concat(clustering_results)
    clustering_results.to_csv(figpath + "/cluster_agreement.csv")
    # plot results
    plt.clf()
    sns.lineplot(clustering_results, x=clustering_results.index, y="agreement", hue="NTK type")
    plt.xlabel("number of clusters")
    plt.title("NTK clustering on " + args.pattern + " " + args.shape)
    plt.savefig(figpath + "/cluster_analysis.png", bbox_inches="tight")
    plt.clf()
    # significance test
    test_result = compare_maxima(clustering_results, args)
    with open(figpath + "/rank_comparison.txt", "w") as f:
        f.write(f"Result of comparing maximum agreement between labels and {args.pattern} NTK clusters to agreement between labels and {args.shape} NTK clusters.\n\n")
        f.write(str(test_result.summary()))
        f.write("\n\n")
        f.write("Full statistics:\n" + str(test_result))
