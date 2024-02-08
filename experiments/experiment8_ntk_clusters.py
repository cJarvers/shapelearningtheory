import argparse
from matplotlib import pyplot as plt
import os
import sys
import pandas as pd
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
import seaborn as sns
from sklearn import cluster, metrics
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
parser.add_argument("--eval_variant", type=str, default="standard", choices=["standard", "random"])
parser.add_argument("--repetitions", type=int, default=10)
parser.add_argument("--epochs", type=int, default=30)

def run_cluster_analysis(args, pattern, variant, eval_data):
    net = set_up_network(args.nettype)
    data = make_dataset(args.shape, pattern, "small", variant)
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
    trainer = Trainer(max_epochs=epochs, accelerator="gpu", logger=False, enable_checkpointing=False)
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



if __name__ == "__main__":
    args = parser.parse_args()
    figpath = create_save_path("figures", "experiment_8", args.nettype, args.shape)

    # clustering analysis for NTKs 
    n_cluster_range=range(2, 50)
    clustering_results = []
    for rep in range(args.repetitions):
        # set up evaluation data
        shape_eval_data = make_dataset(args.shape, "color", "eval", "shapeonly", batchsize=4096)
        color_eval_data = make_dataset(args.shape, "color", "small", "patternonly", batchsize=4096)
        texture_eval_data = make_dataset(args.shape, "texture", "small", "patternonly", batchsize=4096)
        # train networks
        shape_agreement = run_cluster_analysis(args, "color", "shapeonly", shape_eval_data)
        clustering_results.append(
            pd.DataFrame(zip(shape_agreement, ["shape"] * len(shape_agreement), [rep] * len(shape_agreement)),
                         columns=["agreement", "NTK type", "repetition"])
        )
        color_agreement = run_cluster_analysis(args, "color", "patternonly", color_eval_data)
        clustering_results.append(
            pd.DataFrame(zip(color_agreement, ["color"] * len(color_agreement), [rep] * len(color_agreement)),
                         columns=["agreement", "NTK type", "repetition"])
        )
        texture_agreement = run_cluster_analysis(args, "striped", "patternonly", texture_eval_data)
        clustering_results.append(
            pd.DataFrame(zip(texture_agreement, ["texture"] * len(texture_agreement), [rep] * len(texture_agreement)),
                         columns=["agreement", "NTK type", "repetition"])
        )
    clustering_results = pd.concat(clustering_results)
    # plot results
    plt.clf()
    sns.lineplot(clustering_results, x=clustering_results.index, y="agreement", hue="NTK type")
    plt.xlabel("number of clusters")
    plt.title("NTK clustering on " + args.shape)
    plt.savefig(figpath + "/cluster_analysis.png", bbox_inches="tight")
    plt.clf()