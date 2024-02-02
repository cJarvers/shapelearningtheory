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
parser.add_argument("--pattern", type=str, default="color", choices=["color", "striped"])
parser.add_argument("--eval_variant", type=str, default="standard", choices=["standard", "random"])
parser.add_argument("--repetitions", type=int, default=10)
parser.add_argument("--epochs_main", type=int, default=10)
parser.add_argument("--epochs_reference", type=int, default=30)

def set_up_network(nettype: Literal["ConvNet", "spcConvNet"]):
    if nettype == "ConvNet":
        net = make_convnet_small(channels=3, classes=2)
    else:
        net = make_softmaxconv_small(channels=3, classes=2)
    return net

def plot_ntks(ntk_pre, ntk_post, ntk_pattern, ntk_shape, args, title=""):
    fig, ax = plt.subplots(1, 4, figsize=(20, 8))
    fig.suptitle(title)
    im = ax[0].imshow(ntk_pre, cmap="gray")
    ax[0].set_title(f"{args.nettype} before training")
    ax[0].set_axis_off()
    plt.colorbar(im, ax=ax[0])
    im = ax[1].imshow(ntk_post, cmap="gray")
    ax[1].set_title(f"{args.nettype} after training")
    ax[1].set_axis_off()
    plt.colorbar(im, ax=ax[1])
    im = ax[2].imshow(ntk_shape, cmap="gray")
    ax[2].set_title("shape network")
    ax[2].set_axis_off()
    plt.colorbar(im, ax=ax[2])
    im = ax[3].imshow(ntk_pattern, cmap="gray")
    ax[3].set_title(f"{args.patternname} network")
    ax[3].set_axis_off()
    plt.colorbar(im, ax=ax[3])
    return fig

class NTKSimilarityCallback(Callback):
    def __init__(self, data, shapenet, patternnet):
        self.data = data
        self.shapenet = shapenet
        self.patternnet = patternnet
        self.sim_net2shape = []
        self.sim_net2pattern = []

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        net2shape = get_ntk_similarity(pl_module, self.shapenet, self.data)
        net2pattern = get_ntk_similarity(pl_module, self.patternnet, self.data)
        self.sim_net2shape.append(net2shape)
        self.sim_net2pattern.append(net2pattern)

def train_reference_net(nettype, epochs, **data_args):
    net = set_up_network(nettype)
    trainer = Trainer(max_epochs=epochs, accelerator="gpu", logger=False, enable_checkpointing=False)
    data = make_dataset(**data_args)
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
    return torch.exp(ntk / ntk.std())



if __name__ == "__main__":
    args = parser.parse_args()
    args.patternname = "color" if args.pattern == "color" else "texture"
    args.shapenet = "shape trained"
    args.patternnet = args.patternname + " trained"
    figpath = create_save_path("figures", "experiment_7", args.nettype, args.shape, args.pattern)
    # set up reference networks
    shapenet = train_reference_net(args.nettype, args.epochs_reference,
                                   shape=args.shape, pattern=args.pattern,
                                   size="small", variant="shapeonly")
    patternnet = train_reference_net(args.nettype, args.epochs_reference,
                                     shape=args.shape, pattern=args.pattern,
                                     size="small", variant="patternonly")
    # set up logging
    similarities = []
    ## Evaluate NTK during training
    for rep in range(args.repetitions):
        # set up data   
        train_data = make_dataset(args.shape, args.pattern, "small", "standard")
        eval_data = make_dataset(args.shape, args.pattern, "eval", args.eval_variant, batchsize=512)
        eval_x, eval_y = next(iter(eval_data.test_dataloader()))
        # set up network to train
        net = set_up_network(args.nettype)
        net(eval_x[0:1]) # initialize lazy layers
        # compute NTK for randomly initialized network
        ntk_pre = get_ntk(net, eval_x)
        # train
        callback = NTKSimilarityCallback(eval_x.to(device="cuda"),
                                        shapenet.to(device="cuda"),
                                        patternnet.to(device="cuda"))
        trainer = Trainer(max_epochs=args.epochs_main, accelerator="gpu",
                        logger=False, enable_checkpointing=False,
                        callbacks=[callback])
        # train network
        trainer.fit(net, train_data)
        # store results
        batches = len(callback.sim_net2shape)
        similarities.append(
            pd.DataFrame(zip(callback.sim_net2shape, [args.shapenet]*batches, [rep]*batches),
                         columns=["NTK similarity", args.nettype + " and:", "repetition"])
        )
        similarities.append(
            pd.DataFrame(zip(callback.sim_net2pattern, [args.patternnet]*batches, [rep]*batches),
                         columns=["NTK similarity", args.nettype + " and:", "repetition"])
        )
    similarities = pd.concat(similarities)
    # plot results
    plt.clf()
    sns.lineplot(similarities, x=similarities.index, y="NTK similarity", hue=args.nettype + " and:")
    plt.xlabel("training batch")
    plt.savefig(figpath + "/similarity.png", bbox_inches="tight")
    plt.clf()
    # plot NTKs after training
    ntk_post = get_ntk(net, eval_x.to("cpu"))
    ntk_shape = get_ntk(shapenet.to("cpu"), eval_x.to("cpu"))
    ntk_pattern = get_ntk(patternnet.to("cpu"), eval_x.to("cpu"))
    fig = plot_ntks(ntk_pre, ntk_post, ntk_pattern, ntk_shape, args,
                    title="Neural Tangent Kernels")
    fig.savefig(figpath + "/ntks.png", bbox_inches="tight")

    # clustering analysis for NTKs 
    n_cluster_range=range(2, 50)
    clustering_results = []
    for rep in range(args.repetitions):
        shapenet = train_reference_net(args.nettype, args.epochs_reference,
                                       shape=args.shape, pattern=args.pattern,
                                       size="small", variant="shapeonly")
        shape_agreement = compare_clusters_to_labels(
            affinity_matrix=affinity_matrix_from_NTK(get_ntk(shapenet, eval_x)),
            labels=eval_y,
            n_cluster_range=n_cluster_range
        )
        clustering_results.append(
            pd.DataFrame(zip(shape_agreement, ["shape"] * len(shape_agreement), [rep] * len(shape_agreement)),
                         columns=["agreement", "NTK type", "repetition"])
        )
        patternnet = train_reference_net(args.nettype, args.epochs_reference,
                                         shape=args.shape, pattern=args.pattern,
                                         size="small", variant="patternonly")
        pattern_agreement = compare_clusters_to_labels(
            affinity_matrix=affinity_matrix_from_NTK(get_ntk(patternnet, eval_x)),
            labels=eval_y,
            n_cluster_range=n_cluster_range
        )
        clustering_results.append(
            pd.DataFrame(zip(pattern_agreement, ["pattern"] * len(pattern_agreement), [rep] * len(pattern_agreement)),
                         columns=["agreement", "NTK type", "repetition"])
        )
    clustering_results = pd.concat(clustering_results)
    # plot results
    plt.clf()
    sns.lineplot(clustering_results, x=clustering_results.index, y="agreement", hue="NTK type")
    plt.xlabel("number of clusters")
    plt.savefig(figpath + "/cluster_analysis.png", bbox_inches="tight")
    plt.clf()