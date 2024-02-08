import argparse
from matplotlib import pyplot as plt
import os
import sys
import pandas as pd
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
import seaborn as sns
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
parser.add_argument("--eval_step_size", type=int, default=1)

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
    def __init__(self, data, reference_net, stepsize: int=1):
        self.data = data
        self.reference_net = reference_net
        self.stepsize = stepsize
        self.similarity = []
        self.batch_idx = []

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        if batch_idx % self.stepsize == 0:
            similarity = get_ntk_similarity(pl_module, self.reference_net, self.data)
            self.similarity.append(similarity)
            self.batch_idx.append(batch_idx)

def train_reference_net(nettype, epochs, **data_args):
    net = set_up_network(nettype)
    trainer = Trainer(max_epochs=epochs, accelerator="gpu", logger=False, enable_checkpointing=False)
    data = make_dataset(**data_args)
    trainer.fit(net, data)
    return net



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
        eval_data = make_dataset(args.shape, args.pattern, "eval", args.eval_variant, batchsize=2048)
        eval_x, eval_y = next(iter(eval_data.test_dataloader()))
        # set up network to train
        net = set_up_network(args.nettype)
        net(eval_x[0:1]) # initialize lazy layers
        # compute NTK for randomly initialized network
        ntk_pre = get_ntk(net, eval_x)
        # train
        shape_callback = NTKSimilarityCallback(eval_x.to(device="cuda"),
                                        shapenet.to(device="cuda"),
                                        args.eval_step_size)
        pattern_callback = NTKSimilarityCallback(eval_x.to(device="cuda"),
                                        patternnet.to(device="cuda"),
                                        args.eval_step_size)
        trainer = Trainer(max_epochs=args.epochs_main, accelerator="gpu",
                          logger=False, enable_checkpointing=False,
                          callbacks=[shape_callback, pattern_callback])
        # train network
        trainer.fit(net, train_data)
        # store results
        batches = len(shape_callback.similarity)
        similarities.append(
            pd.DataFrame(zip(shape_callback.similarity, [args.shapenet]*batches, [rep]*batches, shape_callback.batch_idx),
                         columns=["NTK similarity", args.nettype + " and:", "repetition", "training step"])
        )
        similarities.append(
            pd.DataFrame(zip(pattern_callback.similarity, [args.patternnet]*batches, [rep]*batches, pattern_callback.batch_idx),
                         columns=["NTK similarity", args.nettype + " and:", "repetition", "training step"])
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