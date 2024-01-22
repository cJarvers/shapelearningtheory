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
from shapelearningtheory.networks import make_convnet_small, make_softmaxconv_small, \
    SRectangleConvNet, LTConvNet, ColorConvNet, TextureConvNet
from shapelearningtheory.analysis.ntk import *

parser = argparse.ArgumentParser("Run Neural Tangent Kernel experiments.")
parser.add_argument("--nettype", type=str, default="ConvNet", choices=["ConvNet", "spcConvNet"])
parser.add_argument("--shape", type=str, default="rectangles", choices=["rectangles", "LvT"])
parser.add_argument("--pattern", type=str, default="color", choices=["color", "striped"])
parser.add_argument("--eval_variant", type=str, default="standard", choices=["standard", "random"])
parser.add_argument("--repetitions", type=int, default=10)

def set_up_networks(nettype: Literal["ConvNet", "spcConvNet"], shape: Literal["rectangles", "LvT"], pattern: Literal["color", "striped"]):
    if nettype == "ConvNet":
        net = make_convnet_small(channels=3, classes=2)
    else:
        net = make_softmaxconv_small(channels=3, classes=2)
    if shape == "rectangles":
        shapenet = SRectangleConvNet()
    else:
        shapenet = LTConvNet()
    if pattern == "color":
        patternnet = ColorConvNet()
    else:
        patternnet = TextureConvNet()
    return net, shapenet, patternnet

def get_ntk(net: torch.nn.Module, data: torch.Tensor):
    ntk = empirical_ntk(functionalize(net), get_parameters(net), data)
    ntk = ntk.permute(2,0,1) # batchsize * batchsize * neurons -> neurons * batchsize * batchsize
    return ntk

def get_ntk_similarity(net1, net2, data):
    ntk1 = get_ntk(net1, data).flatten(start_dim=1)
    ntk2 = get_ntk(net2, data).flatten(start_dim=1)
    similarity = torch.cosine_similarity(ntk1, ntk2)
    return similarity

def run_analysis(nettype, shape, pattern, repetitions, eval_variant):
    similarities = {"net2shape": [], "net2"+pattern: []}
    for _ in range(repetitions):
        # set up data
        eval_data = make_dataset(shape, pattern, "eval", eval_variant, batchsize=512)
        x, _ = next(iter(eval_data.test_dataloader()))
        #train_data = make_dataset(shape, pattern, "small", "standard")
        # set up networks
        net, shapenet, patternnet = set_up_networks(nettype, shape, pattern)
        net(x[0:1]) # initialize lazy layers
        # compute NTKs and similarities
        sim_net2shape = get_ntk_similarity(net, shapenet, x)
        sim_net2pattern = get_ntk_similarity(net, patternnet, x)
        similarities["net2shape"].append(sim_net2shape)
        similarities["net2"+pattern].append(sim_net2pattern)
        # train network for one epoch
        #train_one_epoch(net, train_data)
        # compare NTKs again
    for k, v in similarities.items():
        similarities[k] = torch.concat(v, dim=0)
    return similarities

def plot_similarities(pattern, shape, similarities):
    sns.violinplot(similarities, inner="point")
    plt.savefig("exp6_softmax_" + pattern + shape + ".png")

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
        self.sim_net2shape.append(net2shape.mean().item())
        self.sim_net2pattern.append(net2pattern.mean().item())

def create_save_path(net, shape, pattern):
    path = os.path.join("figures", "experiment_6", net, shape, pattern)
    os.makedirs(path, exist_ok=True)
    return path


if __name__ == "__main__":
    args = parser.parse_args()
    args.patternname = "color" if args.pattern == "color" else "texture"
    args.shapenet = "RectangleNet" if args.shape == "rectangles" else "LvTNet"
    args.patternnet = "ColorNet" if args.pattern == "color" else "StripeNet"
    figpath = create_save_path(args.nettype, args.shape, args.pattern)
    # set up logging
    similarities = []
    # old code:
    #similarities = run_analysis(args.nettype, args.shape, args.pattern, repetitions, args.eval_variant)
    #plot_similarities(args.pattern, args.shape, similarities)
    ## Evaluate NTK during training
    for rep in range(args.repetitions):
        # set up data   
        train_data = make_dataset(args.shape, args.pattern, "small", "standard")
        eval_data = make_dataset(args.shape, args.pattern, "eval", args.eval_variant, batchsize=512)
        x, _ = next(iter(eval_data.test_dataloader()))
        # set up networks, trainer and callbacks
        net, shapenet, patternnet = set_up_networks(args.nettype, args.shape, args.pattern)
        net(x[0:1]) # initialize lazy layers
        # compute NTK for randomly initialized network
        ntk_pre = get_ntk(net, x)
        # train
        callback = NTKSimilarityCallback(x.to(device="cuda"),
                                        shapenet.to(device="cuda"),
                                        patternnet.to(device="cuda"))
        trainer = Trainer(max_epochs=10, accelerator="gpu",
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
    #plt.plot(callback.sim_net2shape, label=f"{args.nettype} to {args.shape}net")
    #plt.plot(callback.sim_net2pattern, label=f"{args.nettype} to {args.patternname}net")
    #plt.legend()
    sns.lineplot(similarities, x=similarities.index, y="NTK similarity", hue=args.nettype + " and:")
    #plt.title("NTK similarity")
    plt.xlabel("training batch")
    #plt.ylabel("similarity")
    plt.savefig(figpath + "/similarity.png", bbox_inches="tight")
    plt.clf()
    # plot NTKs after training
    ntk_post = get_ntk(net, x.to("cpu"))
    ntk_shape = get_ntk(shapenet.to("cpu"), x.to("cpu"))
    ntk_pattern = get_ntk(patternnet.to("cpu"), x.to("cpu"))
    fig = plot_ntks(ntk_pre[0], ntk_post[0], ntk_pattern[0], ntk_shape[0], args,
                    title="Neural Tangent Kernels")
    fig.savefig(figpath + "/ntks.png", bbox_inches="tight")