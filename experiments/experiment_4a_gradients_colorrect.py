import numpy as np
from matplotlib import pyplot as plt
import os
import pytorch_lightning as pl
import seaborn as sns
import sys
import torch
from torch.func import vmap
from torchmetrics.functional import pairwise_cosine_similarity
# local imports
sys.path.append("..")
from shapelearningtheory.analysis.gradients import *
from shapelearningtheory.datasets import make_rectangles_color
from shapelearningtheory.datasets import RectangleDataModule
from shapelearningtheory.colors import RedXORBlue, NotRedXORBlue, RandomGrey
from shapelearningtheory.networks import make_convnet_small, ColorConvNet, CRectangleConvNet

# Load dataset
print("Loading data")
traindata = make_rectangles_color()
traindata.prepare_data()
eval_data = RectangleDataModule(
    imgheight=18, imgwidth=18, 
    lengths=range(7, 13), widths=range(5, 10, 2),
    pattern1=RedXORBlue, pattern2=NotRedXORBlue,
    background_pattern=RandomGrey,
    oversampling_factor=1,
    stride=4,
    batch_size=32)
eval_data.prepare_data()
eval_data_loader = eval_data.test_dataloader()
imgheight = eval_data.dataset.imgheight
imgwidth = eval_data.dataset.imgwidth
channels = 3
classes = 2

# build networks
print("Building networks")
color_net = ColorConvNet(imageheight=imgheight, imagewidth=imgwidth)
shape_net = CRectangleConvNet()
net = make_convnet_small(channels=channels, classes=classes)
p = net(next(iter(traindata.train_dataloader()))[0]) # call once to initialize lazy modules
del p

# set up helper dicts for layers of interest
color_net_layers = {"RBdiff": 1, "colorclass": 5}
shape_net_layers = {"borders1": 1, "borders2": 3, "distance": 4,
                    "decision": 5, "shapeclass": 6}
#net_layers = {"output": 5}
#net_layers = {"block 2": 3, "output": 5}
net_layers = {"block 0": 1, "block 1": 2, "block 2": 3, "output": 5}

# calculate gradient alignment before training
max_batches = 200
neurons_per_layer = 2000
    
def get_net_jacobians(net, net_layers, data, neurons_per_layer):
    net_jacobians = {}
    for layer, index in net_layers.items():
        subnet = net.layers[:index]
        jacobian = get_jacobians(subnet, data, max_batches)
        net_jacobians[layer] = select_random_subset(jacobian, neurons_per_layer)
    return net_jacobians

def get_feature_jacobians(color_net, color_net_layers, shape_net, shape_net_layers, data):
    feature_jacobians = {}
    for layer, index in color_net_layers.items():
        color_subnet = color_net[:index]
        feature_jacobians[layer] = get_jacobians(color_subnet, data, max_batches)
    for layer, index in shape_net_layers.items():
        shape_subnet = shape_net[:index]
        feature_jacobians[layer] = get_jacobians(shape_subnet, data, max_batches)
    return feature_jacobians

@torch.no_grad()
def get_all_consistencies(net_jacobians, feature_jacobians):
    consistencies = {}
    projections = {}
    for layer_name, layer_grads in net_jacobians.items():
        consistencies[layer_name] = {}
        projections[layer_name] = {}
        for feature_name, feature_grads in feature_jacobians.items():
            print(f"Consistency {layer_name} - {feature_name}", end="", flush=True)
            projection = vmap(project_gradients_to_feature, chunk_size=1)(
                    torch.tensor(layer_grads, device="cuda"),
                    torch.tensor(feature_grads, device="cuda")
                )
            projections[layer_name][feature_name] = projection.cpu().numpy()
            consistency = vmap(pairwise_cosine_similarity,
                               in_dims=1,
                               chunk_size=1)(
                                   projection
                               )
            consistency.nan_to_num_()
            consistencies[layer_name][feature_name] = consistency.cpu().numpy()
            print(" - done")
    return consistencies, projections

@torch.no_grad()
def compare_all_consistencies(net_jacobians, feature_jacobians):
    consistencies = {}
    for layer_name, layer_grads in net_jacobians.items():
        consistencies[layer_name] = {}
        for feature_name, feature_grads in feature_jacobians.items():
            consistency = compare_consistencies(
                    torch.tensor(layer_grads, device="cuda"),
                    torch.tensor(feature_grads, device="cuda")
                )
            consistency.nan_to_num_()
            consistencies[layer_name][feature_name] = consistency.cpu().numpy()
    return consistencies

def sparsity(x):
    return np.abs(x).mean(axis=(1,2)) / np.square(x).mean(axis=(1,2))

def make_violinplot(consistencies):
    fig, ax = plt.subplots(len(consistencies), 1, figsize=(8, 12))
    for i, (layer_name, layer_consistencies) in enumerate(consistencies.items()):
        sns.violinplot({k: sparsity(v)# v.mean(axis=(1, 2)) 
                        for k, v in layer_consistencies.items()},
            ax=ax[i])
        ax[i].set_title(layer_name)
    return fig

def plot_projection_lengths(projections):
    fig, ax = plt.subplots(len(consistencies), 1, figsize=(8, 12))
    for i, (layer_name, layer_projections) in enumerate(projections.items()):
        sns.violinplot({k: v.mean(axis=(1, 2)) 
                        for k, v in layer_projections.items()},
            ax=ax[i])
        ax[i].set_title(layer_name)
    return fig

def make_imshow_plot(consistencies):
    fig, ax = plt.subplots(len(consistencies), len(feature_jacobians), figsize=(28, 16))
    fig2, ax2 = plt.subplots(len(consistencies), len(feature_jacobians), figsize=(28, 16))
    vmin = min(map(min, [[cmat.mean(0).min() for cmat in feature_consistencies.values()] for feature_consistencies in consistencies.values()]))
    vmax = max(map(max, [[cmat.mean(0).max() for cmat in feature_consistencies.values()] for feature_consistencies in consistencies.values()]))
    for i, (layer_name, layer_consistencies) in enumerate(consistencies.items()):
        for j, (feature_name, feature_consistency) in enumerate(layer_consistencies.items()):
            im = ax[i][j].imshow(
                feature_consistency.mean(axis=0),
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                interpolation=None)
            im2 = ax2[i][j].imshow(
                feature_consistency.var(axis=0),
                cmap="viridis",
                #vmin=vmin,
                #vmax=vmax,
                interpolation=None
            )
            ax[i][j].set_xticks([])
            ax2[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            ax2[i][j].set_yticks([])
            #plt.colorbar(im, ax=ax[i][j])
            plt.colorbar(im2, ax=ax2[i][j])
            if i == 0:
                ax[i][j].set_title(feature_name)
                ax2[i][j].set_title(feature_name)
            if j == 0:
                ax[i][j].set_ylabel(layer_name)
                ax2[i][j].set_ylabel(layer_name)
    fig.suptitle("Gradient consistency (mean over neurons)")
    fig2.suptitle("Gradient consistency (variance over neurons)")
    return fig, fig2


print("Computing network jacobians")
net_jacobians = get_net_jacobians(net, net_layers, eval_data_loader, neurons_per_layer)
print("Computing feature jacobians")
feature_jacobians = get_feature_jacobians(color_net, color_net_layers, shape_net, shape_net_layers, eval_data_loader)
print("Computing consistencies")
#consistencies, projections = get_all_consistencies(net_jacobians, feature_jacobians)
consistencies = compare_all_consistencies(net_jacobians, feature_jacobians)
os.makedirs("figures/experiment_4a", exist_ok=True)
fig = make_violinplot(consistencies)
fig.savefig("figures/experiment_4a/gradient_sparsity.png", bbox_inches="tight")
#fig = plot_projection_lengths(projections)
#fig.savefig("figures/experiment_4a/projection_lengths.png", bbox_inches="tight")
fig, fig2 = make_imshow_plot(consistencies)
fig.savefig("figures/experiment_4a/gradient_consistency_imgs.png", bbox_inches="tight")
fig2.savefig("figures/experiment_4a/gradient_consistency_variance.png", bbox_inches="tight")
del net_jacobians, feature_jacobians, consistencies#, projections

# train for an epoch and plot again
print("Training network for 1 epoch")
trainer = pl.Trainer(max_epochs=1, accelerator="gpu", logger=False,
    enable_checkpointing=False)
trainer.fit(net, traindata)

print("Computing network jacobians")
net_jacobians = get_net_jacobians(net, net_layers, eval_data_loader, neurons_per_layer)
print("Computing feature jacobians")
feature_jacobians = get_feature_jacobians(color_net, color_net_layers, shape_net, shape_net_layers, eval_data_loader)
print("Computing consistencies")
#consistencies, projections = get_all_consistencies(net_jacobians, feature_jacobians)
consistencies = compare_all_consistencies(net_jacobians, feature_jacobians)
fig = make_violinplot(consistencies)
fig.savefig("figures/experiment_4a/gradient_sparsity_epoch1.png", bbox_inches="tight")
fig, fig2 = make_imshow_plot(consistencies)
fig.savefig("figures/experiment_4a/gradient_consistency_imgs_epoch1.png", bbox_inches="tight")
fig2.savefig("figures/experiment_4a/gradient_consistency_variance_epoch1.png", bbox_inches="tight")
#fig = plot_projection_lengths(projections)
#fig.savefig("figures/experiment_4a/projection_lengths_epoch1.png", bbox_inches="tight")
del net_jacobians, feature_jacobians, consistencies#, projections

# train longer and repeat
print("Training for another 9 epochs")
trainer.fit_loop.max_epochs = 10
trainer.fit(net, traindata)

print("Computing network jacobians")
net_jacobians = get_net_jacobians(net, net_layers, eval_data_loader, neurons_per_layer)
print("Computing feature jacobians")
feature_jacobians = get_feature_jacobians(color_net, color_net_layers, shape_net, shape_net_layers, eval_data_loader)
print("Computing consistencies")
#consistencies, projections = get_all_consistencies(net_jacobians, feature_jacobians)
consistencies = compare_all_consistencies(net_jacobians, feature_jacobians)
fig = make_violinplot(consistencies)
fig.savefig("figures/experiment_4a/gradient_sparsity_epoch10.png", bbox_inches="tight")
fig, fig2 = make_imshow_plot(consistencies)
fig.savefig("figures/experiment_4a/gradient_consistency_imgs_epoch10.png", bbox_inches="tight")
fig2.savefig("figures/experiment_4a/gradient_consistency_variance_epoch10.png", bbox_inches="tight")
#fig = plot_projection_lengths(projections)
#fig.savefig("figures/experiment_4a/projection_lengths_epoch_10.png", bbox_inches="tight")