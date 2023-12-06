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
from shapelearningtheory.analysis.helpers import zip_dicts_cartesian, apply_with_torch, Table
from shapelearningtheory.analysis.plots import plot_column_wise, plot_table
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
imgheight = eval_data.val.imgheight
imgwidth = eval_data.val.imgwidth
channels = 3
classes = 2

# build networks
print("Building networks")
color_net = ColorConvNet(imageheight=imgheight, imagewidth=imgwidth)
shape_net = CRectangleConvNet()
net = make_convnet_small(channels=channels, classes=classes)
p = net(next(iter(traindata.train_dataloader()))[0]) # call once to initialize lazy modules
del p

# calculate gradient alignment before training
max_batches = 200
neurons_per_layer = 2000
    
@torch.no_grad()
def compare_all_consistencies(net_jacobians, feature_jacobians):
    f = lambda net_grad, feature_grad: apply_with_torch(
            compare_consistencies, [net_grad, feature_grad])
    consistencies = zip_dicts_cartesian(f, net_jacobians, feature_jacobians)
    return consistencies

def sparsity(x): # TODO: square the denominator (after the mean)?
    return np.abs(x).mean(axis=(1,2)) / np.square(x).mean(axis=(1,2))

def make_violinplot(consistencies: Table):
    plotfun = lambda row, axis: sns.violinplot(row, ax=axis)
    sparsities = consistencies.apply_to_cells(sparsity) 
    fig = plot_column_wise(sparsities, plotfun,
        fig_params={"figsize": (8, 12)},
    )
    return fig

def plot_mean_consistencies(consistencies: Table):
    means = consistencies.apply_to_cells(lambda x: x.mean(axis=0))
    # variances = consistencies.apply_to_cells(lambda x: x.var(axis=0))
    minval = min(means.map_over_cells(np.min))
    maxval = max(means.map_over_cells(np.max))
    plotfun = lambda x, ax: ax.imshow(x, cmap="viridis", vmin=minval, vmax=maxval, interpolation=None)
    fig = plot_table(means, plotfun, fig_params={"figsize": (28, 16)})
    return fig


print("Computing network jacobians")
net_jacobians = get_jacobians_per_layer(net, eval_data_loader, neurons_per_layer)
print("Computing feature jacobians")
feature_jacobians = get_jacobians_per_layer(color_net, eval_data_loader)
feature_jacobians.update(
    get_jacobians_per_layer(shape_net, eval_data_loader)
)
print("Computing consistencies")
consistencies = compare_all_consistencies(net_jacobians, feature_jacobians)
os.makedirs("figures/experiment_4a", exist_ok=True)
fig = make_violinplot(consistencies)
fig.savefig("figures/experiment_4a/gradient_sparsity.png", bbox_inches="tight")
fig = plot_mean_consistencies(consistencies)
fig.savefig("figures/experiment_4a/gradient_consistency_imgs.png", bbox_inches="tight")
del net_jacobians, feature_jacobians, consistencies

# train for an epoch and plot again
print("Training network for 1 epoch")
trainer = pl.Trainer(max_epochs=1, accelerator="gpu", logger=False,
    enable_checkpointing=False)
trainer.fit(net, traindata)

print("Computing network jacobians")
net_jacobians = get_jacobians_per_layer(net, eval_data_loader, neurons_per_layer)
print("Computing feature jacobians")
feature_jacobians = get_jacobians_per_layer(color_net, eval_data_loader)
feature_jacobians.update(
    get_jacobians_per_layer(shape_net, eval_data_loader)
)
print("Computing consistencies")
consistencies = compare_all_consistencies(net_jacobians, feature_jacobians)
fig = make_violinplot(consistencies)
fig.savefig("figures/experiment_4a/gradient_sparsity_epoch1.png", bbox_inches="tight")
fig = plot_mean_consistencies(consistencies)
fig.savefig("figures/experiment_4a/gradient_consistency_imgs_epoch1.png", bbox_inches="tight")
del net_jacobians, feature_jacobians, consistencies

# train longer and repeat
print("Training for another 9 epochs")
trainer.fit_loop.max_epochs = 10
trainer.fit(net, traindata)

print("Computing network jacobians")
net_jacobians = get_jacobians_per_layer(net, eval_data_loader, neurons_per_layer)
print("Computing feature jacobians")
feature_jacobians = get_jacobians_per_layer(color_net, eval_data_loader)
feature_jacobians.update(
    get_jacobians_per_layer(shape_net, eval_data_loader)
)
print("Computing consistencies")
consistencies = compare_all_consistencies(net_jacobians, feature_jacobians)
fig = make_violinplot(consistencies)
fig.savefig("figures/experiment_4a/gradient_sparsity_epoch10.png", bbox_inches="tight")
fig = plot_mean_consistencies(consistencies)
fig.savefig("figures/experiment_4a/gradient_consistency_imgs_epoch10.png", bbox_inches="tight")