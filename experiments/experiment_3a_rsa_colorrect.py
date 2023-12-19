import os
import pandas as pd
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import rsatoolbox
import seaborn as sns
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.analysis.rsa import get_shape_RDMs, get_color_RDMs, get_model_RDMs
from shapelearningtheory.analysis.plots import plot_rdm_grid
from shapelearningtheory.datasets import make_dataset
from shapelearningtheory.networks import make_convnet_small, ColorConvNet, SRectangleConvNet

# Load dataset
traindata = make_dataset("rectangles", "color", "small", "standard")
traindata.prepare_data()
testdata = make_dataset("rectangles", "color", "small", "random")
testdata.prepare_data()
imgheight = traindata.train.imgheight
imgwidth = traindata.train.imgwidth
channels = 3
classes = 2

# Generate RDMs for shape and color features
print("Getting shape RDMs")
shape_rdms = get_shape_RDMs(testdata.val)
print("Getting colour RDMs")
color_rdms = get_color_RDMs(testdata.val)

# Load constructed models and generate RDMs for them
color_net = ColorConvNet()
shape_net = SRectangleConvNet()
print("Getting feature network RDMs")
colornet_rdms = get_model_RDMs(color_net, testdata.test_dataloader(), use_image=False)
colornet_rdms.dissimilarity_measure = "Euclidean"
colornet_rdms.rdm_descriptors["property"] = colornet_rdms.rdm_descriptors["layer"]
colornet_rdms.rdm_descriptors["feature_type"] = ["colornet"] * len(colornet_rdms.rdm_descriptors["layer"])
shapenet_rdms = get_model_RDMs(shape_net, testdata.test_dataloader(), use_image=False)
shapenet_rdms.dissimilarity_measure = "Euclidean"
shapenet_rdms.rdm_descriptors["property"] = shapenet_rdms.rdm_descriptors["layer"]
shapenet_rdms.rdm_descriptors["feature_type"] = ["shapenet"] * len(shapenet_rdms.rdm_descriptors["layer"])
del color_net, shape_net

# Combine RDMs into one object and plot them
feature_rdms_list = [shape_rdms, color_rdms, colornet_rdms, shapenet_rdms]
feature_rdms = rsatoolbox.rdm.concat(*feature_rdms_list)
#print("Plotting")
figpath = "figures/experiment_3a/" 
os.makedirs(figpath, exist_ok=True)
#fig = plot_rdm_grid(feature_rdms_list)
#fig.savefig(figpath + "feature_rdms.png", bbox_inches="tight")
#plt.clf()
del feature_rdms_list, color_rdms, shape_rdms, colornet_rdms, shapenet_rdms

# Set up the model
print("Setting up model")
net = make_convnet_small(channels=channels, classes=classes)

# Generate RDMs for untrained model
print("Plotting RSA for untrained model")
def perform_rsa(model, data, feature_rdms, descriptor="untrained"):
    model_rdms = get_model_RDMs(model, data.test_dataloader())
    rsatoolbox.vis.rdm_plot.show_rdm(model_rdms, rdm_descriptor="layer")
    plt.savefig(figpath + "model_rdms_{}.png".format(descriptor))
    plt.clf()

    comparison = {
        "similarity": [],
        "layer": [],
        "property": [],
        "feature_type": []
    }
    for model_rdm in model_rdms:
        for feature_rdm in feature_rdms:
            sim = rsatoolbox.rdm.compare(model_rdm, feature_rdm, method="rho-a")
            comparison["similarity"].append(sim.item())
            comparison["layer"].append(model_rdm.rdm_descriptors["layer"][0])
            comparison["property"].append(feature_rdm.rdm_descriptors["property"][0])
            comparison["feature_type"].append(feature_rdm.rdm_descriptors["feature_type"][0])
    comparison = pd.DataFrame(comparison)
    feature_types = comparison["feature_type"].unique()
    fig, ax = plt.subplots(len(feature_types), 2,
        gridspec_kw={"width_ratios": [4, 1]},
        sharex=True,
        figsize=(7, 10))
    palettes = ["Blues", "Greens", "Reds", "YlOrBr"]
    for row, feature_type in enumerate(feature_types):
        subset = comparison.loc[comparison["feature_type"]==feature_type]
        palette = sns.color_palette(palettes[row],
            n_colors=len(subset["property"].unique()))
        sns.barplot(subset, x="layer", y="similarity", hue="property",
                    ax=ax[row][0], palette=palette)
        ax[row][0].set_ylim(-0.1, 1)
        h, l = ax[row][0].get_legend_handles_labels()
        ax[row][1].legend(h, l, borderaxespad=0)
        ax[row][1].axis("off")
        ax[row][0].get_legend().remove()
    fig.suptitle("RSA - network {}".format(descriptor))
    plt.savefig(figpath + "comparison_{}.png".format(descriptor))
perform_rsa(net, testdata, feature_rdms, "untrained")

# Train model and compare RDMs at several points during training
print("Training")
trainer = pl.Trainer(max_epochs=10, accelerator="gpu", logger=False,
    enable_checkpointing=False)
trainer.fit(net, traindata)
print("Performing RSA after 10 epochs")
perform_rsa(net, testdata, feature_rdms, "epoch10")

trainer.fit_loop.max_epochs = 20
trainer.fit(net, traindata)
print("Performing RSA after 20 epochs")
perform_rsa(net, testdata, feature_rdms, "epoch20")