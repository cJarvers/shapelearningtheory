from .autoencoder import AutoEncoder
from .convnet import SimpleConvNet, RecurrentConvNet
from .linearnetworks import ShallowLinear, DeepLinear
from .mlp import MLP
from .softmaxnet import SpatialSoftmax2d, SoftmaxConvNet
from .solutionnetworks import make_color_mlp, make_color_convnet, make_rectangle_convnet
from .transformer import VisionTransformer


###########################
# Standard configurations #
###########################
import torch
from torchmetrics import Accuracy
def make_mlp_small(num_inputs, num_outputs):
    """Standard MLP configuration for training on small datasets."""
    mlp = MLP(num_inputs=num_inputs,
        num_hidden=1024,
        num_layers=3,
        num_outputs=num_outputs,
        loss_fun=torch.nn.functional.cross_entropy, 
        metric=Accuracy("multiclass", num_classes=num_outputs)
    )
    return mlp

def make_convnet_small(channels, classes):
    """Standard ConvNet configuration for training on small datasets."""
    convnet = SimpleConvNet(
        in_channels=channels,
        channels_per_layer=[16, 32, 64],
        kernel_sizes=[3,3,3],
        out_units=classes,
        loss_fun=torch.nn.functional.cross_entropy, 
        metric=Accuracy("multiclass", num_classes=2)
    )
    return convnet

def make_rconvnet_small(channels, classes):
    """Standard recurrent convnet configuration for training on small datasets."""
    rconvnet = RecurrentConvNet(
        in_channels=channels,
        out_units=classes,
        channels_per_layer=[16, 32, 64],
        kernel_sizes=[3,3,3],
        num_steps=10,
        loss_fun=torch.nn.functional.cross_entropy, 
        metric=Accuracy("multiclass", num_classes=2)
    )
    return rconvnet

def make_softmaxconv_small(channels, classes):
    """Standard configuration for a small convnet with spatial softmax blocks
    for training on small datasets."""
    softmaxconv = SoftmaxConvNet(
        in_channels=channels,
        out_units=classes,
        channels_per_layer=[16, 32, 64],
        kernel_sizes=[3,3,3],
        softmax_sizes=[7,7,7],
        version="cscl",
        loss_fun=torch.nn.functional.cross_entropy, 
        metric=Accuracy("multiclass", num_classes=2)
    )
    return softmaxconv

def make_ViT_small(imgsize, classes):
    """Standard Vision Transformer configuration for training on small datasets."""
    vit = VisionTransformer(
        image_size=imgsize,
        num_classes=classes,
        patch_size=3,
        num_layers=12,
        num_heads=8,
        hidden_dim=128,
        mlp_dim=1024,
        loss_fun=torch.nn.functional.cross_entropy, 
        metric=Accuracy("multiclass", num_classes=2)
    )
    return vit

def make_AE_small(num_inputs, classes):
    """Standard autoencoder configuration for training on small datasets."""
    autoencoder = AutoEncoder(
        input_dim=num_inputs,
        num_classes=classes,
        hidden_dims=[1024] * 3,
        representation_dim=500
    )
    return autoencoder