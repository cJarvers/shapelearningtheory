from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg19_bn
from torchvision.models.swin_transformer import swin_t
from .autoencoder import AutoEncoder
from .convnet import SimpleConvNet, RecurrentConvNet
from .linearnetworks import ShallowLinear, DeepLinear
from .mlp import MLP
from .softmaxnet import SpatialSoftmax2d, SoftmaxConvNet
from .solutionnetworks import ColorConvNet, CRectangleConvNet, \
    SRectangleConvNet, TextureConvNet, LTConvNet
from .transformer import VisionTransformer
from .trainingwrapper import TrainingWrapper


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
        kernel_sizes=[5,5,5],
        out_units=classes,
        loss_fun=torch.nn.functional.cross_entropy, 
        metric=Accuracy("multiclass", num_classes=classes)
    )
    return convnet

def make_rconvnet_small(channels, classes):
    """Standard recurrent convnet configuration for training on small datasets."""
    rconvnet = RecurrentConvNet(
        in_channels=channels,
        out_units=classes,
        channels_per_layer=[16, 32, 64],
        kernel_sizes=[5,5,5],
        num_steps=10,
        loss_fun=torch.nn.functional.cross_entropy, 
        metric=Accuracy("multiclass", num_classes=classes)
    )
    return rconvnet

def make_softmaxconv_small(channels, classes):
    """Standard configuration for a small convnet with spatial softmax blocks
    for training on small datasets."""
    softmaxconv = SoftmaxConvNet(
        in_channels=channels,
        out_units=classes,
        channels_per_layer=[16, 32, 64],
        kernel_sizes=[5,5,5],
        softmax_sizes=[9,9,9],
        version="cscl",
        loss_fun=torch.nn.functional.cross_entropy, 
        metric=Accuracy("multiclass", num_classes=classes)
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
        metric=Accuracy("multiclass", num_classes=classes)
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

def make_resnet50(classes):
    """Standard config for ResNet50 to train on large datasets."""
    net = TrainingWrapper(
        net = resnet50(num_classes=classes),
        loss_fun=torch.nn.functional.cross_entropy, 
        metric=Accuracy("multiclass", num_classes=classes),
        lr=0.001
    )
    return net

def make_vit_b_16(imgsize, classes):
    vit = VisionTransformer(
        image_size=imgsize,
        num_classes=classes,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        loss_fun=torch.nn.functional.cross_entropy, 
        metric=Accuracy("multiclass", num_classes=classes),
        lr=0.001
    )
    return vit

def make_vgg_19(classes):
    vgg = TrainingWrapper(
        net = vgg19_bn(num_classes=classes),
        loss_fun=torch.nn.functional.cross_entropy,
        metric=Accuracy("multiclass", num_classes=classes),
        lr=0.001
    )
    return vgg

def make_swin_t(classes):
    return TrainingWrapper(
        net = swin_t(num_classes=classes),
        loss_fun=torch.nn.functional.cross_entropy,
        metric=Accuracy("multiclass", num_classes=classes),
        lr=0.001
    )