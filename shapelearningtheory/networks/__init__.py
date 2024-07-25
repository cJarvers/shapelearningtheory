from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg19_bn
from torchvision.models.vision_transformer import vit_b_16
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

def make_resnet50(classes, pretrained=False):
    """Standard config for ResNet50 to train on large datasets."""
    if pretrained:
        net = resnet50(weights="IMAGENET1K_V2")
        for param in net.parameters():
            param.requires_grad = False
        net.fc = torch.nn.Linear(2048, classes)
    else:
        net = resnet50(num_classes=classes)
    net = TrainingWrapper(
        net = net,
        loss_fun=torch.nn.functional.cross_entropy, 
        metric=Accuracy("multiclass", num_classes=classes),
        lr=0.001
    )
    return net

def make_vit_b_16(imgsize, classes, pretrained=False):
    if pretrained:
        net = vit_b_16(image_size=224, weights="IMAGENET1K_V1")
        net.image_size = imgsize
        net.conv_proj.stride = (8, 8) # Tuned for 112x112 images
        net.conv_proj.padding = (4, 4) # Tuned for 112x112 images
        net.patch_size = 8 # trick the transformer into accepting output of conv_proj
        net.heads.head = torch.nn.Linear(768, 2)
    else:
        net = vit_b_16(image_size=imgsize, num_classes=classes)
    vit = TrainingWrapper(
        net = net,
        loss_fun=torch.nn.functional.cross_entropy, 
        metric=Accuracy("multiclass", num_classes=classes),
        lr=0.001
    )
    return vit

def make_vgg_19(classes, pretrained=False):
    if pretrained:
        net = vgg19_bn(weights = "IMAGENET1K_V1")
        for param in net.parameters():
            param.requires_grad = False
        net.classifier[6] = torch.nn.Linear(4096, classes)
    else:
        net = vgg19_bn(num_classes=classes)
    vgg = TrainingWrapper(
        net = net,
        loss_fun=torch.nn.functional.cross_entropy,
        metric=Accuracy("multiclass", num_classes=classes),
        lr=0.001
    )
    return vgg

def make_swin_t(classes, pretrained=False):
    if pretrained:
        net = swin_t(weights="IMAGENET1K_V1")
        net.head = torch.nn.Linear(768, 2)
    else:
        net = swin_t(num_classes=classes)
    return TrainingWrapper(
        net = net,
        loss_fun=torch.nn.functional.cross_entropy,
        metric=Accuracy("multiclass", num_classes=classes),
        lr=0.001
    )