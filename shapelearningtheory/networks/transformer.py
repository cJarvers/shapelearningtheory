import torch
import torchvision
from typing import Tuple, Callable, Any
from .trainingwrapper import TrainingWrapper

class VisionTransformer(TrainingWrapper):
    "Wrapper around torchvisions VisionTransformer"
    
    def __init__(self, image_size: int, patch_size: int, num_layers:int,
            num_heads: int, hidden_dim: int, mlp_dim: int, num_classes: int,
            loss_fun: Callable, metric: Callable,
            lr: float=0.01, weight_decay: float=1e-4, momentum: float=0.9,
            gamma: float=0.99):
        # generate layers:
        layers = torchvision.models.VisionTransformer(
            image_size=image_size, patch_size=patch_size, num_layers=num_layers,
            num_heads=num_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim,
            num_classes=num_classes
        )
        super().__init__(net=layers, loss_fun=loss_fun, metric=metric, lr=lr,
                         weight_decay=weight_decay, momentum=momentum, gamma=gamma)
    
    def layer_activations(self, x):
        outputs = {'image': x}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, torch.nn.Linear):
                outputs["layer {}".format(i)] = x
        return outputs