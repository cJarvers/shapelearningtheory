import torch
from torch.func import vmap, jacrev
from torch.nn.functional import cosine_similarity
from torchmetrics.functional import pairwise_cosine_similarity

def get_activations(net, x, use_image=True):
    """
    Get activations of `net` for input `x`.
    Extracts outputs of Linear and Conv2d layers.
    Assumes that iterating over `net` returns layers in correct order.
    """
    if use_image:
        outputs = {'image': x}
    else:
        outputs = {}
    layers = net.modules()
    layers = filter(lambda x: not isinstance(x, torch.nn.Sequential), layers)
    for i, layer in enumerate(layers):
        x = layer(x)
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            outputs["layer {}".format(i)] = x
    return outputs

def get_activations_dataset(net, data, use_image=True):
    all_activations = [get_activations(net, x, use_image=use_image) for (x, y) in data]
    keys = all_activations[0].keys()
    return {
        k: torch.concat([act[k] for act in all_activations], dim=0)
        for k in keys
    }


