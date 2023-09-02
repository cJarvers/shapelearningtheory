import torch
from torch.func import vmap, jacrev

def get_activations(net, x):
    """
    Get activations of `net` for input `x`.
    Extracts outputs of Linear and Conv2d layers.
    Assumes that iterating over `net` returns layers in correct order.
    """
    outputs = {'image': x}
    for i, layer in enumerate(net):
        x = layer(x)
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            outputs["layer {}".format(i)] = x
    return outputs


def compute_jacobian(net, images: torch.Tensor, **kwargs):
    """Compute the jacobian of `net` on `images`.
    For N images with P pixels each and M neurons in the output layer of `net`,
    this will result in an N x M x P tensor, where the vector [n, m, :] is the
    gradient of neuron m for image n. That is, using this vector to perform
    gradient ascent in image n should increase the activation of neuron m.
    
    Additional kwargs are passed on to vmap."""
    flat = images.flatten(start_dim=1)
    def reshape_and_apply(x):
        x = x.reshape((1, *images.size()[1:]))
        out = net(x)
        return out.flatten()
    is_training = net.training
    net.eval() # BatchNorm has to be in eval mode, otherwise cannot compute jacobian due to in-place update
    jacobian = vmap(jacrev(reshape_and_apply), **kwargs)(flat)
    if is_training:
        net.train()
    return jacobian