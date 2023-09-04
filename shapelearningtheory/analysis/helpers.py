import torch
from torch.func import vmap, jacrev
from torch.nn.functional import cosine_similarity

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
    For B images with P pixels each and M neurons in the output layer of `net`,
    this will result in an B x M x P tensor, where the vector [b, m, :] is the
    gradient of neuron m for image b. That is, using this vector to perform
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


def compute_gradient_alignment(net1, net2, images: torch.Tensor):
    """Computes the alignment between image gradients of all output neurons in
    `net1` to all output neurons in `net2`.""" # with B = batch size, P = number of pixels
    jacobian1 = compute_jacobian(net1, images) # size B x M x P,   M = number of output neurons of net1
    jacobian2 = compute_jacobian(net2, images) # size B x N x P,   N = number of output neurons of net2
    similarity = cosine_similarity(jacobian1.unsqueeze(1), jacobian2.unsqueeze(2), dim=3)
    return similarity                          # size B x M x N