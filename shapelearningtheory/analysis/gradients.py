# Helper functions for analyzing network gradients and comparing
# alignment between networks.
import numpy as np
import torch
from torch.func import vmap, jacrev
from torchmetrics.functional import pairwise_cosine_similarity

def compute_jacobian(net, images: torch.Tensor, flatten: bool=True, **kwargs):
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
        if flatten:
            return out.flatten()
        else:
            return out.squeeze(0)
    is_training = net.training
    net.eval() # BatchNorm has to be in eval mode, otherwise cannot compute jacobian due to in-place update
    jacobian = vmap(jacrev(reshape_and_apply), **kwargs)(flat)
    if is_training:
        net.train()
    return jacobian


def compute_gradient_alignment(net1, net2, images: torch.Tensor, subsample_neurons=None):
    """Computes the alignment between image gradients of all output neurons in
    `net1` to all output neurons in `net2`.""" # with B = batch size, P = number of pixels
    jacobian1 = compute_jacobian(net1, images) # size B x M x P,   M = number of output neurons of net1
    if subsample_neurons:
        m = jacobian1.size(1)
        inds = torch.randperm(m)[:subsample_neurons]
        jacobian1 = jacobian1[:, inds, :]
    jacobian2 = compute_jacobian(net2, images) # size B x N x P,   N = number of output neurons of net2
    similarity = vmap(pairwise_cosine_similarity)(
        jacobian1, jacobian2
    )
    return similarity                          # size B x M x N

@torch.no_grad()
def get_jacobians(net, dataset, max_batches=None, device="cuda", normalize=False):
    jacobian_list = []
    for i, (x, _) in enumerate(dataset):
        x = x.to(device)
        net = net.to(device)
        if max_batches and i >= max_batches:
            break
        jacobian = compute_jacobian(net, x, chunk_size=1)
        if normalize:
            jacobian = torch.linalg.vector_norm(jacobian, dim=2, keepdim=True)
        jacobian_list.append(
            jacobian.detach().cpu().numpy()
        )
    return np.concatenate(jacobian_list, axis=0)


def select_random_subset(jacobian: np.array, num_neurons: int):
    """From a given jacobian of shape B x N x D, with N neurons,
    select a random subset of num_neurons neurons"""
    N = jacobian.shape[1]
    inds = np.random.permutation(N)[:num_neurons]
    return jacobian[:, inds, :]

@torch.no_grad()
def find_most_similar_gradient(grad: torch.Tensor, feature_grad: torch.Tensor
                               ) -> torch.Tensor:
    """For each gradient in grad, find the most similar one in jac2.
    grad has dimensions N x D,
    feature_grad has dimension M x D.
    Comparison is done along the last dimension."""
    inds = pairwise_cosine_similarity(grad, feature_grad).argmax(dim=1)
    return feature_grad[inds]

@torch.no_grad()
def project_gradients_to_feature(grad: torch.Tensor, feature_grad: torch.Tensor
                                 ) -> torch.Tensor:
    alignment = pairwise_cosine_similarity(grad, feature_grad)
    alignment.nan_to_num_()
    result = torch.matmul(alignment, feature_grad)
    return result

@torch.no_grad()
def compare_consistencies(grads: torch.Tensor, feature_grads: torch.Tensor,
                          chunk_size: int=1) -> torch.Tensor:
    consistency_fun = vmap(pairwise_cosine_similarity, in_dims=1, chunk_size=chunk_size)
    grad_consistency = consistency_fun(grads)
    feature_projection = vmap(project_gradients_to_feature, chunk_size=chunk_size)(
        grads, feature_grads
    )
    feature_consistency = consistency_fun(feature_projection)
    return grad_consistency * feature_consistency