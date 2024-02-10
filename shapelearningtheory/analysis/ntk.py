import torch
from torch.func import functional_call, vmap, jacrev

# adapted from https://pytorch.org/tutorials/intermediate/neural_tangent_kernels.html (2023-12-28)
def get_parameters(net: torch.nn.Module):
    return {k: v.detach() for k, v in net.named_parameters()}

def functionalize(net, neuron_indices=None):
    def net_fun(params, x, neuron_indices=neuron_indices):
        y = functional_call(net, params, (x.unsqueeze(0),)).squeeze(0)
        if neuron_indices is not None:
            y = y.flatten()[neuron_indices]
        return y
    return net_fun

def empirical_ntk(net_fun, params, xs):
    # compute jacobians
    jacobians =  vmap(jacrev(net_fun), (None, 0))(params, xs)
    jacobians = [j.flatten(2) for j in jacobians.values()]
    # compute matrix product of jacobians J: J(xs) @ J(xs).T
    kernel = torch.stack([torch.einsum('Naf,Maf->NM', j, j) for j in jacobians])
    kernel = kernel.sum(0)
    return kernel

def get_ntk(net: torch.nn.Module, data: torch.Tensor):
    ntk = empirical_ntk(functionalize(net), get_parameters(net), data)
    return ntk

def get_ntk_similarity(ntk1, ntk2):
    ntk1 = ntk1.flatten().unsqueeze(0)
    ntk2 = ntk2.flatten().unsqueeze(0)
    similarity = torch.cosine_similarity(ntk1, ntk2).item()
    return similarity