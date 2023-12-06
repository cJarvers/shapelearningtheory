import torch
from typing import Callable, List

class Table:
    def __init__(self, col_names: List, row_names: List):
        self.col_names = col_names
        self.row_names = row_names
        self.content = {
            col_name: {
                row_name: None
                for row_name in row_names
            } for col_name in col_names
        }

    def __getitem__(self, indices):
        col, row = indices
        return self.content[col][row]
    
    def __setitem__(self, indices, value):
        col, row = indices
        self.content[col][row] = value
    
    def get_size(self):
        return len(self.col_names), len(self.row_names)
    
    def get_columns(self):
        return self.content.items()
    
    def apply_to_cells(self, f: Callable):
        result = Table(self.col_names, self.row_names)
        for col_name, column in self.content.items():
            for row_name, value in column.items():
                result[col_name, row_name] = f(value)
        return result

    def map_over_cells(self, f: Callable):
        return [f(self.content[col][row]) for col in self.col_names for row in self.row_names]

def apply_to_layers(f: Callable, net) -> dict:
    """Apply function f to each layer of interest of a network."""
    results = {}
    for layer, index in net.get_layers_of_interest().items():
        subnet = net[:index]
        results[layer] = f(subnet)
    return results

def zip_dicts_cartesian(f: Callable, dict1: dict, dict2: dict) -> Table:
    """Combine two dictionaries by applying function f to each pair
    (v1, v2) of values from dict1 and dict2, respectively. The results
    are returned in a nested dictionary."""
    results = Table(dict1.keys(), dict2.keys())
    for key1, val1 in dict1.items():
        for key2, val2 in dict2.items():
            results[key1, key2] = f(val1, val2)
    return results

@torch.no_grad()
def apply_with_torch(f, arrays):
    """Apply a function f to a list of numpy arrays, converting each to a
    torch.Tensor on a cuda device. Convert the result back to a numpy array."""
    result = f(*[torch.tensor(arr, device="cuda") for arr in arrays])
    return result.cpu().numpy()

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


