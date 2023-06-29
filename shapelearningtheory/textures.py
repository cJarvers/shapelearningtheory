import torch
from torch import Tensor

class Texture:
    """
    Base class for textures.
    """
    def value_at_coordinate(self, x, y):
        raise NotImplementedError()

    def fill_tensor(self, height, width) -> Tensor:
        xs = torch.arange(height)
        ys = torch.arange(width)
        grid_x, grid_y = torch.meshgrid([xs, ys])
        pattern = self.value_at_coordinate(grid_x, grid_y)
        return pattern

class SineGrating:
    """
    
    """
    def __init__(self, frequency, orientation, phase):
        self.frequency = frequency
        self.orientation = orientation
        self.phase = phase