from math import sin, cos
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

class SineGrating(Texture):
    """
    Base class for sine gratings. Not probabilistic, so it should not be used as a
    pattern class.
    """
    def __init__(self, frequency, orientation, phase):
        self.frequency = frequency
        self.orientation = orientation
        self.phase = phase

    def value_at_coordinate(self, x, y, channels=3):
        x_prime = x * cos(self.orientation) + y * sin(self.orientation)
        value = torch.sin(2 * torch.pi * self.frequency * x_prime + self.phase)
        return value
    
class HorizontalGrating(SineGrating):
    """Horizontal grating with somewhat randomized frequency, phase, and orientation."""
    def __init__(self, frequency_min=0.1, frequency_max=0.3, orientation_range=torch.pi / 5):
        frequency = torch.rand(1).item() * (frequency_max - frequency_min) + frequency_min
        orientation = torch.rand(1).item() * (2 * orientation_range) - orientation_range
        phase = torch.rand(1).item() / frequency
        super().__init__(frequency, orientation, phase)

class VerticalGrating(SineGrating):
    """Horizontal grating with somewhat randomized frequency, phase, and orientation."""
    def __init__(self, frequency_min=0.1, frequency_max=0.3, orientation_range=torch.pi / 5):
        frequency = torch.rand(1).item() * (frequency_max - frequency_min) + frequency_min
        orientation = torch.pi / 2 + torch.rand(1).item() * (2 * orientation_range) - orientation_range
        phase = torch.rand(1).item() / frequency
        super().__init__(frequency, orientation, phase)