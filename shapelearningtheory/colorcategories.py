# Defines different color classes used to generate artificial training stimuli.
import torch

class Color:
    "Base class"
    def colorval(self) -> torch.Tensor:
        raise NotImplementedError

################################################
# Color class set 1: reds vs. greens vs. blues #
################################################
class SingleColor(Color):
    pass

class RandomRed(SingleColor):
    """
    Generates a random red color. Only the R channel (out of RGB) will have a
    non-zero value. The precise value is sampled between `min` and `max`.
    """
    def __init__(self, min=0.5, max=1.0):
        self.min = min
        self.max = max

    def colorval(self) -> torch.Tensor:
        c = torch.zeros(1, 1, 3)
        c[0, 0, 0] = torch.rand(1) * (self.max - self.min) + self.min
        return c

class RandomGreen(SingleColor):
    """
    Generates a random green color. Only the G channel (out of RGB) will have a
    non-zero value. The precise value is sampled between `min` and `max`.
    """
    def __init__(self, min=0.5, max=1.0):
        self.min = min
        self.max = max

    def colorval(self) -> torch.Tensor:
        c = torch.zeros(1, 1, 3)
        c[0, 0, 1] = torch.rand(1) * (self.max - self.min) + self.min
        return c

class RandomBlue(SingleColor):
    """
    Generates a random blue color. Only the B channel (out of RGB) will have a
    non-zero value. The precise value is sampled between `min` and `max`.
    """
    def __init__(self, min=0.5, max=1.0):
        self.min = min
        self.max = max

    def colorval(self) -> torch.Tensor:
        c = torch.zeros(1, 1, 3)
        c[0, 0, 2] = torch.rand(1) * (self.max - self.min) + self.min
        return c

# White / grey versions of SingleColors for testing generalization
class White(SingleColor):
    def colorval(self) -> torch.Tensor:
        return torch.ones(1, 1, 3)
    
class RandomGrey(SingleColor):
    def __init__(self, min=0.5, max=1.0):
        self.min = min
        self.max = max

    def colorval(self) -> torch.Tensor:
        c = torch.rand(1).repeat(1, 1, 3) * (self.max - self.min) + self.min
        return c