# Defines different color classes used to generate artificial training stimuli.
import torch
from torch import Tensor

class Color:
    "Base class"
    def __init__(self):
        self.color = self.colorval()

    def colorval(self) -> Tensor:
        raise NotImplementedError
    
    def fill_tensor(self, height, width) -> Tensor:
        img = torch.zeros((3, height, width))
        return img + self.color.reshape(3,1,1)
    
    def rgb_tuple(self) -> tuple[int, int, int]:
        c = self.colorval().squeeze() * 255
        c = c.to(torch.int32)
        return c[0], c[1], c[2]

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
        super().__init__()

    def colorval(self) -> Tensor:
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
        super().__init__()

    def colorval(self) -> Tensor:
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
        super().__init__()

    def colorval(self) -> Tensor:
        c = torch.zeros(1, 1, 3)
        c[0, 0, 2] = torch.rand(1) * (self.max - self.min) + self.min
        return c

# White / grey versions of SingleColors for testing generalization
class White(SingleColor):
    def __init__(self):
        super().__init__()

    def colorval(self) -> Tensor:
        return torch.ones(1, 1, 3)
    
class Grey(SingleColor):
    def __init__(self, brightness=0.5):
        self.brightness = brightness
        super().__init__()

    def colorval(self) -> Tensor:
        return self.brightness * torch.ones(1, 1, 3)
    
class GreySingleChannel(SingleColor):
    def __init__(self, brightness=0.5):
        self.brightness = brightness
        super().__init__()

    def colorval(self) -> Tensor:
        return self.brightness * torch.ones(1, 1, 1)
    
    def fill_tensor(self, height, width) -> Tensor:
        img = torch.zeros((1, height, width))
        return img + self.colorval()
    
class RandomGrey(SingleColor):
    def __init__(self, min=0.2, max=0.8):
        self.min = min
        self.max = max
        super().__init__()

    def colorval(self) -> Tensor:
        c = torch.rand(1).repeat(1, 1, 3) * (self.max - self.min) + self.min
        return c
    
class RandomGreySingleChannel(SingleColor):
    def __init__(self, min=0.2, max=0.8):
        self.min = min
        self.max = max
        super().__init__()

    def colorval(self) -> Tensor:
        c = torch.rand(1, 1, 1) * (self.max - self.min) + self.min
        return c
    
    def fill_tensor(self, height, width) -> Tensor:
        img = torch.zeros((1, height, width))
        return img + self.colorval()
    

##################################################################
# Color class set 2: two classes that are not linearly seperable #
##################################################################
class RedXORBlue(Color):
    """Generates random colors for which either red is between
    `min` and `max` and blue is between `1-max` and `1-min` or
    vice versa.
    """
    def __init__(self, min=0.9, max=1.0) -> None:
        self.min = min
        self.max = max
        super().__init__()

    def colorval(self) -> Tensor:
        c = torch.rand(3)
        if c[0] > c[2]: # red and not blue
            c[0] = c[0] * (self.max - self.min) + self.min
            c[2] = 1.0 - (c[2] * (self.max - self.min) + self.min)
        else: # blue and not red
            c[2] = c[2] * (self.max - self.min) + self.min
            c[0] = 1.0 - (c[0] * (self.max - self.min) + self.min)
        return c.reshape(1, 1, 3)
    
class NotRedXORBlue(Color):
    """Generates random colors for which red and blue are either
    both between `min` and `max` or both between `1-max` and `1-min`.
    """
    def __init__(self, min=0.9, max=1.0) -> None:
        self.min = min
        self.max = max
        super().__init__()

    def colorval(self) -> Tensor:
        c = torch.rand(3)
        if c[1] < 0.5: # red and blue both "on"; conditioning on green channel prevents very low intensities
            c[0] = c[0] * (self.max - self.min) + self.min
            c[2] = c[2] * (self.max - self.min) + self.min
        else: # red and blue both "off"
            c[0] = 1.0 - (c[0] * (self.max - self.min) + self.min)
            c[2] = 1.0 - (c[2] * (self.max - self.min) + self.min)
        return c.reshape(1, 1, 3)

####################################
# Color class set 3: R=B, B=G, G=R #
####################################
class RedEqBlue(Color):
    """Generates random color values with the constraint that red and blue channels are equal. Green is 1 - red."""
    def __init__(self):
        super().__init__()

    def colorval(self) -> Tensor:
        c = torch.rand(3)
        c[1] = c[0]
        c[2] = 1 - c[0]
        return c.reshape(1,1,3)
    
class BlueEqGreen(Color):
    """Generates random color values with the constraint that blue and green channels are equal. Red is 1 - blue."""
    def __init__(self):
        super().__init__()

    def colorval(self) -> Tensor:
        c = torch.rand(3)
        c[2] = c[1]
        c[0] = 1 - c[1]
        return c.reshape(1,1,3)
    
class GreenEqRed(Color):
    """Generates random color values with the constraint that green and red channels are equal. Blue is 1 - green."""
    def __init__(self):
        super().__init__()

    def colorval(self) -> Tensor:
        c = torch.rand(3)
        c[0] = c[2]
        c[1] = 1 - c[2]
        return c.reshape(1,1,3)
    
########################################
# Color class that combines two others #
########################################
class RandomChoiceColor(Color):
    """Randomly chooses one of the two colors that were passed to it."""
    def __init__(self, class1: Color = RedXORBlue, class2: Color = NotRedXORBlue):
        if torch.rand(1).item() < 0.5:
            self.__color = class1()
        else:
            self.__color = class2()
        super().__init__()

    def colorval(self):
        return self.__color.colorval()