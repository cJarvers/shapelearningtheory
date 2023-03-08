# Defines different classes of artificial shapes used to generate training stimuli.
from enum import Enum
import torch

######################################################
# Image primitives from which shapes are constructed #
######################################################
class Pixel:
    "Class to express image locations."
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class Orientation(Enum):
    HORIZONTAL = 1
    VERTICAL = 2

class Line:
    """A horizontal or vertical line, 1 pixel wide.
    
    Args:
        - start: Pixel - start location of the line (smallest x coordinate for
            horizontal lines, smallest y coordinate for vertical lines)
        - length: int - length of lines (number of pixles)
        - orientation: Orientation - horizontal or vertical
    """
    def __init__(self, start: Pixel, length: int, orientation: Orientation) -> None:
        self.start = start
        self.length = length
        self.orientation = orientation

    def all_pixels(self):
        if self.orientation == Orientation.HORIZONTAL:
            return (Pixel(x, self.start.y) for x in range(self.start.x, self.start.x + self.length))
        else:
            return (Pixel(self.start.x, y) for y in range(self.start.y, self.start.y + self.length))

    def last_pixel(self) -> Pixel:
        if self.orientation == Orientation.HORIZONTAL:
            return Pixel(self.start.x + self.length, self.start.y)
        else:
            return Pixel(self.start.x, self.start.y + self.length)
        
    def draw_to_tensor(self, t: torch.Tensor) -> torch.Tensor:
        if self.orientation == Orientation.HORIZONTAL:
            x_min = self.start.x
            x_max = self.start.x + self.length
            y = self.start.y
            t[x_min:x_max, y] = 1.0
        else:
            y_min = self.start.y
            y_max = self.start.y + self.length
            x = self.start.x
            t[x, y_min:y_max] = 1.0
        return t


##############################
# Shape classes / categories #
##############################
class Shape:
    """Base class for shape categories. Defines the interface."""
    

class ShapeCategory1(Shape):
    "Shapes made up of a rectangle and a line that cuts it."
    def __init__(self):
        pass