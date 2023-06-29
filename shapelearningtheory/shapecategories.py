# Defines different classes of artificial shapes used to generate training stimuli.
from enum import Enum
import torch
from torch import Tensor
from typing import List

from .colorcategories import Color

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

##############
# Base class #
##############
class Shape:
    def all_pixels(self) -> List[Pixel]:
        "Return a list of all Pixels that the shape covers"
        raise(NotImplementedError)
    
    def generate_mask(self, height: int, width: int, wrap: bool=True) -> Tensor:
        """Generate a bool tensor that marks all pixels that belong to the shape.
        If the shape is larger than the given height and width, `wrap` controls whether
        the shape just ends at the image border (wrap == False) or whether the image is
        treated as a torus and the shape wraps around (wrap == True).
        This is an inefficient base implementation. Overload for better performance."""
        mask = torch.zeros((height, width), dtype=torch.bool)
        for p in self.all_pixels():
            if 0 <= p.x < height and 0 <= p.y < width:
                mask[p.x, p.y] = True
            elif wrap:
                mask[p.x % height, p.y % width] = True
        return mask



#################
# Simple shapes #
#################
class Line(Shape):
    """A horizontal or vertical line, 1 pixel wide.
    
    Args:
        - start: Pixel - start location of the line (smallest x coordinate for
            horizontal lines, smallest y coordinate for vertical lines)
        - length: int - length of lines (number of pixles)
        - orientation: Orientation - horizontal or vertical
    """
    def __init__(self, start: Pixel, length: int, orientation: Orientation) -> None:
        super().__init__()
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
        
    def draw_to_tensor(self, t: Tensor, wrap: bool=True) -> Tensor:
        if self.orientation == Orientation.HORIZONTAL:
            x_min = self.start.x
            x_max = self.start.x + self.length
            y = self.start.y
            t[x_min:x_max, y] = 1.0
            if wrap and x_max > t.size(0):
                t[0:x_max - t.size(0), y] = 1.0
        else:
            y_min = self.start.y
            y_max = self.start.y + self.length
            x = self.start.x
            t[x, y_min:y_max] = 1.0
            if wrap and y_max > t.size(1):
                t[x, 0:y_max - t.size(1)] = 1.0
        return t


class Rectangle(Shape):
    """
    A horizontal or vertical rectangle.
    
    Args:
        start: pixel value of the bottom left corner
        length: number of pixels along the longer side"""
    def __init__(self, start: Pixel, length: int, width: int, orientation: Orientation):
        super().__init__()
        self.start = start
        assert length > width, "Length of a rectangle must be larger than width"
        self.length = length
        self.width = width
        self.orientation = orientation
        self.color = 1.0

    def _get_x_y(self):
        x_min, y_min = self.start.x, self.start.y
        if self.orientation == Orientation.HORIZONTAL:
            x_max = x_min + self.length
            y_max = y_min + self.width
        else:
            x_max = x_min + self.width
            y_max = y_min + self.length
        return x_min, x_max, y_min, y_max
    
    def all_pixels(self):
        x_min, x_max, y_min, y_max = self._get_x_y()
        return [Pixel(x, y)
            for x in range(x_min, x_max)
            for y in range(y_min, y_max)]
    
    def draw_to_tensor(self, t: Tensor, wrap: bool=True) -> Tensor:
        x_min, x_max, y_min, y_max = self._get_x_y()
        t[x_min:x_max, y_min:y_max] = self.color
        if wrap:
            if x_max > t.size(0):
                t[0:x_max - t.size(0), y_min:y_max] = self.color
            if y_max > t.size(1):
                t[x_min:x_max, 0:y_max - t.size(1)] = self.color
            if x_max > t.size(0) and y_max > t.size(1):
                t[0:x_max - t.size(0), 0:y_max - t.size(1)] = self.color


class Square(Rectangle):
    def __init__(self, start: Pixel, sidelength: int):
        self.start = start
        self.length = sidelength
        self.width = sidelength
        self.orientation = Orientation.HORIZONTAL # for compatibility with parent class, but irrelevant
