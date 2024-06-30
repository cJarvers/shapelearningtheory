# Defines different classes of artificial shapes used to generate training stimuli.
from enum import Enum
import torch
from torch import Tensor
from typing import List, Literal

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
        
    def generate_mask(self, height: int, width: int, wrap: bool = True) -> Tensor:
        mask = torch.zeros((height, width), dtype=torch.bool)
        if wrap: # if we want the image to wrap, we initialize the line at (0,0) and use torch.roll
            x_min = 0
            y_min = 0
        else:
            x_min = self.start.x
            y_min = self.start.y
        x_max = x_min + (self.length if self.orientation is Orientation.HORIZONTAL else 1)
        y_max = y_min + (self.length if self.orientation is Orientation.VERTICAL else 1)
        mask[x_min:x_max, y_min:y_max] = True
        if wrap:
            mask = mask.roll(shifts=(self.start.x, self.start.y), dims=(0,1))
        return mask


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
    
    def get_position(self):
        "Return x and y coordinates of center of mass of the rectangle."
        x_min, x_max, y_min, y_max = self._get_x_y()
        return (x_min + x_max) / 2, (y_min + y_max) / 2
    
    def all_pixels(self):
        x_min, x_max, y_min, y_max = self._get_x_y()
        return [Pixel(x, y)
            for x in range(x_min, x_max)
            for y in range(y_min, y_max)]
    
    def generate_mask(self, height: int, width: int, wrap: bool = True) -> Tensor:
        mask = torch.zeros((height, width), dtype=torch.bool)
        if wrap: # if we want the image to wrap, we initialize the line at (0,0) and use torch.roll
            x_min = 0
            y_min = 0
        else:
            x_min = self.start.x
            y_min = self.start.y
        x_max = x_min + (self.length if self.orientation is Orientation.HORIZONTAL else self.width)
        y_max = y_min + (self.length if self.orientation is Orientation.VERTICAL else self.width)
        mask[x_min:x_max, y_min:y_max] = True
        if wrap:
            mask = mask.roll(shifts=(self.start.x, self.start.y), dims=(0,1))
        return mask


class Square(Rectangle):
    def __init__(self, start: Pixel, sidelength: int):
        self.start = start
        self.length = sidelength
        self.width = sidelength
        self.orientation = Orientation.HORIZONTAL # for compatibility with parent class, but irrelevant


####################
# Composite shapes #
####################
class LShape(Shape):
    """Two lines that meet to form a corner.
    
    Args:
        start: Pixel - start location (top left pixel)
        height: int - length of the vertical line
        width: int - length of the horizontal line
        strength: int - how many pixels each line should be wide
        corner: Literal["topright", "topleft", "bottomright", "bottomleft"]
          - determines orientation (e.g., for "topright" the two lines meet at the top right;
            "bottomleft" corresponds to a normal L)
    """
    def __init__(self, start: Pixel, height: int, width: int, strength: int,
            corner: Literal["topright", "topleft", "bottomright", "bottomleft"]):
        super().__init__()
        if height <= strength or width <= strength:
            raise ValueError("Height / width of LShape is smaller than stroke strength; one stroke will not be visible.")
        self.start = start
        self.height = height
        self.width = width
        self.strength = strength
        self.corner = corner
        # generate the two bars that make up the L shape
        self.horizontalbar, self.verticalbar = self._calculate_bars()

    def _calculate_bars(self):
        if self.corner == "topright":
            x_horizontal = self.start.x
            y_horizontal = self.start.y
            x_vertical = self.start.x + self.width - self.strength
            y_vertical = self.start.y
        elif self.corner == "topleft":
            x_horizontal = self.start.x
            y_horizontal = self.start.y
            x_vertical = self.start.x
            y_vertical = self.start.y
        elif self.corner == "bottomright":
            x_horizontal = self.start.x
            y_horizontal = self.start.y + self.height - self.strength
            x_vertical = self.start.x + self.width - self.strength
            y_vertical = self.start.y
        elif self.corner == "bottomleft":
            x_horizontal = self.start.x
            y_horizontal = self.start.y + self.height - self.strength
            x_vertical = self.start.x
            y_vertical = self.start.y
        horizontalbar = Rectangle(
            start = Pixel(x_horizontal, y_horizontal),
            length = self.width,
            width = self.strength,
            orientation = Orientation.HORIZONTAL
        )
        verticalbar = Rectangle(
            start = Pixel(x_vertical, y_vertical),
            length = self.height,
            width = self.strength,
            orientation = Orientation.VERTICAL
        )
        return horizontalbar, verticalbar

    def generate_mask(self, height: int, width: int, wrap: bool = True) -> Tensor:
        horizontal_mask = self.horizontalbar.generate_mask(height, width, wrap)
        vertical_mask = self.verticalbar.generate_mask(height, width, wrap)
        return torch.logical_or(horizontal_mask, vertical_mask)
    
class TShape(Shape):
    """Two lines that meet to form a T-junction.
    
    Args:
        start: Pixel - start location (top left pixel)
        height: int - length of the vertical line
        width: int - length of the horizontal line
        strength: int - how many pixels each line should be wide
        topside: Literal["right", "left", "top", "left"]
          - determines orientation (e.g., for "top" the upper line of the T is at the top,
            resulting in a normal T)
    """
    def __init__(self, start: Pixel, height: int, width: int, strength: int,
            topside: Literal["right", "left", "top", "bottom"]):
        super().__init__()
        if height <= strength or width <= strength:
            raise ValueError("Height / width of TShape is smaller than stroke strength; one stroke will not be visible.")
        if height == strength + 1 or width == strength + 1:
            raise ValueError("Height / width of TShape is only 1 larger than stroke strength; will be indistinguishable from L.")
        self.start = start
        self.height = height
        self.width = width
        self.strength = strength
        self.topside = topside
        # generate the two bars that make up the T shape
        self.horizontalbar, self.verticalbar = self._calculate_bars()

    def _calculate_bars(self):
        if self.topside == "right":
            x_horizontal = self.start.x
            y_horizontal = self.start.y + self.height // 2 - self.strength // 2
            x_vertical = self.start.x
            y_vertical = self.start.y
        elif self.topside == "left":
            x_horizontal = self.start.x
            y_horizontal = self.start.y + self.height // 2 - self.strength // 2
            x_vertical = self.start.x + self.width - self.strength
            y_vertical = self.start.y
        elif self.topside == "top":
            x_horizontal = self.start.x
            y_horizontal = self.start.y
            x_vertical = self.start.x + self.width // 2 - self.strength // 2
            y_vertical = self.start.y
        elif self.topside == "bottom":
            x_horizontal = self.start.x
            y_horizontal = self.start.y + self.height - self.strength
            x_vertical = self.start.x + self.width // 2 - self.strength // 2
            y_vertical = self.start.y
        horizontalbar = Rectangle(
            start = Pixel(x_horizontal, y_horizontal),
            length = self.width,
            width = self.strength,
            orientation = Orientation.HORIZONTAL
        )
        verticalbar = Rectangle(
            start = Pixel(x_vertical, y_vertical),
            length = self.height,
            width = self.strength,
            orientation = Orientation.VERTICAL
        )
        return horizontalbar, verticalbar

    def generate_mask(self, height: int, width: int, wrap: bool = True) -> Tensor:
        horizontal_mask = self.horizontalbar.generate_mask(height, width, wrap)
        vertical_mask = self.verticalbar.generate_mask(height, width, wrap)
        return torch.logical_or(horizontal_mask, vertical_mask)
    
class PlusShape(Shape):
    """Two lines that cross each other.
    
    Args:
        start: Pixel - start location (top left pixel)
        height: int - length of the vertical line
        width: int - length of the horizontal line
        strength: int - how many pixels each line should be wide
        x_offset: int - offset applied to the vertical bar in x direction
        y_offset: int - offset applied to the horizontal bar in y direction
    """
    def __init__(self, start: Pixel, height: int, width: int, strength: int,
            x_offset: int = 0, y_offset: int = 0):
        super().__init__()
        if height <= strength or width <= strength:
            raise ValueError("Height / width of PlusShape is smaller than stroke strength; one stroke will not be visible.")
        if height == strength + 1 or width == strength + 1:
            raise ValueError("Height / width of PlusShape is only 1 larger than stroke strength; will be indistinguishable from L or T.")
        if abs(x_offset) >= (width - strength) // 2 or abs(y_offset) >= (height - strength) // 2:
            raise ValueError("Offset in a PlusShape cannot be larger than height or width - strength; otherwise, shape will become an L or T.")
        self.start = start
        self.height = height
        self.width = width
        self.strength = strength
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.horizontalbar, self.verticalbar = self._calculate_bars()

    def _calculate_bars(self):
        x_horizontal = self.start.x
        y_horizontal = self.start.y + self.height // 2 - self.strength // 2 + self.y_offset
        x_vertical = self.start.x + self.width // 2 - self.strength // 2 + self.x_offset
        y_vertical = self.start.y
        horizontalbar = Rectangle(
            start = Pixel(x_horizontal, y_horizontal),
            length = self.width,
            width = self.strength,
            orientation = Orientation.HORIZONTAL
        )
        verticalbar = Rectangle(
            start = Pixel(x_vertical, y_vertical),
            length = self.height,
            width = self.strength,
            orientation = Orientation.VERTICAL
        )
        return horizontalbar, verticalbar

    def generate_mask(self, height: int, width: int, wrap: bool = True) -> Tensor:
        horizontal_mask = self.horizontalbar.generate_mask(height, width, wrap)
        vertical_mask = self.verticalbar.generate_mask(height, width, wrap)
        return torch.logical_or(horizontal_mask, vertical_mask)