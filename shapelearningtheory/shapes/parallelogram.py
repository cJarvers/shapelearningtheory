import random
from PIL import ImageDraw
from typing import Optional

from .shape import Shape, calculate_short_side

class Parallelogram(Shape):
    def __init__(self, x: int, y: int, width: int, height: int,
                 offset: Optional[float] = None, flip: Optional[bool] = None):
        super().__init__(x, y, width, height)
        if offset is None:
            self.offset = random.random() * 0.2 + 0.1
        else:
            if offset < 0.1 or offset > 0.3:
                raise ValueError("offset for a parallelogram should in interval [0.1, 0.3]")
            self.offset = offset
        if flip is None:
            self.flip = random.random() > 0.5
        else:
            self.flip = flip

    def draw(self, draw: ImageDraw.Draw, color: tuple[int, int, int]):
        if self.height > self.width:
            shift = round(self.height * self.offset)
            top_left_x = self.x
            top_left_y = self.y + shift if self.flip else self.y
            top_right_x = self.x + self.width
            top_right_y = self.y if self.flip else self.y + shift
            bottom_left_x = self.x
            bottom_left_y = self.y + self.height if self.flip else self.y + self.height - shift
            bottom_right_x = self.x + self.width
            bottom_right_y = self.y + self.height - shift if self.flip else self.y + self.height
        else:
            shift = round(self.width * self.offset)
            top_left_x = self.x + shift if self.flip else self.x
            top_left_y = self.y
            top_right_x = self.x + self.width if self.flip else self.x + self.width - shift
            top_right_y = self.y
            bottom_left_x = self.x if self.flip else self.x + shift
            bottom_left_y = self.y + self.height
            bottom_right_x = self.x + self.width - shift if self.flip else self.x + self.width
            bottom_right_y = self.y + self.height
        draw.polygon((
            (top_right_x, top_right_y),
            (top_left_x, top_left_y),
            (bottom_left_x, bottom_left_y),
            (bottom_right_x, bottom_right_y)
        ), fill=color)


class HorizontalParallelogram(Parallelogram):
    def __init__(self, x: int, y: int, length: int, aspect: Optional[float] = None,
                 offset: Optional[float] = None, flip: Optional[bool] = None):
        super().__init__(x, y, length, calculate_short_side(length, aspect), offset, flip)


class VerticalParallelogram(Parallelogram):
    def __init__(self, x: int, y: int, length: int, aspect: Optional[float] = None,
                 offset: Optional[float] = None, flip: Optional[bool] = None):
        super().__init__(x, y, calculate_short_side(length, aspect), length, offset, flip)