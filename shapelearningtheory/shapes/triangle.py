import random
from PIL import ImageDraw
from typing import Optional

from .shape import Shape, calculate_short_side

class Triangle(Shape):
    def __init__(self, x: int, y: int, width: int, height: int,
                 tip_offset: Optional[float] = None, flip: Optional[bool] = None):
        super().__init__(x, y, width, height)
        if tip_offset is None:
            self.tip_offset = random.random()
        else:
            self.tip_offset = tip_offset
        if flip is None:
            self.flip = random.random() > 0.5
        else:
            self.flip = flip

    def draw(self, draw: ImageDraw.Draw, color: tuple[int, int, int]):
        if self.height > self.width:
            first_x = self.x if not self.flip else self.x + self.width
            first_y = self.y
            second_x = self.x + self.width if not self.flip else self.x
            second_y = self.y + round(self.height * self.tip_offset)
            third_x = first_x
            third_y = first_y + self.height
        else:
            first_x = self.x
            first_y = self.y if not self.flip else self.y + self.height
            second_x = self.x + round(self.width * self.tip_offset)
            second_y = self.y + self.height if not self.flip else self.y
            third_x = first_x + self.width
            third_y = first_y
        draw.polygon(((first_x, first_y), (second_x, second_y), (third_x, third_y)), fill=color)


class HorizontalTriangle(Triangle):
    def __init__(self, x: int, y: int, length: int, aspect: Optional[float] = None,
                 tip_offset: Optional[float] = None, flip: Optional[bool] = None):
        super().__init__(x, y, length, calculate_short_side(length, aspect), tip_offset, flip)


class VerticalTriangle(Triangle):
    def __init__(self, x: int, y: int, length: int, aspect: Optional[float] = None,
                 tip_offset: Optional[float] = None, flip: Optional[bool] = None):
        super().__init__(x, y, calculate_short_side(length, aspect), length, tip_offset, flip)
