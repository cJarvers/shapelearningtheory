from PIL import ImageDraw
from typing import Optional

from .shape import Shape, calculate_short_side

class Ellipse(Shape):
    def draw(self, draw: ImageDraw.Draw, color: tuple[int, int, int]):
        draw.ellipse((self.x, self.y, self.x + self.width, self.y + self.height), fill=color)


class HorizontalEllipse(Ellipse):
    def __init__(self, x: int, y: int, length: int, aspect: Optional[float] = None):
        super().__init__(x, y, length, calculate_short_side(length, aspect))


class VerticalEllipse(Ellipse):
    def __init__(self, x: int, y: int, length: int, aspect: Optional[float] = None):
        super().__init__(x, y, calculate_short_side(length, aspect), length)
