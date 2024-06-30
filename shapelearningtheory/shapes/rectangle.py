from PIL import ImageDraw
from typing import Optional

from .shape import Shape, calculate_short_side

class Rectangle(Shape):
    def draw(self, draw: ImageDraw.Draw, color: tuple[int, int, int]):
        draw.rectangle((self.x, self.y, self.x + self.width, self.y + self.height), fill=color)


class HorizontalRectangle(Rectangle):
    def __init__(self, x: int, y: int, length: int, aspect: Optional[float] = None):
        super().__init__(x, y, length, calculate_short_side(length, aspect))


class VerticalRectangle(Rectangle):
    def __init__(self, x: int, y: int, length: int, aspect: Optional[float] = None):
        super().__init__(x, y, calculate_short_side(length, aspect), length)