from PIL import ImageDraw
from typing import Optional

from .shape import Shape, calculate_short_side

class Cross(Shape):
    def draw(self, draw: ImageDraw.Draw, color: tuple[int, int, int]):
        x_vertical = self.x + self.width // 4
        y_vertical = self.y
        x_horizontal = self.x
        y_horizontal = self.y + self.height // 4
        draw.rectangle((x_vertical, y_vertical, x_vertical + self.width // 2, y_vertical + self.height), fill=color)
        draw.rectangle((x_horizontal, y_horizontal, x_horizontal + self.width, y_horizontal + self.height // 2), fill=color)


class HorizontalCross(Cross):
    def __init__(self, x: int, y: int, length: int, aspect: Optional[float] = None):
        super().__init__(x, y, length, calculate_short_side(length, aspect))


class VerticalCross(Cross):
    def __init__(self, x: int, y: int, length: int, aspect: Optional[float] = None):
        super().__init__(x, y, calculate_short_side(length, aspect), length)
    