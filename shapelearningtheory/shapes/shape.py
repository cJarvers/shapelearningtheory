from math import floor
from PIL import ImageDraw
import random
from typing import Optional

class Shape:
    """Base class for shapes"""
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def draw(self, draw: ImageDraw.Draw, color: tuple[int, int, int]):
        raise NotImplementedError("Base shape cannot be drawn. Use subclasses.")
    

def calculate_short_side(long_side: int, aspect: Optional[float] = None):
    """Utility method for shapes with fixed orientation (HorizontalSomething and VerticalSomething).
    Calculates how long the shorter side should be given the long side and an optional aspect ratio.
    If no aspect ratio is specificed, it is sampled from [0.3, 0.95]."""
    if aspect is None:
        aspect = random.random() * 0.65 + 0.3
    if aspect < 0.3 or aspect > 0.95:
        raise ValueError("Aspect ratio should be between 0.3 and 0.95.")
    short_side = floor(long_side * aspect)
    if short_side < 1:
        short_side = 1
    if short_side >= long_side and long_side > 1:
        short_side = long_side - 1
    return short_side
