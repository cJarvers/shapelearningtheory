from .colors import Color
from .shapes import Shape
from .textures import Texture
from typing import Optional

class Stimulus:
    """Combination of a shape and a color or texture.
    Used to generate the actual stimulus picture."""
    def __init__(self, shape: Shape, pattern: Color | Texture,
            background_pattern: Optional[Color | Texture] = None, wrap: bool=True):
        self.shape = shape
        self.pattern = pattern
        self.background_pattern = background_pattern
        self.wrap = wrap

    def create_image(self, height, width):
        mask = self.shape.generate_mask(height, width, self.wrap)
        image = self.pattern.fill_tensor(height, width)
        if self.background_pattern:
            background = self.background_pattern.fill_tensor(height, width)
            stimulus = mask * image + mask.logical_not() * background
        else:
            stimulus = mask * image
        return stimulus