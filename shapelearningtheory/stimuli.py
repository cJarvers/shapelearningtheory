from .colors import Color
from .old_shapes import Shape
from .textures import Texture
from typing import Optional

class Stimulus:
    """Combination of a shape and a color or texture.
    Used to generate the actual stimulus picture."""
    def __init__(self, shape: Shape, pattern: Color | Texture,
            background_pattern: Optional[Color | Texture] = None,
            wrap: bool=True, correct_range: bool=False):
        self.shape = shape
        self.pattern = pattern
        self.background_pattern = background_pattern
        self.wrap = wrap
        self.correct_range = correct_range

    def create_image(self, height, width):
        mask = self.shape.generate_mask(height, width, self.wrap)
        image = self.pattern.fill_tensor(height, width)
        if self.background_pattern:
            background = self.background_pattern.fill_tensor(height, width)
            stimulus = mask * image + mask.logical_not() * background
        else:
            stimulus = mask * image
        if self.correct_range:
            # images are created in range [0, 1]; change to range [-1, 1] instead (better for training)
            stimulus = 2 * stimulus - 1
        return stimulus