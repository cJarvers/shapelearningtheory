from .colorcategories import Color
from .shapecategories import Shape
from .textures import Texture

class Stimulus:
    """Combination of a shape and a color or texture.
    Used to generate the actual stimulus picture."""
    def __init__(self, shape: Shape, pattern: Color | Texture, wrap: bool=True):
        self.shape = shape
        self.pattern = pattern
        self.wrap = wrap

    def create_image(self, height, width):
        mask = self.shape.generate_mask(height, width, self.wrap)
        image = self.pattern.fill_tensor(height, width)
        return mask * image