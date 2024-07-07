import colorsys
from math import floor
import numpy as np

class ColorSet:
    def __init__(self, number_of_classes: int, hues_per_class: int = 2):
        self.number_of_classes = number_of_classes
        self.hues_per_class = hues_per_class
        self.number_of_hues = (number_of_classes * hues_per_class)
        hues = np.arange(0, 1, 1 / self.number_of_hues)
        np.random.shuffle(hues)
        self.base_hues = np.split(hues, number_of_classes)

    def sample(self, class_index):
        base_hues_for_class = self.base_hues[class_index]
        base_hue = np.random.choice(base_hues_for_class)
        hue = base_hue + (np.random.rand() - 0.5) / self.number_of_hues
        saturation = np.random.rand() / 2 + 0.5
        intensity = np.random.rand() / 2 + 0.5
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, intensity)
        return float_to_int_color(r), float_to_int_color(g), float_to_int_color(b)

def float_to_int_color(f: float):
    return floor(f * 255)


class RandomColorSet(ColorSet):
    """Behaves like a ColorSet, but returns a random color each time."""
    def __init__(self):
        pass

    def sample(self, class_index):
        hue = np.random.rand()
        saturation = np.random.rand() / 2 + 0.5
        intensity = np.random.rand() / 2 + 0.5
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, intensity)
        return float_to_int_color(r), float_to_int_color(g), float_to_int_color(b)
