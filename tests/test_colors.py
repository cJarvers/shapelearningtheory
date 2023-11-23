import unittest
from shapelearningtheory.colors import *

class TestRedXORBlue(unittest.TestCase):
    def test_red_or_blue(self):
        colors = [RedXORBlue() for _ in range(10)]
        red_values = [color.color[:, :, 0] for color in colors]
        blue_values = [color.color[:, :, 2] for color in colors]
        red_or_blue = [r > 0.9 or b > 0.9 for (r, b) in zip(red_values, blue_values)]
        self.assertTrue(all(red_or_blue))

    def test_not_red_and_blue(self):
        colors = [RedXORBlue() for _ in range(10)]
        red_values = [color.color[:, :, 0] for color in colors]
        blue_values = [color.color[:, :, 2] for color in colors]
        not_red_and_blue = [
            r < 0.1 or b < 0.1 for (r, b) in zip(red_values, blue_values)
        ]
        self.assertTrue(all(not_red_and_blue))
