import torch
from torch.utils.data import Dataset
from typing import List
# local imports:
from .shapecategories import Pixel, Orientation, Line

class LineDataset(Dataset):
    """
    """
    def __init__(self, height: int, width: int, lengths=List[int]) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.lengths = lengths
        self.lines = self.generate_all_lines()

    def generate_all_lines(self):
        lines = []
        # generate horizontal lines
        for l in self.lengths:
            for x in range(self.height):
                for y in range(self.width - l + 1):
                    lines.append(Line(Pixel(x, y), l, Orientation.HORIZONTAL))
        # generate vertical lines
        for l in self.lengths:
            for x in range(self.height - l + 1):
                for y in range(self.width):
                    lines.append(Line(Pixel(x, y), l, Orientation.VERTICAL))
        return lines

    def __getitem__(self, idx: int):
        line = self.lines[idx]
        label = 1 if line.orientation == Orientation.HORIZONTAL else 0
        image = torch.zeros(self.width, self.height)
        line.draw_to_tensor(image)
        return image, label

    def __len__(self):
        return len(self.lines)