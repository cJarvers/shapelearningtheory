import torch
from torch.utils.data import Dataset, random_split, DataLoader
from typing import List, Any, Type
from pytorch_lightning import LightningDataModule
# local imports:
from .shapecategories import Pixel, Orientation, ColorLine
from .colorcategories import Color, White, RandomRed, RandomBlue

class LineDataset(Dataset):
    """Dataset for classifying horizontal and vertical lines. Horizontal and
    vertical lines can additionally differ in color.

    Args:
        - height: int - height of output images
        - width: int - width of output images
        - lengths: List[int] - lengths of lines to generate
        - horizontalcolor: Type[Color] - class of colors for horizontal lines
        - verticalcolor: Type[Color] - class of colors for vertical lines
    """
    def __init__(self, height: int, width: int, lengths=List[int],
            horizontalcolor: Type[Color] = White, verticalcolor: Type[Color] = White) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.lengths = lengths
        self.horizontalcolor = horizontalcolor
        self.verticalcolor = verticalcolor
        self.lines = self.generate_all_lines()

    def generate_all_lines(self):
        lines = []
        # generate vertical lines
        for l in self.lengths:
            for x in range(self.height):
                for y in range(self.width):
                    lines.append(ColorLine(Pixel(x, y), l, Orientation.VERTICAL, self.verticalcolor()))
        # generate horizontal lines
        for l in self.lengths:
            for x in range(self.height):
                for y in range(self.width):
                    lines.append(ColorLine(Pixel(x, y), l, Orientation.HORIZONTAL, self.horizontalcolor()))
        return lines

    def __getitem__(self, idx: int):
        line = self.lines[idx]
        label = 1 if line.orientation == Orientation.HORIZONTAL else 0
        image = torch.zeros(self.width, self.height, 3)
        line.draw_to_tensor(image)
        return image, label

    def __len__(self):
        return len(self.lines)
    
class LineDataModule(LightningDataModule):
    def __init__(self, height: int, width: int, lengths: List[int],
            batch_size: int = 32, num_workers: int = 4,
            horizontalcolor: Type[Color] = RandomRed,
            verticalcolor: Type[Color] = RandomBlue):
        super().__init__()
        self.lengths = lengths
        self.horizontalcolor = horizontalcolor
        self.verticalcolor = verticalcolor
        self.save_hyperparameters(ignore=["lengths"])

    def prepare_data(self) -> None:
        self.dataset = LineDataset(self.hparams.height, self.hparams.width,
            self.lengths, horizontalcolor=self.horizontalcolor,
            verticalcolor=self.verticalcolor)
        self.train, self.val = random_split(self.dataset, [0.5, 0.5])

    def train_dataloader(self) -> Any:
        return DataLoader(self.train, self.hparams.batch_size, shuffle=True,
            num_workers=self.hparams.num_workers)
    
    def val_dataloader(self) -> Any:
        return DataLoader(self.val, self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers)
    
    def test_dataloader(self) -> Any:
        return DataLoader(self.dataset, self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers)