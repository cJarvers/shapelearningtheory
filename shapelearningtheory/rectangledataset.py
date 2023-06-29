import torch
from torch.utils.data import Dataset, random_split, DataLoader
from typing import Any, List, Type
from pytorch_lightning import LightningDataModule
# local imports
from .colors import Color, RandomRed, RandomBlue
from .shapes import Pixel, Orientation, Rectangle
from .stimuli import Stimulus

class ColorRectangleDataset(Dataset):
    """
    Dataset for classifying recangles according to orientation (whether they are
    higher than wide or wider than high) and color.

    Args:
        - imgheight: int - height of output images
        - imgwidth: int - width of output images
        - lenghts: List[int] - lengths of longer sides for which to generate rectangles
        - widths: List[int] - lengths of shorter sides for which to generate rectangles
        - color1: Type[Color] - color type for class 1
        - color2: Type[Color] - color type for class 2
    """
    def __init__(self, imgheight: int, imgwidth: int, lengths: List[int],
            widths: List[int], color1: Type[Color], color2: Type[Color]):
        super().__init__()
        # store parameters
        self.imgheight = imgheight
        self.imgwidth = imgwidth
        self.lengths = lengths
        self.widths = widths
        self.color1 = color1
        self.color2 = color2
        # generate dataset
        self.rectangles = self.generate_all_rectangles()

    def generate_all_rectangles(self):
        horizontal = []
        vertical = []
        for l in self.lengths:
            for w in self.widths:
                if w < l:
                    for x in range(self.imgheight):
                        for y in range(self.imgwidth):
                            horizontal.append(
                                Stimulus(
                                    shape=Rectangle(
                                        start=Pixel(x, y),
                                        length=l,
                                        width=w,
                                        orientation=Orientation.HORIZONTAL
                                    ),
                                    pattern=self.color1()
                                )
                            )
                            vertical.append(
                                Stimulus(
                                    shape=Rectangle(
                                        start=Pixel(x, y),
                                        length=l,
                                        width=w,
                                        orientation=Orientation.VERTICAL
                                    ),
                                    pattern=self.color2()
                                )
                            )
        return horizontal + vertical
    
    def __getitem__(self, index: int) -> Any:
        rectangle = self.rectangles[index]
        if rectangle.shape.orientation == Orientation.HORIZONTAL:
            label = 0
        else:
            label = 1
        image = rectangle.create_image(self.imgheight, self.imgwidth)
        return image, label
    
    def __len__(self):
        return len(self.rectangles)


class ColorRectangleDataModule(LightningDataModule):
    def __init__(self, imgheight:int, imgwidth: int, lengths: List[int],
            widths: List[int], batch_size: int = 32, num_workers: int = 4,
            color1: Type[Color] = RandomRed,
            color2: Type[Color] = RandomBlue,
            validation_ratio: float = 0.0):
        super().__init__()
        self.lengths = lengths
        self.widths = widths
        self.color1 = color1
        self.color2 = color2
        self.save_hyperparameters(ignore=["lengths", "widths"])

    def prepare_data(self) -> None:
        self.dataset = ColorRectangleDataset(
            self.hparams.imgheight, self.hparams.imgwidth, self.lengths,
            self.widths, self.color1, self.color2
        )
        p_train = 1.0 - self.hparams.validation_ratio
        p_val = self.hparams.validation_ratio
        self.train, self.val = random_split(self.dataset, [p_train, p_val])

    def train_dataloader(self) -> Any:
        return DataLoader(self.train, self.hparams.batch_size, shuffle=True,
            num_workers=self.hparams.num_workers)
    
    def val_dataloader(self) -> Any:
        return DataLoader(self.val, self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers)
    
    def test_dataloader(self) -> Any:
        return DataLoader(self.dataset, self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers)