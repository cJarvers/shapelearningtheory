import torch
from torch.utils.data import Dataset, random_split, DataLoader
from typing import Any, List, Type
from pytorch_lightning import LightningDataModule
# local imports
from ..colors import Color, RandomRed, RandomBlue, Grey
from ..shapes import Pixel, Orientation, Rectangle
from ..textures import Texture
from ..stimuli import Stimulus

class RectangleDataset(Dataset):
    """
    Dataset for classifying recangles according to orientation (whether they are
    higher than wide or wider than high) and color or texture.

    Args:
        - imgheight: int - height of output images
        - imgwidth: int - width of output images
        - lenghts: List[int] - lengths of longer sides for which to generate rectangles
        - widths: List[int] - lengths of shorter sides for which to generate rectangles
        - pattern1: Type[Color] | Type[Texture] - color or texture type for class 1
        - pattern2: Type[Color] | Type[Texture] - color or texture type for class 2
        - background_pattern: Type[Color] | Type[Texture] - color or texture to fill the background with
        - oversampling_factor: int = 1 - number of rectangles to sample at each location
    """
    def __init__(self, imgheight: int, imgwidth: int, lengths: List[int],
            widths: List[int], pattern1: Type[Color] | Type[Texture],
            pattern2: Type[Color] | Type[Texture],
            background_pattern: Type[Color] | Type[Texture] = Grey,
            oversampling_factor: int = 1):
        super().__init__()
        # store parameters
        self.imgheight = imgheight
        self.imgwidth = imgwidth
        self.lengths = lengths
        self.widths = widths
        self.pattern1 = pattern1
        self.pattern2 = pattern2
        self.background_pattern = background_pattern
        self.oversampling_factor = oversampling_factor
        # generate dataset
        self.rectangles = self.generate_all_rectangles()
        self.mean, self.std = self._compute_mean_and_std()

    def _compute_mean_and_std(self):
        all_imgs = torch.cat([r.create_image(self.imgheight, self.imgwidth).unsqueeze(0) for r in self.rectangles], dim=0)
        means = torch.mean(all_imgs, dim=(0,2,3)).unsqueeze(1).unsqueeze(2)
        stds = torch.std(all_imgs, dim=(0,2,3)).unsqueeze(1).unsqueeze(2)
        return means, stds

    def generate_all_rectangles(self):
        horizontal = []
        vertical = []
        for l in self.lengths:
            for w in self.widths:
                if w < l:
                    for x in range(0, self.imgheight+1-l, l):
                        for y in range(0, self.imgwidth+1-w, w):
                            for _ in range(self.oversampling_factor):
                                horizontal.append(
                                    Stimulus(
                                        shape=Rectangle(
                                            start=Pixel(x, y),
                                            length=l,
                                            width=w,
                                            orientation=Orientation.HORIZONTAL
                                        ),
                                        pattern=self.pattern1(),
                                        background_pattern=self.background_pattern()
                                    )
                                )
                    for x in range(0, self.imgheight+1-w, w):
                        for y in range(0, self.imgwidth+1-l, l):
                            for _ in range(self.oversampling_factor):
                                vertical.append(
                                    Stimulus(
                                        shape=Rectangle(
                                            start=Pixel(x, y),
                                            length=l,
                                            width=w,
                                            orientation=Orientation.VERTICAL
                                        ),
                                        pattern=self.pattern2(),
                                        background_pattern=self.background_pattern()
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
        # image = (image - self.mean) / self.std
        return image, label
    
    def __len__(self):
        return len(self.rectangles)


class RectangleDataModule(LightningDataModule):
    def __init__(self, imgheight:int, imgwidth: int, lengths: List[int],
            widths: List[int], batch_size: int = 32, num_workers: int = 4,
            pattern1: Type[Color] | Type[Texture] = RandomRed,
            pattern2: Type[Color] | Type[Texture] = RandomBlue,
            background_pattern: Type[Color] | Type[Texture] = Grey,
            validation_ratio: float = 0.0,
            oversampling_factor: int = 1):
        super().__init__()
        self.lengths = lengths
        self.widths = widths
        self.pattern1 = pattern1
        self.pattern2 = pattern2
        self.background_pattern = background_pattern
        self.save_hyperparameters(ignore=["lengths", "widths"])

    def prepare_data(self) -> None:
        self.dataset = RectangleDataset(
            self.hparams.imgheight, self.hparams.imgwidth, self.lengths,
            self.widths, self.pattern1, self.pattern2,
            background_pattern=self.background_pattern,
            oversampling_factor=self.hparams.oversampling_factor
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