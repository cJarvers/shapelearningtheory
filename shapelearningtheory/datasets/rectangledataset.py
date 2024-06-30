import torch
from torch.utils.data import Dataset, random_split, DataLoader
from typing import Any, List, Type
from pytorch_lightning import LightningDataModule
# local imports
from ..colors import Color, RandomRed, RandomBlue, Grey
from ..old_shapes import Pixel, Orientation, Rectangle
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
        - stride: int - step size between rectangle positions
        - oversampling_factor: int = 1 - number of rectangles to sample at each location
    """
    def __init__(self, imgheight: int, imgwidth: int, lengths: List[int],
            widths: List[int], pattern1: Type[Color] | Type[Texture],
            pattern2: Type[Color] | Type[Texture],
            background_pattern: Type[Color] | Type[Texture] = Grey,
            stride: int = 1,
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
        self.stride = stride
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
                    for x in range(2, self.imgheight-l-1, self.stride):
                        for y in range(2, self.imgwidth-w-1, self.stride):
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
                    for x in range(2, self.imgheight-w-1, self.stride):
                        for y in range(2, self.imgwidth-l-1, self.stride):
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
    def __init__(self, imgheight: int, imgwidth: int, lengths: List[int],
            widths: List[int], batch_size: int = 32, num_workers: int = 4,
            pattern1: Type[Color] | Type[Texture] = RandomRed,
            pattern2: Type[Color] | Type[Texture] = RandomBlue,
            background_pattern: Type[Color] | Type[Texture] = Grey,
            stride: int = 1,
            oversampling_factor: int = 1):
        super().__init__()
        self.imgheight = imgheight
        self.imgwidth = imgwidth
        self.lengths = lengths
        self.widths = widths
        self.pattern1 = pattern1
        self.pattern2 = pattern2
        self.background_pattern = background_pattern
        self.stride = stride
        self.oversampling_factor = oversampling_factor
        self.save_hyperparameters(ignore=["lengths", "widths"])

    def prepare_data(self) -> None:
        self.train = RectangleDataset(
            self.imgheight, self.imgwidth, self.lengths,
            self.widths, self.pattern1, self.pattern2,
            background_pattern=self.background_pattern,
            stride = self.stride,
            oversampling_factor=self.oversampling_factor
        )
        self.val = RectangleDataset(
            self.imgheight, self.imgwidth, self.lengths,
            self.widths, self.pattern1, self.pattern2,
            background_pattern=self.background_pattern,
            stride = self.stride,
            oversampling_factor=self.oversampling_factor
        )

    def train_dataloader(self) -> Any:
        return DataLoader(self.train, self.hparams.batch_size, shuffle=True,
            num_workers=self.hparams.num_workers)
    
    def val_dataloader(self) -> Any:
        return DataLoader(self.val, self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers)
    
    def test_dataloader(self) -> Any:
        return DataLoader(self.val, self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers)