import torch
from torch.utils.data import Dataset, random_split, DataLoader
from typing import List, Any, Type
from pytorch_lightning import LightningDataModule
# local imports:
from ..old_shapes import Pixel, Square
from ..colors import Color, RandomRed, RandomBlue, Grey
from ..stimuli import Stimulus
from ..textures import Texture

class SquareDataset(Dataset):
    """Dataset for classifying squares according to color

    Args:
        - height: int - height of output images
        - width: int - width of output images
        - sidelengths: List[int] - side lengths of squares to generate
        - pattern1: Type[Color] | Type[Texture] - color or texture type for class 1
        - pattern2: Type[Color] | Type[Texture] - color or texture type for class 2
        - background_pattern: Type[Color] | Type[Texture] - color or texture to fill the background with
        - oversampling_factor: int = 1 - number of rectangles to sample at each location
    """
    def __init__(self, height: int, width: int, sidelengths: List[int],
            pattern1: Type[Color] | Type[Texture],
            pattern2: Type[Color] | Type[Texture],
            background_pattern: Type[Color] | Type[Texture] = Grey,
            stride: int = 1,
            oversampling_factor: int = 1) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.sidelengths = sidelengths
        self.pattern1 = pattern1
        self.pattern2 = pattern2
        self.background_pattern = background_pattern
        self.stride = stride
        self.oversampling_factor = oversampling_factor
        self.squares1, self.squares2 = self.generate_all_squares()
        self.num_class1 = len(self.squares1)
        self.num_class2 = len(self.squares2)

    def generate_all_squares(self):
        squares1 = []
        # generate class 1
        for l in self.sidelengths:
            for x in range(2, self.height-l-1, self.stride):
                for y in range(1, self.width-l, self.stride):
                    for _ in range(self.oversampling_factor):
                        squares1.append(
                            Stimulus(
                                shape=Square(start=Pixel(x, y), sidelength=l),
                                pattern=self.pattern1(),
                                background_pattern=self.background_pattern()
                            )
                        )
        # generate class 2
        squares2 = []
        for l in self.sidelengths:
            for x in range(2, self.height-l-1, self.stride):
                for y in range(2, self.width-l-1, self.stride):
                    for _ in range(self.oversampling_factor):
                        squares2.append(
                            Stimulus(
                                shape=Square(start=Pixel(x, y), sidelength=l),
                                pattern=self.pattern2(),
                                background_pattern=self.background_pattern()
                            )
                        )
        return squares1, squares2

    def __getitem__(self, idx: int):
        if idx < self.num_class1:
            label = 0
            square = self.squares1[idx]
        else:
            label = 1
            square = self.squares2[idx - self.num_class1]
        image = square.create_image(self.height, self.width)
        return image, label

    def __len__(self):
        return self.num_class1 + self.num_class2
    

class SquaresDataModule(LightningDataModule):
    def __init__(self, height: int, width: int, lengths: List[int],
            batch_size: int = 32, num_workers: int = 4,
            pattern1: Type[Color] | Type[Texture] = RandomRed,
            pattern2: Type[Color] | Type[Texture] = RandomBlue,
            background_pattern: Type[Color] | Type[Texture] = Grey,
            oversampling_factor: int = 1,
            stride: int = 1):
        super().__init__()
        self.lengths = lengths
        self.pattern1 = pattern1
        self.pattern2 = pattern2
        self.background_pattern = background_pattern
        self.stride = stride
        self.save_hyperparameters(ignore=["lengths"])

    def prepare_data(self) -> None:
        self.train = SquareDataset(self.hparams.height, self.hparams.width,
            self.lengths, pattern1=self.pattern1,
            pattern2=self.pattern2,
            background_pattern=self.background_pattern,
            stride=self.stride,
            oversampling_factor=self.hparams.oversampling_factor)
        self.val = SquareDataset(self.hparams.height, self.hparams.width,
            self.lengths, pattern1=self.pattern1,
            pattern2=self.pattern2,
            background_pattern=self.background_pattern,
            stride=self.stride,
            oversampling_factor=self.hparams.oversampling_factor)

    def train_dataloader(self) -> Any:
        return DataLoader(self.train, self.hparams.batch_size, shuffle=True,
            num_workers=self.hparams.num_workers)
    
    def val_dataloader(self) -> Any:
        return DataLoader(self.val, self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers)
    
    def test_dataloader(self) -> Any:
        return DataLoader(self.val, self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers)