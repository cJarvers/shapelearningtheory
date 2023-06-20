import torch
from torch.utils.data import Dataset, random_split, DataLoader
from typing import List, Any, Type
from pytorch_lightning import LightningDataModule
# local imports:
from .shapecategories import Pixel, ColorSquare
from .colorcategories import Color, RandomRed, RandomBlue

class SquareDataset(Dataset):
    """Dataset for classifying squares according to color

    Args:
        - height: int - height of output images
        - width: int - width of output images
        - sidelengths: List[int] - side lengths of squares to generate
        - color1: Type[Color] - color type for class 1
        - verticalcolor: Type[Color] - color type for class 2
    """
    def __init__(self, height: int, width: int, sidelengths=List[int],
            color1: Type[Color] = RandomRed, color2: Type[Color] = RandomBlue) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.sidelengths = sidelengths
        self.color1 = color1
        self.color2 = color2
        self.squares1, self.squares2 = self.generate_all_squares()
        self.num_class1 = len(self.squares1)
        self.num_class2 = len(self.squares2)

    def generate_all_squares(self):
        squares1 = []
        # generate class 1
        for l in self.sidelengths:
            for x in range(self.height):
                for y in range(self.width):
                    squares1.append(ColorSquare(start=Pixel(x, y), sidelength=l, color=self.color1()))
        # generate class 2
        squares2 = []
        for l in self.sidelengths:
            for x in range(self.height):
                for y in range(self.width):
                    squares2.append(ColorSquare(start=Pixel(x, y), sidelength=l, color=self.color2()))
        return squares1, squares2

    def __getitem__(self, idx: int):
        if idx < self.num_class1:
            label = 0
            square = self.squares1[idx]
        else:
            label = 1
            square = self.squares2[idx - self.num_class1]
        image = torch.zeros(self.width, self.height, 3)
        square.draw_to_tensor(image)
        return image.permute(2,0,1), label

    def __len__(self):
        return self.num_class1 + self.num_class2
    

class SquaresDataModule(LightningDataModule):
    def __init__(self, height: int, width: int, lengths: List[int],
            batch_size: int = 32, num_workers: int = 4,
            color1: Type[Color] = RandomRed,
            color2: Type[Color] = RandomBlue):
        super().__init__()
        self.lengths = lengths
        self.color1 = color1
        self.color2 = color2
        self.save_hyperparameters(ignore=["lengths"])

    def prepare_data(self) -> None:
        self.dataset = SquareDataset(self.hparams.height, self.hparams.width,
            self.lengths, color1=self.color1,
            color2=self.color2)
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