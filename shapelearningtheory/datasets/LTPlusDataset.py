import torch
from torch.utils.data import Dataset, random_split, DataLoader
from pytorch_lightning import LightningDataModule
from typing import Any, List, Type
# local imports
from ..colors import Color, Grey
from ..textures import Texture
from ..stimuli import Stimulus
from ..shapes import LShape, TShape, PlusShape, Pixel

class LTPlusDataset(Dataset):
    """
    """
    def __init__(self, imgheight: int, imgwidth: int, heights: List[int],
            widths: List[int], strengths: List[int],
            patternL: Type[Color] | Type[Texture],
            patternT: Type[Color] | Type[Texture],
            patternPlus: Type[Color] | Type[Texture],
            background_pattern: Type[Color] | Type[Texture] = Grey):
        super().__init__()
        self.imgheight = imgheight
        self.imgwidth = imgwidth
        self.heights = heights
        self.widths = widths
        self.strengths = strengths
        self.patternL = patternL
        self.patternT = patternT
        self.patternPlus = patternPlus
        self.background_pattern = background_pattern
        self.ls, self.ts, self.plusses = self._generate_stimuli()

    def _generate_stimuli(self):
        ls = []
        ts = []
        plusses = []
        for h in self.heights:
            for w in self.widths:
                for s in self.strengths:
                    for x in range(self.imgwidth - w):
                        for y in range(self.imgheight - h):
                            for corner in ["topright", "topleft", "bottomright", "bottomleft"]:
                                ls.append(Stimulus(
                                    shape=LShape(
                                        start=Pixel(x, y),
                                        height=h, width=w, strength=s,
                                        corner=corner
                                    ),
                                    pattern=self.patternL,
                                    background_pattern=self.background_pattern
                                ))
                            for topside in ["right", "left", "top", "bottom"]:
                                ts.append(Stimulus(
                                    shape=TShape(
                                        start=Pixel(x, y),
                                        height=h, width=w, strength=s,
                                        topside=topside
                                    )
                                ))
                            for x_offset, y_offset in zip([-s, s], [-s, s]):
                                plusses.append(Stimulus(
                                    shape=PlusShape(
                                        start=Pixel(x, y),
                                        height=h, width=w, strength=s,
                                        x_offset=x_offset, y_offset=y_offset
                                    )
                                ))
        return ls, ts, plusses

    def __getitem__(self, index) -> Any:
        num_ls = len(self.ls)
        num_ts = len(self.ts)
        if index < num_ls:
            stimulus = self.ls[index]
            label = 0
        elif index < num_ls + num_ts:
            stimulus = self.ts[index - num_ls]
            label = 1
        else:
            stimulus = self.plusses[index - num_ls - num_ts]
            label = 2
        image = stimulus.create_image(self.imgheight, self.imgwidth)
        return image, label
    
    def __len__(self):
        return len(self.ls) + len(self.ts) + len(self.plusses)
        

class LTPlusDataModule(LightningDataModule):
    def __init__(self, imgheight: int, imgwidth: int, heights: List[int],
            widths: List[int], strengths: List[int],
            patternL: Type[Color] | Type[Texture],
            patternT: Type[Color] | Type[Texture],
            patternPlus: Type[Color] | Type[Texture],
            background_pattern: Type[Color] | Type[Texture],
            batch_size: int = 32, num_workers: int = 4,
            validation_ration: float = 0.0):
        super().__init__()
        self.imgheight = imgheight
        self.imgwidth = imgwidth
        self.heights = heights
        self.widths = widths
        self.strengths = strengths
        self.patternL = patternL
        self.patternT = patternT
        self.patternPlus = patternPlus
        self.background_pattern = background_pattern
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_ratio = validation_ration

    def prepare_data(self) -> None:
        self.dataset = LTPlusDataset(
            self.imgheight, self.imgwidth,
            self.heights, self.widths, self.strengths,
            self.patternL, self.patternT, self.patternPlus,
            self.background_pattern
        )
        p_train = 1.0 - self.validation_ratio
        p_val = self.validation_ratio
        self.train, self.val = random_split(self.data, [p_train, p_val])

    def train_dataloader(self) -> Any:
        return DataLoader(self.train, self.batch_size, shuffle=True,
            num_workers=self.num_workers)
    
    def val_dataloader(self) -> Any:
        return DataLoader(self.val, self.batch_size, shuffle=False,
            num_workers=self.num_workers)
    
    def test_dataloader(self) -> Any:
        return DataLoader(self.dataset, self.batch_size, shuffle=False,
            num_workers=self.num_workers)
