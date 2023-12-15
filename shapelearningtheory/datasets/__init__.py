from typing import Literal

from .LTPlusDataset import LTDataModule
from .rectangledataset import RectangleDataModule
from .squaredataset import SquaresDataModule

# Standard parameterizations of datasets used in experiments
from ..colors import RedXORBlue, NotRedXORBlue, RandomGrey, Grey, RandomChoiceColor
from ..textures import HorizontalGrating, VerticalGrating, RandomGrating

def make_dataset(shape: Literal["rectangles", "LvT"],
                 pattern: Literal["color", "stripes"],
                 size: Literal["small", "large"],
                 variant: Literal["standard", "shapeonly", "patternonly", "conflict", "random"],
                 batchsize=128, num_workers=4):
    if pattern == "color":
        if variant == "shapeonly":
            pattern1 = Grey
            pattern2 = Grey
        elif variant == "random":
            pattern1 = RandomChoiceColor
            pattern2 = RandomChoiceColor
        else:
            pattern1 = RedXORBlue
            pattern2 = NotRedXORBlue
    else:
        if variant == "shapeonly":
            pattern1 = Grey
            pattern2 = Grey
        elif variant == "random":
            pattern1 = RandomGrating
            pattern2 = RandomGrating
        else:
            pattern1 = HorizontalGrating
            pattern2 = VerticalGrating
    background_pattern = RandomGrey
    if size == "small":
        height = 18
        width = 18
        stride = 1
        oversampling_factor = 5
        if shape == "rectangles":
            lengths = range(7, 13)
            widths = range(5, 10)
        else:
            lengths = range(7, 13)
            widths = range(7, 13)
            strengths = range(1, 3)
    else:
        height = 224
        width = 224
        oversampling_factor = 2
        stride = 20
        if shape == "rectangles":
            lengths=[50, 90, 100, 110, 150]
            widths=[40, 80, 90, 100, 140]
        else:
            lengths = [50, 100, 150]
            widths = [50, 100, 150]
            strengths = [5, 10, 15]
    if variant == "patternonly":
        dataset = SquaresDataModule(
            height=height, width=width,
            lengths=lengths,
            pattern1=pattern1, pattern2=pattern2,
            background_pattern=background_pattern,
            oversampling_factor=oversampling_factor,
            stride=stride,
            batch_size=batchsize,
            num_workers=num_workers
        )
    else:
        if shape == "rectangles":
            dataset = RectangleDataModule(
                imgheight=height, imgwidth=width,
                lengths=lengths,
                widths=widths,
                pattern1=pattern1,
                pattern2=pattern2,
                background_pattern=background_pattern,
                oversampling_factor=oversampling_factor,
                stride=stride,
                batch_size=batchsize,
                num_workers=num_workers
            )
        elif shape == "LvT":
            dataset = LTDataModule(
                imgheight=height, imgwidth=width,
                heights=lengths,
                widths=widths,
                strengths=strengths,
                patternL=pattern1,
                patternT=pattern2,
                background_pattern=background_pattern,
                stride=stride,
                batch_size=batchsize,
                num_workers=num_workers
            )
        else:
            raise ValueError("Unkown shape: " + shape)
    return dataset
    
