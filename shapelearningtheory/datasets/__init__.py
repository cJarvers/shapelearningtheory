from .linedataset import LineDataModule
from .LTPlusDataset import LTDataModule, LTPlusDataModule
from .rectangledataset import RectangleDataModule
from .squaredataset import SquaresDataModule

# Standard parameterizations of datasets used in experiments
from ..colors import RedXORBlue, NotRedXORBlue, RandomGrey, Grey
from ..textures import HorizontalGrating, VerticalGrating

def make_rectangles_color():
    """Standard parametrization for colored rectangles dataset."""
    dataset = RectangleDataModule(
        imgheight=15, imgwidth=15, 
        lengths=range(7, 13), widths=range(5, 9),
        pattern1=RedXORBlue, pattern2=NotRedXORBlue,
        background_pattern=RandomGrey,
        oversampling_factor=5, batch_size=128)
    return dataset
    
def make_rectangles_texture():
    """Standard parametrization for striped rectangles dataset."""
    dataset = RectangleDataModule(
        imgheight=15, imgwidth=15, 
        lengths=range(7, 13), widths=range(5, 9),
        pattern1=HorizontalGrating, pattern2=VerticalGrating,
        background_pattern=RandomGrey,
        oversampling_factor=5, batch_size=128)
    return dataset

def make_rectangles_shapeonly():
    """Standard parametrization for rectangle dataset without color/texture."""
    dataset = RectangleDataModule(
        imgheight=15, imgwidth=15, 
        lengths=range(7, 13), widths=range(5, 9),
        pattern1=Grey, pattern2=Grey,
        background_pattern=RandomGrey,
        oversampling_factor=5, batch_size=128)
    return dataset

def make_rectangles_coloronly():
    """Standard parametrization for colored squares (no shape feature)."""
    dataset = SquaresDataModule(
        height=15, width=15,
        lengths=range(5, 9),
        pattern1=RedXORBlue, pattern2=NotRedXORBlue,
        background_pattern=RandomGrey,
        oversampling_factor=5, batch_size=128)
    return dataset

def make_rectangles_textureonly():
    """Standard parametrization for striped squares (no shape feature)."""
    dataset = SquaresDataModule(
        height=15, width=15,
        lengths=range(5, 9),
        pattern1=HorizontalGrating, pattern2=VerticalGrating,
        background_pattern=RandomGrey,
        oversampling_factor=5, batch_size=128)
    return dataset

def make_rectangles_wrong_color():
    """Cue conflict version of colored rectangles."""
    dataset = RectangleDataModule(
        imgheight=15, imgwidth=15, 
        lengths=range(7, 13), widths=range(5, 9),
        pattern1=NotRedXORBlue, pattern2=RedXORBlue,
        background_pattern=RandomGrey,
        oversampling_factor=5, batch_size=128)
    return dataset

def make_rectangles_wrong_texture():
    """Cue conflict version of striped rectangles."""
    dataset = RectangleDataModule(
        imgheight=15, imgwidth=15, 
        lengths=range(7, 13), widths=range(5, 9),
        pattern1=VerticalGrating, pattern2=HorizontalGrating,
        background_pattern=RandomGrey,
        oversampling_factor=5, batch_size=128)
    return dataset


def make_LT_color():
    """Standard parametrization for LT dataset (small)."""
    dataset = LTDataModule(
        imgheight=18, imgwidth=18,
        heights=range(8, 12), widths=range(8, 12),
        strengths=range(1, 3),
        patternL=RedXORBlue, patternT=NotRedXORBlue,
        background_pattern=RandomGrey,
        batch_size=128)
    return dataset

def make_LT_shapeonly():
    """LT dataset (small) without color/texture feature."""
    dataset = LTDataModule(
        imgheight=18, imgwidth=18,
        heights=range(8, 12), widths=range(8, 12),
        strengths=range(1, 3),
        patternL=Grey, patternT=Grey,
        background_pattern=RandomGrey,
        batch_size=128)
    return dataset

def make_LT_coloronly():
    dataset = SquaresDataModule(
        height=18, width=18,
        lengths=range(8,12),
        pattern1=RedXORBlue, pattern2=NotRedXORBlue, # correct color, squares (no clear shape)
        background_pattern=RandomGrey,
        batch_size=128,
        oversampling_factor=4)
    return dataset

def make_LT_wrong_color():
    dataset = LTDataModule(
        imgheight=18, imgwidth=18,
        heights=range(8, 12), widths=range(8, 12),
        strengths=range(1, 3),
        patternL=NotRedXORBlue, patternT=RedXORBlue,
        background_pattern=RandomGrey,
        batch_size=128)
    return dataset