from .linedataset import LineDataModule
from .LTPlusDataset import LTDataModule, LTPlusDataModule
from .rectangledataset import RectangleDataModule
from .squaredataset import SquaresDataModule

# Standard parameterizations of datasets used in experiments
from ..colors import RedXORBlue, NotRedXORBlue, RandomGrey, Grey, RandomChoiceColor
from ..textures import HorizontalGrating, VerticalGrating, RandomGrating

def make_rectangles_color(batchsize: int = 128):
    """Standard parametrization for colored rectangles dataset."""
    dataset = RectangleDataModule(
        imgheight=18, imgwidth=18, 
        lengths=range(7, 13), widths=range(5, 10),
        pattern1=RedXORBlue, pattern2=NotRedXORBlue,
        background_pattern=RandomGrey,
        oversampling_factor=5,
        stride=1,
        batch_size=batchsize)
    return dataset

def make_rectangles_color_large(batchsize: int = 128):
    """Standard parametrization for colored rectangles dataset (large)."""
    dataset = RectangleDataModule(
        imgheight=224, imgwidth=224, 
        lengths=[50, 99, 100, 101, 150], widths=[49, 98, 99, 199, 149],
        pattern1=RedXORBlue, pattern2=NotRedXORBlue,
        background_pattern=RandomGrey,
        oversampling_factor=2,
        stride=20,
        batch_size=batchsize)
    return dataset

def make_rectangles_texture(batchsize: int = 128):
    """Standard parametrization for striped rectangles dataset."""
    dataset = RectangleDataModule(
        imgheight=18, imgwidth=18, 
        lengths=range(7, 13), widths=range(5, 10),
        pattern1=HorizontalGrating, pattern2=VerticalGrating,
        background_pattern=RandomGrey,
        oversampling_factor=5,
        stride=1,
        batch_size=batchsize)
    return dataset

def make_rectangles_texture_large(batchsize: int = 128):
    """Standard parametrization for striped rectangles dataset."""
    dataset = RectangleDataModule(
        imgheight=224, imgwidth=224, 
        lengths=[50, 99, 100, 101, 150], widths=[49, 98, 99, 199, 149],
        pattern1=HorizontalGrating, pattern2=VerticalGrating,
        background_pattern=RandomGrey,
        oversampling_factor=2,
        stride=20,
        batch_size=batchsize)
    return dataset

def make_rectangles_shapeonly(batchsize: int = 128):
    """Standard parametrization for rectangle dataset without color/texture."""
    dataset = RectangleDataModule(
        imgheight=18, imgwidth=18, 
        lengths=range(7, 13), widths=range(5, 10),
        pattern1=Grey, pattern2=Grey,
        background_pattern=RandomGrey,
        oversampling_factor=5,
        stride=1,
        batch_size=batchsize)
    return dataset

def make_rectangles_shapeonly_large(batchsize: int = 128):
    """Standard parametrization for rectangle dataset without color/texture."""
    dataset = RectangleDataModule(
        imgheight=224, imgwidth=224, 
        lengths=[50, 99, 100, 101, 150], widths=[49, 98, 99, 199, 149],
        pattern1=Grey, pattern2=Grey,
        background_pattern=RandomGrey,
        oversampling_factor=2,
        stride=20,
        batch_size=batchsize)
    return dataset

def make_rectangles_coloronly(batchsize: int = 128):
    """Standard parametrization for colored squares (no shape feature)."""
    dataset = SquaresDataModule(
        height=18, width=18,
        lengths=range(5, 13),
        pattern1=RedXORBlue, pattern2=NotRedXORBlue,
        background_pattern=RandomGrey,
        oversampling_factor=5,
        stride=1,
        batch_size=batchsize)
    return dataset

def make_rectangles_coloronly_large(batchsize: int = 128):
    """Standard parametrization for colored squares (no shape feature)."""
    dataset = SquaresDataModule(
        height=224, width=224, 
        lengths=[50, 99, 100, 101, 150],
        pattern1=RedXORBlue, pattern2=NotRedXORBlue,
        background_pattern=RandomGrey,
        oversampling_factor=2,
        stride=20,
        batch_size=batchsize)
    return dataset

def make_rectangles_textureonly(batchsize: int = 128):
    """Standard parametrization for striped squares (no shape feature)."""
    dataset = SquaresDataModule(
        height=18, width=18,
        lengths=range(5, 13),
        pattern1=HorizontalGrating, pattern2=VerticalGrating,
        background_pattern=RandomGrey,
        oversampling_factor=5,
        stride=1,
        batch_size=batchsize)
    return dataset

def make_rectangles_textureonly_large(batchsize: int = 128):
    """Standard parametrization for striped squares (no shape feature)."""
    dataset = SquaresDataModule(
        height=224, width=224, 
        lengths=[50, 99, 100, 101, 150],
        pattern1=HorizontalGrating, pattern2=VerticalGrating,
        background_pattern=RandomGrey,
        oversampling_factor=2,
        stride=20,
        batch_size=batchsize)
    return dataset

def make_rectangles_wrong_color(batchsize: int = 128):
    """Cue conflict version of colored rectangles."""
    dataset = RectangleDataModule(
        imgheight=18, imgwidth=18, 
        lengths=range(7, 13), widths=range(5, 10),
        pattern1=NotRedXORBlue, pattern2=RedXORBlue,
        background_pattern=RandomGrey,
        oversampling_factor=5,
        stride=1,
        batch_size=batchsize)
    return dataset

def make_rectangles_wrong_color_large(batchsize: int = 128):
    """Cue conflict version of colored rectangles."""
    dataset = RectangleDataModule(
        imgheight=224, imgwidth=224, 
        lengths=[50, 99, 100, 101, 150], widths=[49, 98, 99, 199, 149],
        pattern1=NotRedXORBlue, pattern2=RedXORBlue,
        background_pattern=RandomGrey,
        oversampling_factor=2,
        stride=20,
        batch_size=batchsize)
    return dataset

def make_rectangles_wrong_texture(batchsize: int = 128):
    """Cue conflict version of striped rectangles."""
    dataset = RectangleDataModule(
        imgheight=18, imgwidth=18, 
        lengths=range(7, 13), widths=range(5, 10),
        pattern1=VerticalGrating, pattern2=HorizontalGrating,
        background_pattern=RandomGrey,
        oversampling_factor=5,
        stride=1,
        batch_size=batchsize)
    return dataset

def make_rectangles_wrong_texture_large(batchsize: int = 128):
    """Cue conflict version of striped rectangles."""
    dataset = RectangleDataModule(
        imgheight=224, imgwidth=224, 
        lengths=[50, 99, 100, 101, 150], widths=[49, 98, 99, 199, 149],
        pattern1=VerticalGrating, pattern2=HorizontalGrating,
        background_pattern=RandomGrey,
        oversampling_factor=2,
        stride=20,
        batch_size=batchsize)
    return dataset

def make_rectangles_random_color(batchsize: int = 128):
    """Similar to color rectangle dataset, but color assignment does not match shape class"""
    dataset = RectangleDataModule(
        imgheight=18, imgwidth=18, 
        lengths=range(7, 13), widths=range(5, 10),
        pattern1=RandomChoiceColor, pattern2=RandomChoiceColor,
        background_pattern=RandomGrey,
        oversampling_factor=5,
        stride=1,
        batch_size=batchsize)
    return dataset

def make_rectangles_random_texture(batchsize: int = 128):
    """Similar to texture rectangle dataset, but texture orientation is random, independent of class."""
    dataset = RectangleDataModule(
        imgheight=18, imgwidth=18, 
        lengths=range(7, 13), widths=range(5, 10),
        pattern1=RandomGrating, pattern2=RandomGrating,
        background_pattern=RandomGrey,
        oversampling_factor=5,
        stride=1,
        batch_size=batchsize)
    return dataset

def make_LT_color(batchsize: int = 128):
    """Standard parametrization for LT dataset (small, color)."""
    dataset = LTDataModule(
        imgheight=18, imgwidth=18,
        heights=range(7, 13), widths=range(7, 13),
        strengths=range(1, 3),
        patternL=RedXORBlue, patternT=NotRedXORBlue,
        background_pattern=RandomGrey,
        batch_size=batchsize)
    return dataset

def make_LT_color_large(batchsize: int = 128):
    """Standard parametrization for LT dataset (large, color)."""
    dataset = LTDataModule(
        imgheight=224, imgwidth=224,
        heights=[50, 99, 100, 101, 150], widths=[50, 99, 100, 101, 150],
        strengths=[5, 7, 10, 12, 15],
        patternL=RedXORBlue, patternT=NotRedXORBlue,
        background_pattern=RandomGrey,
        batch_size=batchsize)
    return dataset

def make_LT_texture(batchsize: int = 128):
    """Standard parametrization for LT dataset (small, texture)."""
    dataset = LTDataModule(
        imgheight=18, imgwidth=18,
        heights=range(7, 13), widths=range(7, 13),
        strengths=range(1, 3),
        patternL=HorizontalGrating, patternT=VerticalGrating,
        background_pattern=RandomGrey,
        batch_size=batchsize)
    return dataset

def make_LT_texture_large(batchsize: int = 128):
    """Standard parametrization for LT dataset (large, texture)."""
    dataset = LTDataModule(
        imgheight=224, imgwidth=224,
        heights=[50, 99, 100, 101, 150], widths=[50, 99, 100, 101, 150],
        strengths=[5, 7, 10, 12, 15],
        patternL=HorizontalGrating, patternT=VerticalGrating,
        background_pattern=RandomGrey,
        batch_size=batchsize)
    return dataset

def make_LT_shapeonly(batchsize: int = 128):
    """LT dataset (small) without color/texture feature."""
    dataset = LTDataModule(
        imgheight=18, imgwidth=18,
        heights=range(7, 13), widths=range(7, 13),
        strengths=range(1, 3),
        patternL=Grey, patternT=Grey,
        background_pattern=RandomGrey,
        batch_size=batchsize)
    return dataset

def make_LT_shapeonly_large(batchsize: int = 128):
    """Standard parametrization for LT dataset (large, color)."""
    dataset = LTDataModule(
        imgheight=224, imgwidth=224,
        heights=[50, 99, 100, 101, 150], widths=[50, 99, 100, 101, 150],
        strengths=[5, 7, 10, 12, 15],
        patternL=Grey, patternT=Grey,
        background_pattern=Grey,
        batch_size=batchsize)
    return dataset

def make_LT_coloronly(batchsize: int = 128):
    dataset = SquaresDataModule(
        height=18, width=18,
        lengths=range(7, 13),
        pattern1=RedXORBlue, pattern2=NotRedXORBlue, # correct color, squares (no clear shape)
        background_pattern=RandomGrey,
        batch_size=batchsize,
        oversampling_factor=4)
    return dataset

def make_LT_textureonly(batchsize: int = 128):
    dataset = SquaresDataModule(
        height=18, width=18,
        lengths=range(7, 13),
        pattern1=HorizontalGrating, pattern2=VerticalGrating, # correct texture, squares (no clear shape)
        background_pattern=RandomGrey,
        batch_size=batchsize,
        oversampling_factor=4)
    return dataset

def make_LT_wrong_color(batchsize: int = 128):
    dataset = LTDataModule(
        imgheight=18, imgwidth=18,
        heights=range(7, 13), widths=range(7, 13),
        strengths=range(1, 3),
        patternL=NotRedXORBlue, patternT=RedXORBlue,
        background_pattern=RandomGrey,
        batch_size=batchsize)
    return dataset

def make_LT_wrong_texture(batchsize: int = 128):
    dataset = LTDataModule(
        imgheight=18, imgwidth=18,
        heights=range(7, 13), widths=range(7, 13),
        strengths=range(1, 3),
        patternL=VerticalGrating, patternT=HorizontalGrating,
        background_pattern=RandomGrey,
        batch_size=batchsize)
    return dataset