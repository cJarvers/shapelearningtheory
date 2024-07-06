from PIL import Image, ImageDraw
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule

from ..shapes import *
from ..color_set import ColorSet
from ..colors import Color, RandomGrey

default_shape_classes = [
    HorizontalRectangle,
    VerticalRectangle,
    HorizontalEllipse,
    VerticalEllipse,
    HorizontalCross,
    VerticalCross,
    HorizontalTriangle,
    VerticalTriangle,
    HorizontalParallelogram,
    VerticalParallelogram
]

default_color_set = ColorSet(
    number_of_classes=len(default_shape_classes),
    hues_per_class=2
)


class MultiShapeDataset(Dataset):
    """
    Dataset for classifying simple shapes according to shape or color.
    
    """
    def __init__(self,
                 shape_classes: list[Shape] = default_shape_classes,
                 color_set: ColorSet = default_color_set,
                 image_size: int = 112,
                 images_per_class: int = 1000,
                 background_color: type[Color] = RandomGrey):
        self.shape_classes = shape_classes
        self.color_set = color_set
        self.image_size = image_size
        self.images_per_class = images_per_class
        self.background_color = background_color
        self.images, self.labels = self.__generate_images()

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]
    
    def __len__(self):
        return len(self.images)

    def __generate_images(self):
        image_transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])
        labels = torch.zeros(self.images_per_class * len(self.shape_classes), dtype=torch.long)
        images = []
        for i in range(self.images_per_class):
            x, y, length, aspect = self.__generate_random_shape_parameters()
            for shape_idx, shape in enumerate(self.shape_classes):
                index = i * len(self.shape_classes) + shape_idx
                labels[index] = shape_idx
                image = Image.new("RGB", (self.image_size, self.image_size), self.background_color().rgb_tuple())
                canvas = ImageDraw.Draw(image)
                color = self.color_set.sample(shape_idx)
                shape(x, y, length, aspect).draw(canvas, color)
                images.append(image_transform(image))
        return images, labels

    def __generate_random_shape_parameters(self):
        length = random.randint(self.image_size // 10, self.image_size // 2)
        aspect = random.random() * 0.65 + 0.3
        x = random.randint(1, self.image_size - length - 1)
        y = random.randint(1, self.image_size - length - 1)
        return x, y, length, aspect
    

class MultiShapeDataModule(LightningDataModule):
    def __init__(self, image_size: int, images_per_class: int,
                 batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.image_size = image_size
        self.images_per_class = images_per_class
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        self.train = MultiShapeDataset(
            image_size=self.image_size,
            images_per_class=self.images_per_class)
        self.val = MultiShapeDataset(
            image_size=self.image_size,
            images_per_class=self.images_per_class // 10
        )
        self.test = MultiShapeDataset(
            image_size=self.image_size,
            images_per_class=self.images_per_class
        )

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, shuffle=True,
            num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size, shuffle=False,
            num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size, shuffle=False,
            num_workers=self.num_workers)