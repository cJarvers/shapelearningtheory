import random

from .shape import Shape

class RandomShapeSelector:
    def __init__(self, shapes: list[Shape]):
        self.__shapes = shapes

    def __len__(self):
        return len(self.__shapes)
    
    def __getitem__(self, idx):
        if idx >= len(self.__shapes):
            raise StopIteration
        idx = random.randint(0, len(self.__shapes)-1)
        return self.__shapes[idx]