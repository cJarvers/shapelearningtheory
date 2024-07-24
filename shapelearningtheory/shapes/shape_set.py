import random

from .shape import Shape

class ShapeSet:
    def __init__(self, shape_groups: list[list[Shape]]):
        self.__shape_groups = shape_groups

    def __len__(self):
        return len(self.__shape_groups)
    
    def __getitem__(self, idx):
        if idx >= len(self.__shape_groups):
            raise StopIteration
        shape_in_group = random.randint(0, len(self.__shape_groups[idx])-1)
        return self.__shape_groups[idx][shape_in_group]
    
