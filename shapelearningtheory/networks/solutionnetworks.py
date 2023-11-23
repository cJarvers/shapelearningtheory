# Small networks with pre-specified weights that "solve" the datasets with pre-specified strategies.
import torch

class ColorConvNet(torch.nn.Sequential):
    """Convolutional network that classifies RedXORBlue vs. NotRedXORBlue by construction."""
    def __init__(self, imageheight: int, imagewidth: int,
                 min_pixels: int = 35, max_pixels: int = 117):
        super().__init__()
        self.imageheight = imageheight
        self.imagewidth = imagewidth
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.append(self.make_rbdiff_layer())
        self.append(torch.nn.ReLU())
        self.append(torch.nn.AdaptiveAvgPool2d(output_size=1))
        self.append(torch.nn.Flatten())
        self.append(self.make_colorclass_layer())

    def make_rbdiff_layer(self) -> torch.nn.Module:
        "Create layer that computes difference between red and blue channel."
        layer = torch.nn.Conv2d(3, 2, 1)
        layer.weight.data[0] = torch.tensor([1.0, 0.0, -1.0]).reshape(3,1,1) # First channel detects if red is much larger than blue
        layer.weight.data[1] = torch.tensor([-1.0, 0.0, 1.0]).reshape(3,1,1) # Second channel detects if blue is much larger than red
        layer.bias.data = torch.zeros(2)
        return layer

    def make_colorclass_layer(self) -> torch.nn.Module:
        """Create layer that classifies as RedXORBlue or NotRedXORBlue,
        depending on difference between red and blue channel (from previous layer)."""
        layer = torch.nn.Linear(2, 2)
        max_for_NotRedXORBlue = self.max_pixels * 0.1 / (self.imageheight * self.imagewidth)
        min_for_RedXORBlue = self.min_pixels * 0.8 / (self.imageheight * self.imagewidth)
        if max_for_NotRedXORBlue > min_for_RedXORBlue:
            raise ValueError("Cannot correctly set threshold.")
        threshold = (min_for_RedXORBlue + max_for_NotRedXORBlue) / 2
        layer.weight.data = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
        layer.bias.data = torch.tensor([-threshold, threshold])
        return layer


class CRectangleConvNet(torch.nn.Sequential):
    """ConvNet that classifies color rectangles by height / width."""
    def __init__(self):
        super().__init__()
        self.append(self.sobel_layer())
        self.append(torch.nn.ReLU())
        self.append(self.borders_layer())
        self.append(self.distance_layer())
        self.append(torch.nn.AdaptiveMaxPool2d(output_size=1))
        self.append(torch.nn.Flatten())
        self.append(self.decision_layer())

    def sobel_layer(self) -> torch.nn.Conv2d:
        """Create layer that performs Sobel filtering to detect horzontal
        and vertical contrasts in each color channel."""
        # Detect horizontal and vertical contrast with sobel filters
        sobel_x = torch.tensor([[-0.25, 0., 0.25], [-0.5, 0., 0.5], [-0.25, 0., 0.25]])
        sobel_y = sobel_x.T
        # Detect borders in each color channel for both possible polarities and both orientations: 12 channels
        conv_borders = torch.nn.Conv2d(3, 12, 3)
        conv_borders.weight.data.zero_()
        conv_borders.bias.data.zero_()
        # Set sobel filters:
        conv_borders.weight.data[0, 0, :, :] = sobel_x
        conv_borders.weight.data[1, 0, :, :] = -sobel_x
        conv_borders.weight.data[2, 1, :, :] = sobel_x
        conv_borders.weight.data[3, 1, :, :] = -sobel_x
        conv_borders.weight.data[4, 2, :, :] = sobel_x
        conv_borders.weight.data[5, 2, :, :] = -sobel_x
        conv_borders.weight.data[6, 0, :, :] = sobel_y
        conv_borders.weight.data[7, 0, :, :] = -sobel_y
        conv_borders.weight.data[8, 1, :, :] = sobel_y
        conv_borders.weight.data[9, 1, :, :] = -sobel_y
        conv_borders.weight.data[10, 2, :, :] = sobel_y
        conv_borders.weight.data[11, 2, :, :] = -sobel_y
        # use a slightly negative bias to suppress weak responses
        conv_borders.bias.data[:] = -0.1
        return conv_borders
    
    def borders_layer(self) -> torch.nn.Conv2d:
        # This layer sums up signals for edges across colors and polarities,
        # but keeps the 2 orientations in different channels
        conv_bordertypes = torch.nn.Conv2d(12, 2, 1)
        conv_bordertypes.bias.data.zero_()
        conv_bordertypes.weight.data.zero_()
        # The first channel sums up all vertical edges
        conv_bordertypes.weight.data[0, 0:6, :, :] = 1.0
        # The second channels sums up all horizontal edges
        conv_bordertypes.weight.data[1, 6:12, :, :] = 1.0
        return conv_bordertypes
    
    def distance_layer(self) -> torch.nn.Conv2d:
        # This layer calculates the distance from
        #  (1) vertical borders on the left
        #  (2) vertical borders on the right
        #  (3) horizontal borders above
        #  (4) horizontal borders below
        # First, we generate masks for the distance calculation 
        distances = torch.arange(-6, 7.).unsqueeze(0).repeat(13, 1)
        distance_to_right = distances.relu()
        distance_to_left = (-distances).relu()
        distance_to_bottom = distances.T.relu()
        distance_to_top = (-distances).T.relu()
        # Then we use these masks to set the weights
        distance_layer = torch.nn.Conv2d(2, 4, 13, padding=7)
        distance_layer.weight.data.zero_()
        distance_layer.bias.data.zero_()
        # first channel computes distance to vertical line to the left
        distance_layer.weight.data[0, 0] = distance_to_left.sqrt()
        # second channel computes distance to vertical line to the right
        distance_layer.weight.data[1, 0] = distance_to_right.sqrt()
        # third channel computes distance to horizontal line above
        distance_layer.weight.data[2, 1] = distance_to_top.sqrt()
        # fourth channel computes distance to horizontal line below
        distance_layer.weight.data[3, 1] = distance_to_bottom.sqrt()
        return distance_layer
    
    def decision_layer(self) -> torch.nn.Linear:
        # It checks whether the distance to left-right vertical borders
        # is larger (-> rectangle is wider than high) or if the distance
        # to top-bottom horizontal borders is larger (-> rectangle is higher than wide)
        layer = torch.nn.Linear(4, 2)
        layer.bias.data.zero_()
        layer.weight.data[0] = torch.tensor([1.0, 1.0, 0.0, 0.0])
        layer.weight.data[1] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        return layer