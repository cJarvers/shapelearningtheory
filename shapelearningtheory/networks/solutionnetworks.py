# Small networks with pre-specified weights that "solve" the datasets with pre-specified strategies.
import numpy as np
import torch
from skimage.filters import gabor_kernel
from collections import OrderedDict

class FixedSequential(torch.nn.Sequential):
    """Sequential model that has a fixed structure (constructor does not take arguments),
    but that can still be subscripted correctly."""
    def __init__(self):
        super().__init__()

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return torch.nn.Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

############
# Networks #
############
class ColorConvNet(FixedSequential):
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


class CRectangleConvNet(FixedSequential):
    """ConvNet that classifies color rectangles by height / width."""
    def __init__(self):
        super().__init__()
        self.append(SobelLayer())
        self.append(torch.nn.ReLU())
        self.append(SumChannels(12, 2))
        self.append(self.distance_layer())
        self.append(torch.nn.AdaptiveMaxPool2d(output_size=1))
        self.append(torch.nn.Flatten())
        self.append(self.decision_layer())
    
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
    
class SRectangleConvNet(FixedSequential):
    """ConvNet that classifies striped rectangles by shape."""
    def __init__(self):
        super().__init__()
    
class TextureConvNet(FixedSequential):
    """Convolutional network that classifies input images based on
    whether they contain a horizontal or vertical texture."""
    def __init__(self):
        super().__init__()
        self.append(GaborLayer())
        self.append(torch.nn.ReLU())
        self.append(SumChannels(24, 2))
        self.append(torch.nn.AdaptiveMaxPool2d(output_size=1))
        self.append(torch.nn.Flatten())

class CLTConvNet(FixedSequential):
    """ConvNet that classifies colorful L and T shapes by shape."""
    def __init__(self):
        super().__init__()

#################
# Helper layers #
#################
class LaplaceLayer(torch.nn.Conv2d):
    """Convolution layer that performs Laplace filtering."""
    def __init__(self, in_channels=3):
        super().__init__(
            in_channels = in_channels,
            out_channels = 2,
            kernel_size = 3,
            bias = True,
            padding = "same",
            padding_mode = "replicate"
        )
        with torch.no_grad():
            self.bias.data[:] = -0.01
            laplace_kernel = torch.tensor(
                [[0, 1, 0],
                 [1, -4, 1],
                 [0, 1, 0]]
            )
            self.weight.data[0, :, :, :] = laplace_kernel
            self.weight.data[1, :, :, :] = -laplace_kernel


class Heaviside(torch.nn.Module):
    """Heaviside activation function."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.heaviside(values=torch.zeros_like(x))

class GaborLayer(torch.nn.Conv2d):
    """Convolution layer that is initialized with a Gabor filter bank."""
    def __init__(self, frequency=0.1, bandwidth=3):
        horizontal_gabor = gabor_kernel(
            frequency=frequency, theta=np.pi/2, bandwidth=bandwidth)
        vertical_gabor = gabor_kernel(
            frequency=frequency, theta=0, bandwidth=bandwidth)
        kernel_size = horizontal_gabor.shape
        super().__init__(
            in_channels=3,
            out_channels=24,
            kernel_size=kernel_size,
            bias=False, 
            padding="same",
            padding_mode="replicate"
        )
        with torch.no_grad():
            self.weight.data.zero_()
            self.weight[ 0, 0, :, :] = torch.from_numpy(  horizontal_gabor.real - horizontal_gabor.real.mean())
            self.weight[ 1, 0, :, :] = torch.from_numpy(- horizontal_gabor.real + horizontal_gabor.real.mean())
            self.weight[ 2, 0, :, :] = torch.from_numpy(  horizontal_gabor.imag)
            self.weight[ 3, 0, :, :] = torch.from_numpy(- horizontal_gabor.imag)
            self.weight[ 4, 1, :, :] = torch.from_numpy(  horizontal_gabor.real - horizontal_gabor.real.mean())
            self.weight[ 5, 1, :, :] = torch.from_numpy(- horizontal_gabor.real + horizontal_gabor.real.mean())
            self.weight[ 6, 1, :, :] = torch.from_numpy(  horizontal_gabor.imag)
            self.weight[ 7, 1, :, :] = torch.from_numpy(- horizontal_gabor.imag)
            self.weight[ 8, 2, :, :] = torch.from_numpy(  horizontal_gabor.real - horizontal_gabor.real.mean())
            self.weight[ 9, 2, :, :] = torch.from_numpy(- horizontal_gabor.real + horizontal_gabor.real.mean())
            self.weight[10, 2, :, :] = torch.from_numpy(  horizontal_gabor.imag)
            self.weight[11, 2, :, :] = torch.from_numpy(- horizontal_gabor.imag)
            self.weight[12, 0, :, :] = torch.from_numpy(  vertical_gabor.real - vertical_gabor.real.mean())
            self.weight[13, 0, :, :] = torch.from_numpy(- vertical_gabor.real + vertical_gabor.real.mean())
            self.weight[14, 0, :, :] = torch.from_numpy(  vertical_gabor.imag)
            self.weight[15, 0, :, :] = torch.from_numpy(- vertical_gabor.imag)
            self.weight[16, 1, :, :] = torch.from_numpy(  vertical_gabor.real - vertical_gabor.real.mean())
            self.weight[17, 1, :, :] = torch.from_numpy(- vertical_gabor.real + vertical_gabor.real.mean())
            self.weight[18, 1, :, :] = torch.from_numpy(  vertical_gabor.imag)
            self.weight[19, 1, :, :] = torch.from_numpy(- vertical_gabor.imag)
            self.weight[20, 2, :, :] = torch.from_numpy(  vertical_gabor.real - vertical_gabor.real.mean())
            self.weight[21, 2, :, :] = torch.from_numpy(- vertical_gabor.real + vertical_gabor.real.mean())
            self.weight[22, 2, :, :] = torch.from_numpy(  vertical_gabor.imag)
            self.weight[23, 2, :, :] = torch.from_numpy(- vertical_gabor.imag)

class SobelLayer(torch.nn.Conv2d):
    """Convolution layer that performs Sobel filtering in x and y direction."""
    def __init__(self, in_channels=3):
        super().__init__(
            in_channels = in_channels,
            out_channels = 4 * in_channels,
            kernel_size = 3,
            bias=True,
            padding="same",
            padding_mode="replicate"
        )
        sobel_x = torch.tensor([[-0.25, 0., 0.25], [-0.5, 0., 0.5], [-0.25, 0., 0.25]])
        sobel_y = sobel_x.T
        with torch.no_grad():
            self.weight.data.zero_()
            for i in range(in_channels):
                self.weight.data[2*i, i, :, :] = sobel_x
                self.weight.data[2*i+1, i, :, :] = -sobel_x
                n = in_channels
                self.weight.data[2*n+2*i, i, :, :] = sobel_y
                self.weight.data[2*n+2*i+1, i, :, :] = -sobel_y
            self.bias.data[:] = -0.1

class SumChannels(torch.nn.Conv2d):
    """1x1 convolution that sums over channels in equal groups."""
    def __init__(self, in_channels: int, groups: int):
        super().__init__(
            in_channels = in_channels,
            out_channels = groups,
            kernel_size = 1,
            bias = False,
        )
        assert(in_channels % groups == 0 and in_channels > groups)
        channels_per_group = in_channels // groups
        with torch.no_grad():
            self.weight.data.zero_()
            for i in range(groups):
                start = i * channels_per_group
                stop = start + channels_per_group
                self.weight.data[i, start:stop, :] = 1.0