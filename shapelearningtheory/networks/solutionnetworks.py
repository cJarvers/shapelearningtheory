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
    def __init__(self):
        super().__init__()
        self.append(self.make_rbdiff_layer())
        self.append(torch.nn.ReLU())
        self.append(torch.nn.AdaptiveMaxPool2d(output_size=1))
        self.append(torch.nn.Flatten())
        self.append(self.make_colorclass_layer())

    def get_layers_of_interest(self):
        return {
            "R-B diff": 2,
            "pooling": 3,
            "color class": 5
        }

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
        max_for_NotRedXORBlue = 0.1
        min_for_RedXORBlue = 0.8
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
        self.append(DistanceLayer(kernel_size=13))
        self.append(torch.nn.AdaptiveMaxPool2d(output_size=1))
        self.append(torch.nn.Flatten())
        self.append(self.decision_layer())

    def get_layers_of_interest(self):
        return {
            "Sobel": 2,
            "borders": 3,
            "distance": 4,
            "shape class": 7
        }
    
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
        self.append(LaplaceLayer())
        self.append(RootLU())
        self.append(SumChannels(in_channels=2, groups=1))
        self.append(SobelLayer(in_channels=1, bias_level=-0.3))
        self.append(RootLU(coeff=0.5))
        self.append(SumChannels(4, 2))
        self.append(DistanceLayer(13))
        self.append(torch.nn.AdaptiveMaxPool2d(output_size=1))
        self.append(SumChannels(in_channels=4, groups=2))
        self.append(torch.nn.Flatten())

    def get_layers_of_interest(self):
        return {
            "Laplace": 2,
            "Sobel": 5,
            "borders": 6,
            "distance": 7,
            "decision": 10
        }

    
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

    def get_layers_of_interest(self):
        return {
            "Gabor": 2,
            "orientation": 3,
            "texture class": 5
        }

class LTConvNet(FixedSequential):
    """ConvNet that classifies letters L and T by shape.
    Works for color and stripe version."""
    def __init__(self):
        super().__init__()
        self.append(LaplaceLayer())
        self.append(RootLU())
        self.append(SumChannels(in_channels=2, groups=1))
        self.append(SobelLayer(in_channels=1, bias_level=-0.3))
        self.append(RootLU(coeff=0.5))
        self.append(SumChannels(4, 1))
        self.append(EndDetector(inhibition_factor=1.5))
        self.append(RootLU())
        self.append(torch.nn.AdaptiveMaxPool2d(1))
        self.append(SumChannels(4, 1))
        self.append(self.make_decision_layer())
        self.append(torch.nn.Flatten())

    def get_layers_of_interest(self):
        return {
            "Laplace": 2,
            "Sobel": 3,
            "borders": 6,
            "line ends": 8,
            "shape class": 11
        }

    def make_decision_layer(self):
        layer = torch.nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=1
        )
        with torch.no_grad():
            layer.weight[0, :] = -1.0
            layer.weight[1, :] = 1.0
            layer.bias[0] = 2.5
            layer.bias[1] = -2.5
        return layer

#################
# Helper layers #
#################
class LaplaceLayer(torch.nn.Conv2d):
    """Convolution layer that performs Laplace filtering."""
    def __init__(self, in_channels=3, bias_level=-0.01):
        super().__init__(
            in_channels = in_channels,
            out_channels = 2,
            kernel_size = 3,
            bias = True,
            padding = "same",
            padding_mode = "replicate"
        )
        self.bias_level = bias_level
        with torch.no_grad():
            self.bias.data[:] = self.bias_level
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
    
class RootLU(torch.nn.Module):
    """ReLU combined with root function (to squash output range)."""
    def __init__(self, coeff=0.1):
        super().__init__()
        self.coeff = coeff
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.relu().pow(self.coeff)

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
    def __init__(self, in_channels=3, bias_level=-0.1):
        super().__init__(
            in_channels = in_channels,
            out_channels = 4 * in_channels,
            kernel_size = 3,
            bias=True,
            padding="same",
            padding_mode="replicate"
        )
        self.bias_level = bias_level
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
            self.bias.data[:] = bias_level

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

class DistanceLayer(torch.nn.Conv2d):
    def __init__(self, kernel_size: int = 13):
        super().__init__(
            in_channels = 2,
            out_channels = 4,
            kernel_size = kernel_size,
            padding = kernel_size // 2
        )
        # generate masks for the distance calculation 
        self.weight.data.zero_()
        self.bias.data.zero_()
        distances = self.distance_masks(kernel_size)
        self.weight.data[0, 0] = distances[0].sqrt()
        self.weight.data[1, 0] = distances[1].sqrt()
        self.weight.data[2, 1] = distances[2].sqrt()
        self.weight.data[3, 1] = distances[3].sqrt()

    def distance_masks(self, kernel_size):
        distances = torch.arange(-(kernel_size//2), kernel_size // 2 + 1.0)
        distances = distances.unsqueeze(0).repeat(kernel_size, 1)
        distance_to_right = distances.relu()
        distance_to_left = (-distances).relu()
        distance_to_bottom = distances.T.relu()
        distance_to_top = (-distances).T.relu()
        return distance_to_right, distance_to_left, distance_to_bottom, distance_to_top

class EndDetector(torch.nn.Conv2d):
    """Convolution layer that detects line ends in 4 possible orientations."""
    def __init__(self, inhibition_factor=1.0, bias_level=0.0):
        super().__init__(
            in_channels=1,
            out_channels=4,
            kernel_size=5,
            bias=True,
            padding=2
        )
        self.inhibition_factor = inhibition_factor
        self.bias_level = bias_level
        self.bias.data[:] = bias_level
        top_end, left_end, bottom_end, right_end = self.create_end_masks()
        self.weight.data[0, :] = top_end
        self.weight.data[1, :] = bottom_end
        self.weight.data[2, :] = right_end
        self.weight.data[3, :] = left_end

    def create_end_masks(self):
        center_mask = torch.zeros(5,5)
        center_mask[2, 2] = 1.0 
        inhibition_mask = torch.tensor([
            [-1, -1, -1, -1, -1.],
            [-1, -1, -1, -1, -1],
            [-1,  0,  0,  0, -1],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0]
        ]) * self.inhibition_factor
        end_mask = center_mask + inhibition_mask
        top_end = end_mask
        left_end = end_mask.T
        bottom_end = end_mask.flipud()
        right_end = bottom_end.T
        return top_end, left_end, bottom_end, right_end