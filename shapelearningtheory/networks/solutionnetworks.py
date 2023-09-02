# Small networks with pre-specified weights that "solve" the datasets with pre-specified strategies.
import torch

def make_color_mlp(imageheight, imagewidth, min_pixels=35, max_pixels=117):
    """Generate a 2-layer neural network with 4 neurons that classifies images
    as RedXORBlue or NotRedXORBlue. It assumes that the background has homogenous
    color and calculates the (average) difference between R and B channels. If this
    exceeds a threshold, the image is RedXORBlue"""
    # Create first layer: 2 neurons
    layer1 = torch.nn.Linear(imageheight * imagewidth * 3, 2)
    # First neuron detects if red is much larger than blue.
    red_minus_blue = torch.zeros(3, imageheight, imagewidth)
    red_minus_blue[0, :, :] = 1.0
    red_minus_blue[2, :, :] = -1.0
    layer1.weight[0] = red_minus_blue.flatten()
    # Second neuron detects of blue is much larger than red
    blue_minus_red = torch.zeros(3, imageheight, imagewidth)
    blue_minus_red[0, :, :] = -1.0
    blue_minus_red[2, :, :] = 1.0
    layer1.weight[1] = blue_minus_red.flatten()
    # Set bias
    layer1.bias.data = torch.ones(2) * -0.1
    #
    # Create second layer: 2 neurons
    layer2 = torch.nn.Linear(2, 2)
    # If either of the R-B or B-R differences is large, we have RedXORBlue, otherwise NotRedXORBlue.
    # For NotRedXORBlue, the maximum difference between R and B at each pixel is 0.1.
    # For RedXORBlue, the minimum difference between R and B at each pixel is 0.8.
    # To choose a threshold, we need to scale this by the max/min number of pixels of the foreground object.
    max_for_NotRedXORBlue = max_pixels * 0.1
    min_for_RedXORBlue = min_pixels * 0.8
    if max_for_NotRedXORBlue > min_for_RedXORBlue:
        raise ValueError("Cannot correctly set threshold.")
    threshold = (min_for_RedXORBlue + max_for_NotRedXORBlue) / 2
    layer2.weight.data = torch.tensor([[1.0, 1.0],[-1.0, -1.0]])
    layer2.bias.data = torch.tensor([-threshold, threshold])
    #
    # Create network:
    return torch.nn.Sequential(layer1, torch.nn.ReLU(), layer2)


def make_color_convnet():
    raise NotImplementedError()


def make_rectangle_convnet():
    # Create first layer:
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
    #
    # Create second layer:
    # This layer sums up signals for edges across colors and polarities,
    # but keeps the 2 orientations in different channels
    conv_bordertypes = torch.nn.Conv2d(12, 2, 1)
    conv_bordertypes.bias.data.zero_()
    conv_bordertypes.weight.data.zero_()
    # The first channel sums up all vertical edges
    conv_bordertypes.weight.data[0, 0:6, :, :] = 1.0
    # The second channels sums up all horizontal edges
    conv_bordertypes.weight.data[1, 6:12, :, :] = 1.0
    #
    # Create third layer:
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
    #
    # Create final decision layer:
    # It checks whether the distance to left-right vertical borders
    # is larger (-> rectangle is wider than high) or if the distance
    # to top-bottom horizontal borders is larger (-> rectangle is higher than wide)
    decision_layer = torch.nn.Conv2d(4, 2, 1)
    decision_layer.weight.data.zero_()
    decision_layer.bias.data.zero_()
    decision_layer.weight.data[0] = torch.tensor([1.0, 1.0, 0.0, 0.0]).unsqueeze(1).unsqueeze(2)
    decision_layer.weight.data[1] = torch.tensor([0.0, 0.0, 1.0, 1.0]).unsqueeze(1).unsqueeze(2)
    #
    # Set up network:
    return torch.nn.Sequential(
        conv_borders,
        torch.nn.ReLU(),
        conv_bordertypes, # next three layers can only return non-negative values, so ReLU is left out
        distance_layer,
        decision_layer,
        torch.nn.AdaptiveMaxPool2d(output_size=1),
        torch.nn.Flatten()
    )