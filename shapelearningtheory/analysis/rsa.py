# helper functions for representational similarity analysis
import numpy as np
import rsatoolbox
import torch
# local imports
from ..datasets.rectangledataset import RectangleDataset
from ..datasets.LTPlusDataset import LTDataset
from ..old_shapes import Orientation
from .helpers import get_activations_dataset
###############################
# Functions to get RDM groups #
###############################
@torch.no_grad()
def get_model_RDMs(model, dataloader, use_image=True):
    activations = get_activations_dataset(model, dataloader, use_image=use_image)
    layer_activations = [
        rsatoolbox.data.Dataset(activation.flatten(start_dim=1).numpy(), descriptors={"layer": name})
        for name, activation in activations.items()
    ]
    rdms = [
        rsatoolbox.rdm.calc_rdm(activation, method="euclidean")
        for activation in layer_activations
    ]
    return rsatoolbox.rdm.rdms.concat(rdms)

def get_shape_RDMs(dataset):
    """
    Compute representational dissimilarity matrices for shape properties of
    `dataset`.
    """
    if isinstance(dataset, RectangleDataset):
        properties = ["orientation", "position", "height", "width"]
        rdms = [
            get_rectangle_orientation_RDM(dataset),
            get_rectangle_position_RDM(dataset),
            get_rectangle_height_RDM(dataset),
            get_rectangle_width_RDM(dataset)
        ]
    elif isinstance(dataset, LTDataset):
        properties = ["orientation", "height", "width", "line strength", "class"]
        rdms = [
            get_LT_orientation_RDM(dataset),
            get_LT_height_RDM(dataset),
            get_LT_width_RDM(dataset),
            get_LT_linestrength_RDM(dataset),
            get_LT_class_RDM(dataset)
        ]
    rdms = [np.expand_dims(rdm, 0) for rdm in rdms]
    return rsatoolbox.rdm.RDMs(
        np.concatenate(rdms, axis=0),
        dissimilarity_measure="Euclidean",
        rdm_descriptors={
            "property": properties,
            "feature_type": ["shape"] * len(rdms)}
    )

def get_color_RDMs(dataset):
    if isinstance(dataset, RectangleDataset):
        properties = ["R", "G", "B", "abs(R-B)", "color"]
        rdms = [
            get_rectangle_redness_RDM(dataset),
            get_rectangle_greenness_RDM(dataset),
            get_rectangle_blueness_RDM(dataset),
            get_rectangle_redblue_RDM(dataset),
            get_rectangle_color_RDM(dataset)
        ]
    elif isinstance(dataset, LTDataset):
        properties = ["R", "G", "B", "abs(R-B)", "color"]
        rdms = [
            get_LT_redness_RDM(dataset),
            get_LT_greenness_RDM(dataset),
            get_LT_blueness_RDM(dataset),
            get_LT_redblue_RDM(dataset),
            get_LT_color_RDM(dataset)
        ]
    rdms = [np.expand_dims(rdm, 0) for rdm in rdms]
    return rsatoolbox.rdm.RDMs(
        np.concatenate(rdms, axis=0),
        dissimilarity_measure="Euclidean",
        rdm_descriptors={
            "property": properties,
            "feature_type": ["color"] * len(rdms)}
    )

def get_texture_RDMs(dataset):
    """
    Compute representational dissimilarity matrices for texture properties
    of `dataset`."""
    properties = ["theta", "frequency"]
    if isinstance(dataset, RectangleDataset):
        rdms = [
            get_rectangle_texture_orientation_RDM(dataset),
            get_rectangle_texture_frequency_RDM(dataset)
        ]
    elif isinstance(dataset, LTDataset):
        rdms = [
            get_LT_texture_orientation_RDM(dataset),
            get_LT_texture_frequency_RDM(dataset)
        ]
    rdms = [np.expand_dims(rdm, 0) for rdm in rdms]
    return rsatoolbox.rdm.RDMs(
        np.concatenate(rdms, axis=0),
        dissimilarity_measure="Euclidean",
        rdm_descriptors={
            "property": properties,
            "feature_type": ["texture"] * len(rdms)}
    )


############################################################
# Functions to compute specific RDMs for specific datasets #
############################################################
def get_rectangle_orientation_RDM(dataset: RectangleDataset):
    orientations = np.array([
        rectangle.shape.orientation for rectangle in dataset.rectangles
    ])
    orientations = np.expand_dims(orientations, 1)
    rdm = orientations.T != orientations
    return rdm

def get_rectangle_height_RDM(dataset: RectangleDataset):
    heights = np.array([
        rectangle.shape.length 
        if rectangle.shape.orientation == Orientation.VERTICAL
        else rectangle.shape.width
        for rectangle in dataset.rectangles
    ])
    heights = np.expand_dims(heights, 1)
    rdm = np.abs(heights.T - heights)
    return rdm

def get_rectangle_width_RDM(dataset: RectangleDataset):
    widths = np.array([
        rectangle.shape.length 
        if rectangle.shape.orientation == Orientation.HORIZONTAL
        else rectangle.shape.width
        for rectangle in dataset.rectangles
    ])
    widths = np.expand_dims(widths, 1)
    rdm = np.abs(widths.T - widths)
    return rdm

def get_rectangle_position_RDM(dataset: RectangleDataset):
    positions = [rectangle.shape.get_position() for rectangle in dataset.rectangles]
    x_positions = np.expand_dims(np.array([position[0] for position in positions]), 1)
    y_positions = np.expand_dims(np.array([position[1] for position in positions]), 1)
    x_distances = x_positions.T - x_positions
    y_distances = y_positions.T - y_positions
    rdm = np.sqrt(np.power(x_distances, 2) + np.power(y_distances, 2))
    return rdm

def get_rectangle_color_RDM(dataset: RectangleDataset):
    colors = [rectangle.pattern.color for rectangle in dataset.rectangles]
    colors = torch.cat(colors).squeeze()
    rdm = torch.cdist(colors, colors).numpy()
    return rdm

def get_rectangle_redness_RDM(dataset: RectangleDataset):
    red_intensity = [rectangle.pattern.color.squeeze()[0].item() for rectangle in dataset.rectangles]
    red_intensity = np.expand_dims(np.array(red_intensity), 1)
    rdm = np.abs(red_intensity.T - red_intensity)
    return rdm

def get_rectangle_greenness_RDM(dataset: RectangleDataset):
    green_intensity = [rectangle.pattern.color.squeeze()[1].item() for rectangle in dataset.rectangles]
    green_intensity = np.expand_dims(np.array(green_intensity), 1)
    rdm = np.abs(green_intensity.T - green_intensity)
    return rdm

def get_rectangle_blueness_RDM(dataset: RectangleDataset):
    blue_intensity = [rectangle.pattern.color.squeeze()[2].item() for rectangle in dataset.rectangles]
    blue_intensity = np.expand_dims(np.array(blue_intensity), 1)
    rdm = np.abs(blue_intensity.T - blue_intensity)
    return rdm

def get_rectangle_redblue_RDM(dataset: RectangleDataset):
    red_intensity = [rectangle.pattern.color.squeeze()[0].item() for rectangle in dataset.rectangles]
    blue_intensity = [rectangle.pattern.color.squeeze()[2].item() for rectangle in dataset.rectangles]
    difference = np.abs(np.array(red_intensity) - np.array(blue_intensity))
    difference = np.expand_dims(difference, 1)
    rdm = np.abs(difference.T - difference)
    return rdm


def get_rectangle_texture_orientation_RDM(dataset: RectangleDataset):
    texture_orientations = np.array([
        rectangle.pattern.orientation for rectangle in dataset.rectangles
    ])
    texture_orientations = np.expand_dims(texture_orientations, 1)
    rdm = 1 - np.cos(texture_orientations.T - texture_orientations)
    return rdm

def get_rectangle_texture_frequency_RDM(dataset: RectangleDataset):
    texture_frequencies = np.array([
        rectangle.pattern.frequency for rectangle in dataset.rectangles
    ])
    texture_frequencies = np.expand_dims(texture_frequencies, 1)
    rdm = np.abs(texture_frequencies.T - texture_frequencies)
    return rdm

def get_LT_height_RDM(dataset: LTDataset):
    l_heights = np.array([l.shape.height for l in dataset.ls])
    t_heights = np.array([t.shape.height for t in dataset.ts])
    heights = np.concatenate([l_heights, t_heights])
    heights = np.expand_dims(heights, 1)
    rdm = np.abs(heights.T - heights)
    return rdm

def get_LT_width_RDM(dataset: LTDataset):
    l_widths = np.array([l.shape.width for l in dataset.ls])
    t_widths = np.array([t.shape.width for t in dataset.ts])
    widths = np.concatenate([l_widths, t_widths])
    widths = np.expand_dims(widths, 1)
    rdm = np.abs(widths.T - widths)
    return rdm

def get_LT_orientation_RDM(dataset: LTDataset):
    def corner_to_angle(corner):
        if corner == "topright":
            return 1/4 * np.pi
        elif corner == "topleft":
            return 3/4 * np.pi
        elif corner == "bottomleft":
            return 5/4 * np.pi
        elif corner == "bottomright":
            return 7/4 * np.pi
    def topside_to_angle(topside):
        if topside == "top":
            return 1/2 * np.pi
        elif topside == "left":
            return np.pi
        elif topside == "bottom":
            return 3/2 * np.pi
        elif topside == "right":
            return 0
    l_angles = np.array([
        corner_to_angle(l.shape.corner) for l in dataset.ls
    ])
    t_angles = np.array([
        topside_to_angle(t.shape.topside) for t in dataset.ts
    ])
    angles = np.concatenate([l_angles, t_angles])
    angles = np.expand_dims(angles, 1)
    rdm = 1 - np.cos(angles.T - angles)
    return rdm

def get_LT_linestrength_RDM(dataset: LTDataset):
    l_strengths = np.array([l.shape.strength for l in dataset.ls])
    t_strengths = np.array([t.shape.strength for t in dataset.ts])
    strengths = np.concatenate([l_strengths, t_strengths])
    strengths = np.expand_dims(strengths, 1)
    rdm = np.abs(strengths.T - strengths)
    return rdm

def get_LT_class_RDM(dataset: LTDataset):
    same = np.zeros((len(dataset.ls), len(dataset.ls)))
    different = np.ones((len(dataset.ls), len(dataset.ls)))
    upper = np.concatenate([same, different], axis=0)
    lower = np.concatenate([different, same], axis=0)
    rdm = np.concatenate([upper, lower], axis=1)
    return rdm

def get_LT_redness_RDM(dataset: LTDataset):
    l_red = [l.pattern.color.squeeze()[0].item() for l in dataset.ls]
    t_red = [t.pattern.color.squeeze()[0].item() for t in dataset.ts]
    red_intensity = l_red + t_red
    red_intensity = np.expand_dims(np.array(red_intensity), 1)
    rdm = np.abs(red_intensity.T - red_intensity)
    return rdm

def get_LT_greenness_RDM(dataset: LTDataset):
    l_green = [l.pattern.color.squeeze()[1].item() for l in dataset.ls]
    t_green = [t.pattern.color.squeeze()[1].item() for t in dataset.ts]
    green_intensity = l_green + t_green
    green_intensity = np.expand_dims(np.array(green_intensity), 1)
    rdm = np.abs(green_intensity.T - green_intensity)
    return rdm

def get_LT_blueness_RDM(dataset: LTDataset):
    l_blue = [l.pattern.color.squeeze()[2].item() for l in dataset.ls]
    t_blue = [t.pattern.color.squeeze()[2].item() for t in dataset.ts]
    blue_intensity = l_blue + t_blue
    blue_intensity = np.expand_dims(np.array(blue_intensity), 1)
    rdm = np.abs(blue_intensity.T - blue_intensity)
    return rdm

def get_LT_redblue_RDM(dataset: LTDataset):
    l_red = [l.pattern.color.squeeze()[0].item() for l in dataset.ls]
    t_red = [t.pattern.color.squeeze()[0].item() for t in dataset.ts]
    red_intensity = l_red + t_red
    l_blue = [l.pattern.color.squeeze()[2].item() for l in dataset.ls]
    t_blue = [t.pattern.color.squeeze()[2].item() for t in dataset.ts]
    blue_intensity = l_blue + t_blue
    difference = np.abs(np.array(red_intensity) - np.array(blue_intensity))
    difference = np.expand_dims(difference, 1)
    rdm = np.abs(difference.T - difference)
    return rdm

def get_LT_color_RDM(dataset: LTDataset):
    l_colors = [l.pattern.color for l in dataset.ls]
    t_colors = [t.pattern.color for t in dataset.ts]
    colors = l_colors + t_colors
    colors = torch.cat(colors).squeeze()
    rdm = torch.cdist(colors, colors).numpy()
    return rdm

def get_LT_texture_orientation_RDM(dataset: LTDataset):
    l_orientations = [l.pattern.orientation for l in dataset.ls]
    t_orientations = [t.pattern.orientation for t in dataset.ts]
    texture_orientations = np.array(l_orientations + t_orientations)
    texture_orientations = np.expand_dims(texture_orientations, 1)
    rdm = 1 - np.cos(texture_orientations.T - texture_orientations)
    return rdm

def get_LT_texture_frequency_RDM(dataset: LTDataset):
    l_frequencies = [l.pattern.frequency for l in dataset.ls]
    t_frequencies = [t.pattern.frequency for t in dataset.ts]
    texture_frequencies = np.array(l_frequencies + t_frequencies)
    texture_frequencies = np.expand_dims(texture_frequencies, 1)
    rdm = np.abs(texture_frequencies.T - texture_frequencies)
    return rdm