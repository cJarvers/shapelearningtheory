# Integration tests for solution networks
# -> check that they classify their target datasets correctly
import unittest
import torch
from shapelearningtheory.networks import ColorConvNet, CRectangleConvNet, TextureConvNet
from shapelearningtheory.datasets import make_rectangles_wrong_color,  \
    make_rectangles_wrong_texture, make_LT_wrong_color, make_LT_wrong_texture

def get_accuracy(net, data):
    accuracies = []
    for (images, labels) in data.test_dataloader():
        predictions = net(images).argmax(dim=1)
        accuracy = (predictions == labels).to(torch.float32).mean().item()
        accuracies.append(accuracy)
    average_accuracy = sum(accuracies) / len(accuracies)
    return average_accuracy

class TestColorConvnet(unittest.TestCase):
    def test_onconflict_rectangles(self):
        data = make_rectangles_wrong_color()
        data.prepare_data()
        net = ColorConvNet(data.dataset.imgheight, data.dataset.imgwidth)
        average_accuracy = get_accuracy(net, data)
        self.assertEqual(average_accuracy, 0)

    def test_onconflict_lvt(self):
        data = make_LT_wrong_color()
        data.prepare_data()
        net = ColorConvNet(data.dataset.imgheight, data.dataset.imgwidth,
                           min_pixels=13, max_pixels=69)
        average_accuracy = get_accuracy(net, data)
        self.assertEqual(average_accuracy, 0)

class TestCRectangleConvNet(unittest.TestCase):
    def test_onconflict(self):
        data = make_rectangles_wrong_color()
        data.prepare_data()
        net = CRectangleConvNet()
        average_accuracy = get_accuracy(net, data)
        self.assertEqual(average_accuracy, 1.0)


class TestTextureConvNet(unittest.TestCase):
    def test_on_conflict_rectangles(self):
        data = make_rectangles_wrong_texture()
        data.prepare_data()
        net = TextureConvNet()
        average_accuracy = get_accuracy(net, data)
        self.assertEqual(average_accuracy, 0.0)

    @unittest.expectedFailure # not quite working yet
    def test_on_conflict_lvt(self):
        data = make_LT_wrong_texture()
        data.prepare_data()
        net = TextureConvNet()
        average_accuracy = get_accuracy(net, data)
        self.assertEqual(average_accuracy, 0.0)