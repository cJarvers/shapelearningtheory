# Integration tests for solution networks
# -> check that they classify their target datasets correctly
import unittest
import torch
from shapelearningtheory.networks import ColorConvNet, CRectangleConvNet
from shapelearningtheory.datasets import make_rectangles_wrong_color, make_LT_wrong_color

class TestColorConvnet(unittest.TestCase):
    def test_onconflict_rectangles(self):
        data = make_rectangles_wrong_color()
        data.prepare_data()
        net = ColorConvNet(data.dataset.imgheight, data.dataset.imgwidth)
        accuracies = []
        for (images, labels) in data.test_dataloader():
            predictions = net(images).argmax(dim=1)
            accuracy = (predictions == labels).to(torch.float32).mean().item()
            accuracies.append(accuracy)
        average_accuracy = sum(accuracies) / len(accuracies)
        self.assertEqual(average_accuracy, 0)

    def test_onconflict_lvt(self):
        data = make_LT_wrong_color()
        data.prepare_data()
        net = ColorConvNet(data.dataset.imgheight, data.dataset.imgwidth,
                           min_pixels=13, max_pixels=69)
        accuracies = []
        for (images, labels) in data.test_dataloader():
            predictions = net(images).argmax(dim=1)
            accuracy = (predictions == labels).to(torch.float32).mean().item()
            accuracies.append(accuracy)
        average_accuracy = sum(accuracies) / len(accuracies)
        self.assertEqual(average_accuracy, 0)

class TestCRectangleConvNet(unittest.TestCase):
    def test_onconflict(self):
        data = make_rectangles_wrong_color()
        data.prepare_data()
        net = CRectangleConvNet()
        accuracies = []
        for (images, labels) in data.test_dataloader():
            predictions = net(images).argmax(dim=1)
            accuracy = (predictions == labels).to(torch.float32).mean().item()
            accuracies.append(accuracy)
        average_accuracy = sum(accuracies) / len(accuracies)
        self.assertEqual(average_accuracy, 1.0)
