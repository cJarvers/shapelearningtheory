# Integration tests for solution networks
# -> check that they classify their target datasets correctly
import unittest
import torch
from shapelearningtheory.networks import ColorConvNet, CRectangleConvNet, \
    SRectangleConvNet, TextureConvNet, LTConvNet
from shapelearningtheory.datasets import make_rectangles_wrong_color,  \
    make_rectangles_wrong_texture, make_LT_wrong_color, make_LT_wrong_texture, \
    make_rectangles_color, make_rectangles_texture, make_LT_color, make_LT_texture

class DataSetTest(unittest.TestCase):
    """Test case subclass to run test on dataset."""
    def evaluate_network(self, net, data, expected_accuracy=1.0, delta=0.0):
        data.prepare_data()
        average_accuracy = get_accuracy(net, data)
        self.assertAlmostEqual(average_accuracy, expected_accuracy, delta=delta)

def get_accuracy(net, data):
    accuracies = []
    for (images, labels) in data.test_dataloader():
        predictions = net(images).argmax(dim=1)
        accuracy = (predictions == labels).to(torch.float32).mean().item()
        accuracies.append(accuracy)
    average_accuracy = sum(accuracies) / len(accuracies)
    return average_accuracy

class TestColorConvnet(DataSetTest):
    def test_onconflict_rectangles(self):
        data = make_rectangles_wrong_color()
        data.prepare_data()
        self.evaluate_network(
            ColorConvNet(
                data.dataset.imgheight,
                data.dataset.imgwidth),
            data,
            expected_accuracy=0.0
        )

    def test_onconflict_lvt(self):
        data = make_LT_wrong_color()
        data.prepare_data()
        self.evaluate_network(
            ColorConvNet(
                data.dataset.imgheight,
                data.dataset.imgwidth,
                min_pixels=13, max_pixels=69),
            data,
            expected_accuracy=0.0
        )

class TestCRectangleConvNet(DataSetTest):
    def test_onconflict(self):
        self.evaluate_network(
            CRectangleConvNet(),
            make_rectangles_wrong_color(),
            expected_accuracy=1.0
        )

class TestSRectangleConvNet(DataSetTest):
    def test_on_striped_rectangles(self):
        self.evaluate_network(
            SRectangleConvNet(),
            make_rectangles_texture(),
            expected_accuracy=1.0,
            delta=0.01
        )

    def test_on_conflict(self):
        self.evaluate_network(
            SRectangleConvNet(),
            make_rectangles_wrong_texture(),
            expected_accuracy=1.0
        )

class TestTextureConvNet(DataSetTest):
    def test_on_conflict_rectangles(self):
        self.evaluate_network(
            TextureConvNet(),
            make_rectangles_wrong_texture(),
            expected_accuracy=0.0
        )

    @unittest.expectedFailure # not quite working yet
    def test_on_conflict_lvt(self):
        self.evaluate_network(
            TextureConvNet(),
            make_LT_wrong_texture(),
            expected_accuracy=0.0
        )

class TestLTConvNet(DataSetTest):
    def test_on_correct_color(self):
        self.evaluate_network(
            LTConvNet(),
            make_LT_color(),
            delta=0.02
        )
    
    def test_on_correct_texture(self):
        self.evaluate_network(
            LTConvNet(),
            make_LT_texture(),
            delta=0.02
        )

    def test_on_wrong_color(self):
        self.evaluate_network(
            LTConvNet(),
            make_LT_wrong_color(),
            delta=0.02
        )

    def test_on_wrong_texture(self):
        self.evaluate_network(
            LTConvNet(),
            make_LT_wrong_texture(),
            delta=0.02
        )