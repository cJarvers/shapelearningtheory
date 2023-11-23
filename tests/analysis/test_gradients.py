import unittest
import torch
# local imports
from shapelearningtheory.analysis.gradients import *

class TestJacobian(unittest.TestCase):
    def test_identity(self):
        net = torch.nn.Identity()
        batchsize = 4
        height = 8
        width = 6
        channels = 3
        x = torch.randn(batchsize, channels, height, width)
        jac = compute_jacobian(net, x)
        # jacobian of identity function is identity matrix
        identity = torch.eye(height*width*channels).unsqueeze(0).repeat([batchsize, 1, 1, 1])
        self.assertTrue((jac == identity).all())