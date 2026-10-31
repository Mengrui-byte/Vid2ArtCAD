"""Validate URDF rotation labels and SO(3) prediction/loss conversion."""

import unittest

import numpy as np
import torch

from Model.representation import _rpy_matrix
from Model.rotations import geodesic_angle, matrix_to_rpy, rotation_6d_to_matrix


class RotationTest(unittest.TestCase):
    def test_rpy_roundtrip(self):
        original = [0.4, -0.2, 0.6]
        matrix = _rpy_matrix(original)
        reconstructed = _rpy_matrix(matrix_to_rpy(matrix))
        np.testing.assert_allclose(reconstructed, matrix, atol=1e-6)

    def test_six_dimensional_rotation_and_geodesic_gradient(self):
        target = torch.tensor(_rpy_matrix([0.4, -0.2, 0.6])).float()
        guess = torch.tensor([[1., 0., 0., 0., 1., 0.]], requires_grad=True)
        loss = geodesic_angle(rotation_6d_to_matrix(guess), target)
        self.assertGreater(loss.item(), 0.5)
        loss.backward()
        self.assertTrue(torch.isfinite(guess.grad).all())
        self.assertGreater(guess.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
