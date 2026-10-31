import unittest

import numpy as np

from pipeline.validation.checks import gate, reference_pose_recovery, rigid_transform_error


def test_rigid_error_identity():
    assert rigid_transform_error(np.eye(4)) < 1e-12


def test_reference_pose_gate():
    d = 1.0
    t_over_d, r_err = reference_pose_recovery(np.eye(4), np.eye(4), d)
    assert t_over_d == 0 and r_err == 0
    report = gate("clip_0", {"projection_median_px": 0.5}, [])
    assert report.status == "accepted"

class ValidationTest(unittest.TestCase):
    def test_rigid(self):
        test_rigid_error_identity()
    def test_gate(self):
        test_reference_pose_gate()
