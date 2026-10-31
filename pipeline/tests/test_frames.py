import unittest

import numpy as np

from pipeline.kinematics.frames import KinematicGraph, fk, fk_q_zero, joint_transform, joint_transform_dofs


def test_q_zero_restores_reference_pose():
    A = np.eye(4)
    A[:3, 3] = [0.1, 0, 0.2]
    B = np.eye(4)
    graph = KinematicGraph(("base", "lid"), "base", {"lid": "base"}, {"lid": "hinge"})
    links = {}
    joints = {
        "hinge": {
            "type": "revolute",
            "axis_in_joint": [0, 0, 1],
            "T_parent_joint": A,
            "T_joint_child_ref": B,
        }
    }
    world, valid = fk(graph, links, joints, {"hinge": 0.0})
    assert np.allclose(world["lid"], fk_q_zero(A, B))
    assert valid["lid"]


def test_prismatic_direction():
    T = joint_transform("prismatic", np.array([0, 0, 1]), 0.5)
    assert np.allclose(T[:3, 3], [0, 0, 0.5])


def test_cylindrical_has_rotation_and_translation_dofs():
    T = joint_transform_dofs("cylindrical", np.array([0.0, 0.5]), np.array([[0, 0, 1.0]]))
    assert np.allclose(T[:3, 3], [0, 0, 0.5])


def test_ball_uses_rotation_vector():
    T = joint_transform_dofs("ball", np.array([0.0, 0.0, np.pi / 2]))
    assert np.allclose(T[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)

class FramesTest(unittest.TestCase):
    def test_q_zero(self):
        test_q_zero_restores_reference_pose()
    def test_prismatic(self):
        test_prismatic_direction()
    def test_cylindrical(self):
        test_cylindrical_has_rotation_and_translation_dofs()
    def test_ball(self):
        test_ball_uses_rotation_vector()
