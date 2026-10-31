import unittest

import numpy as np

from pipeline.kinematics.solver import JointSpec, KinematicSolver


def fixed_joint(name, parent, child):
    return JointSpec(name, parent, child, "fixed", np.eye(4), np.eye(4), np.zeros((0, 3)), ())


class SolverTest(unittest.TestCase):
    def test_multi_dof_tree_forward(self):
        joint = JointSpec(
            "cyl", "base", "shaft", "cylindrical", np.eye(4), np.eye(4),
            np.array([[0.0, 0.0, 1.0]]), ("cyl:0", "cyl:1"),
        )
        solver = KinematicSolver(["base", "shaft"], ["base"], [joint])
        pose = solver.forward([0.0, 0.25])["shaft"]
        self.assertTrue(np.allclose(pose[:3, 3], [0.0, 0.0, 0.25]))

    def test_loop_solver_preserves_zero_configuration(self):
        tree = [fixed_joint("ab", "a", "b"), fixed_joint("bc", "b", "c")]
        loop = [fixed_joint("ca", "c", "a")]
        solver = KinematicSolver(["a", "b", "c"], ["a"], tree, loop)
        result = solver.solve()
        self.assertTrue(result.success)
        self.assertLess(np.linalg.norm(result.residual), 1e-12)

    def test_prescribed_dof_is_respected(self):
        joint = JointSpec(
            "slide", "a", "b", "prismatic", np.eye(4), np.eye(4),
            np.array([[1.0, 0.0, 0.0]]), ("slide:0",),
        )
        solver = KinematicSolver(["a", "b"], ["a"], [joint])
        result = solver.solve(prescribed={"slide:0": 0.4})
        self.assertAlmostEqual(result.q[0], 0.4)
        self.assertAlmostEqual(result.transforms["b"][0, 3], 0.4)
