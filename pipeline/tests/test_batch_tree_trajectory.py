import unittest
import warnings

import numpy as np
import trimesh

from pipeline.kinematics.solver import JointSpec, KinematicSolver
from pipeline.render.batch_tree_annotation import solve_commanded_trajectory
from pipeline.render.verified_lamp import camera_fit


class CommandedTrajectoryTest(unittest.TestCase):
    def test_loop_solves_uncommanded_tree_coordinate_as_follower(self):
        axis = np.array([[1.0, 0.0, 0.0]])
        tree = [
            JointSpec('drive', 'base', 'slider', 'prismatic', np.eye(4), np.eye(4),
                      axis, ('drive:0',)),
            JointSpec('follower', 'slider', 'end', 'prismatic', np.eye(4), np.eye(4),
                      axis, ('follower:0',)),
        ]
        loop = [JointSpec('closure', 'base', 'end', 'fixed', np.eye(4), np.eye(4),
                          np.zeros((0, 3)), ())]
        solver = KinematicSolver(['base', 'slider', 'end'], ['base'], tree, loop)

        q, transforms, q_all = solve_commanded_trajectory(
            solver, ['drive:0', 'follower:0'], [[0.0, 0.0], [0.2, 0.0], [0.0, 0.0]],
            [0], [0.0, 1.0, 2.0])

        self.assertAlmostEqual(q[1, 0], 0.2, places=8)
        self.assertAlmostEqual(q[1, 1], -0.2, places=8)
        self.assertAlmostEqual(transforms[1, 2, 0, 3], 0.0, places=8)
        self.assertTrue(np.allclose(q_all[:, :2], q))
        for state in q_all:
            self.assertLess(np.linalg.norm(solver.closure_residual(state)), 1e-7)

    def test_camera_fit_projects_full_multiframe_mesh_envelope_without_warnings(self):
        meshes = {index: trimesh.creation.box(extents=[0.1, 0.2, 0.3]) for index in range(12)}
        frames = np.repeat(np.eye(4)[None, None], 360 * 12, axis=0).reshape(360, 12, 4, 4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', RuntimeWarning)
            intrinsics, extrinsic, _, _, framing = camera_fit(meshes, frames, 1920, 1080)

        self.assertTrue(np.isfinite(intrinsics).all())
        self.assertTrue(np.isfinite(extrinsic).all())
        self.assertTrue(np.isfinite(framing['minimum_pixel']).all())
        self.assertEqual(caught, [])


if __name__ == '__main__':
    unittest.main()
