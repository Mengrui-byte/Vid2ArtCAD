import unittest

import numpy as np

from pipeline.render.edit_pairs import edited_coordinates, joint_indices


class EditPairTest(unittest.TestCase):
    def test_selects_all_coordinates_of_a_multi_dof_joint(self):
        order = np.array(['fixed', 'plane:0', 'plane:1', 'plane:2', 'other'])
        self.assertEqual(joint_indices(order, 'plane'), [1, 2, 3])

    def test_scales_only_selected_coordinates(self):
        q = np.arange(12, dtype=float).reshape(4, 3)
        edited = edited_coordinates(q, [1], 2.0)
        self.assertTrue(np.array_equal(edited[:, 0], q[:, 0]))
        self.assertTrue(np.array_equal(edited[:, 1], q[:, 1] * 2))
        self.assertTrue(np.array_equal(edited[:, 2], q[:, 2]))


if __name__ == '__main__':
    unittest.main()
