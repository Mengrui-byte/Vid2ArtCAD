import unittest

import numpy as np

from pipeline.kinematics.source import build_source_model


def joint(name, parent, child, kind, origin=(0, 0, 0), primary=(0, 0, 1), secondary=None, tertiary=None):
    geometry = {
        'origin': dict(zip(('x', 'y', 'z'), origin)),
        'primary_axis_vector': dict(zip(('x', 'y', 'z'), primary)),
    }
    if secondary is not None:
        geometry['secondary_axis_vector'] = dict(zip(('x', 'y', 'z'), secondary))
    if tertiary is not None:
        geometry['tertiary_axis_vector'] = dict(zip(('x', 'y', 'z'), tertiary))
    return {
        'name': name,
        'occurrence_one': parent,
        'occurrence_two': child,
        'joint_motion': {'joint_type': kind},
        'geometry_or_origin_one': geometry,
    }


class SourceModelTest(unittest.TestCase):
    def test_cylindrical_joint_has_stable_two_dof_model(self):
        transforms = {
            'base': np.eye(4),
            'shaft': np.eye(4),
        }
        raw = {'j': joint('shaft_joint', 'base', 'shaft', 'CylindricalJointType')}
        model = build_source_model(('base', 'shaft'), transforms, raw, 'base')
        self.assertEqual(model.solver.dof_order, ('shaft_joint:0', 'shaft_joint:1'))
        pose = model.solver.forward([0.0, 0.25])['shaft']
        self.assertAlmostEqual(pose[2, 3], 0.25)

    def test_cycle_is_retained_as_loop_constraint(self):
        transforms = {name: np.eye(4) for name in ('a', 'b', 'c')}
        raw = {
            'ab': joint('ab', 'a', 'b', 'RigidJointType'),
            'bc': joint('bc', 'b', 'c', 'RigidJointType'),
            'ca': joint('ca', 'c', 'a', 'RigidJointType'),
        }
        model = build_source_model(('a', 'b', 'c'), transforms, raw, 'a')
        self.assertEqual(len(model.tree), 2)
        self.assertEqual([edge.id for edge in model.loops], ['ca'])
        result = model.solver.solve()
        self.assertTrue(result.success)
        self.assertLess(np.linalg.norm(result.residual), 1e-12)

    def test_planar_axes_are_in_plane_then_normal(self):
        transforms = {'base': np.eye(4), 'plate': np.eye(4)}
        raw = {'p': joint('plane', 'base', 'plate', 'PlanarJointType',
                          primary=(0, 0, 1), secondary=(1, 0, 0), tertiary=(0, 1, 0))}
        model = build_source_model(('base', 'plate'), transforms, raw, 'base')
        axes = model.solver.tree_joints[0].axes_in_joint
        self.assertTrue(np.allclose(axes, np.eye(3)))

    def test_disconnected_components_keep_source_world_poses_as_separate_roots(self):
        transforms = {name: np.eye(4) for name in ('base', 'arm', 'loose_part')}
        transforms['arm'][0, 3] = 1.0
        transforms['loose_part'][:3, 3] = [3.0, -2.0, 0.5]
        transforms['loose_part'][:3, :3] = np.array([
            [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
        ])
        raw = {'j': joint('hinge', 'base', 'arm', 'RevoluteJointType')}

        model = build_source_model(('base', 'arm', 'loose_part'), transforms, raw, 'base')
        actual = model.solver.forward(np.zeros(model.solver.dof_count))

        self.assertEqual(model.roots, ('base', 'loose_part'))
        self.assertEqual(model.solver.roots, model.roots)
        for link, expected in transforms.items():
            self.assertTrue(np.allclose(actual[link], expected))
        moved = model.solver.forward(np.array([0.25]))
        self.assertTrue(np.allclose(moved['loose_part'], transforms['loose_part']))
        self.assertFalse(np.allclose(moved['arm'], transforms['arm']))


if __name__ == '__main__':
    unittest.main()
