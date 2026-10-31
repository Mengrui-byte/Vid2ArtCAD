import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from pipeline.render.mujoco_runtime import set_qpos


class MuJoCoRuntimeTest(unittest.TestCase):
    def test_ball_runtime_maps_rotation_vector_to_quaternion(self):
        xml = '<mujoco><worldbody><body name="link"><joint name="ball_sim" type="ball"/><geom type="sphere" size=".1"/></body></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        q = np.array([0.0, 0.0, np.pi / 2])
        set_qpos(model, data, q, np.array(['ball:0', 'ball:1', 'ball:2']), {
            'joints': [{'q_indices': [0, 1, 2], 'sim_name': 'ball_sim', 'kind': 'ball'}]
        })
        mujoco.mj_forward(model, data)
        self.assertTrue(np.allclose(data.xmat[1].reshape(3, 3), Rotation.from_rotvec(q).as_matrix()))

    def test_ball_runtime_honors_reversed_source_motion(self):
        xml = '<mujoco><worldbody><body name="link"><joint name="ball_sim" type="ball"/><geom type="sphere" size=".1"/></body></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        q = np.array([0.0, 0.0, np.pi / 2])
        set_qpos(model, data, q, np.array(['ball:0', 'ball:1', 'ball:2']), {
            'joints': [{'q_indices': [0, 1, 2], 'sim_name': 'ball_sim', 'kind': 'ball', 'inverse_motion': True}]
        })
        mujoco.mj_forward(model, data)
        self.assertTrue(np.allclose(data.xmat[1].reshape(3, 3), Rotation.from_rotvec(-q).as_matrix()))

    def test_scalar_mapping_rejects_missing_or_duplicate_coordinates_before_write(self):
        xml = '''<mujoco><worldbody><body name="a"><joint name="j0" type="hinge"/>
          <geom type="sphere" size=".1"/><body name="b"><joint name="j1" type="slide"/>
          <geom type="sphere" size=".1"/></body></body></worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        data.qpos[:] = [0.25, -0.5]
        malformed = [
            {'joints': [{'q_indices': [0], 'sim_name': 'j0', 'kind': 'scalar'}]},
            {'joints': [
                {'q_indices': [0], 'sim_name': 'j0', 'kind': 'scalar'},
                {'q_indices': [0], 'sim_name': 'j1', 'kind': 'scalar'},
            ]},
        ]
        for runtime in malformed:
            with self.subTest(runtime=runtime):
                with self.assertRaisesRegex(ValueError, 'cover each canonical'):
                    set_qpos(model, data, np.array([0.1, 0.2]), np.array(['j0', 'j1']), runtime)
                self.assertTrue(np.allclose(data.qpos, [0.25, -0.5]))

    def test_runtime_rejects_joint_order_and_type_mismatches(self):
        xml = '<mujoco><worldbody><body name="link"><joint name="hinge" type="hinge"/><geom type="sphere" size=".1"/></body></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        with self.assertRaisesRegex(ValueError, 'joint_order'):
            set_qpos(model, data, np.array([0.2]), np.array(['hinge']), {
                'joint_order': ['other'],
                'joints': [{'q_indices': [0], 'sim_name': 'hinge', 'kind': 'scalar'}],
            })
        with self.assertRaisesRegex(ValueError, 'ball runtime'):
            set_qpos(model, data, np.array([0.1, 0.2, 0.3]),
                     np.array(['x', 'y', 'z']), {
                         'joints': [{'q_indices': [0, 1, 2], 'sim_name': 'hinge', 'kind': 'ball'}],
                     })

    def test_runtime_rejects_non_finite_coordinates(self):
        xml = '<mujoco><worldbody><body name="link"><joint name="hinge" type="hinge"/><geom type="sphere" size=".1"/></body></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        with self.assertRaisesRegex(ValueError, 'finite'):
            set_qpos(model, data, np.array([np.nan]), np.array(['hinge']))


if __name__ == '__main__':
    unittest.main()
