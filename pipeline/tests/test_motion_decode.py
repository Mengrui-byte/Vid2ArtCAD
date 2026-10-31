import json
import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from pipeline.render.motion_decode import decode_motion
from pipeline.render.mujoco_runtime import set_qpos
from pipeline.render.stereo_annotation import project


class MotionDecodeTest(unittest.TestCase):
    def test_decodes_single_revolute_joint_from_part_centroids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'reference'
            reference.mkdir()
            (reference / 'scene_multiview.xml').write_text('''<mujoco>
              <worldbody>
                <body name="link_00" pos="0 0 4">
                  <geom type="box" size="0.1 0.1 0.1"/>
                  <body name="link_01" pos="0 0 1">
                    <joint name="Rev1" type="hinge" pos="0 0 -1" axis="0 1 0"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                  </body>
                </body>
              </worldbody>
            </mujoco>''')
            camera = {
                'resolution_wh': [64, 64], 'fps': 10,
                'K': [[60, 0, 32], [0, 60, 32], [0, 0, 1]],
                'T_world_camera': np.eye(4).tolist(),
            }
            camera_path = root / 'camera.json'
            camera_path.write_text(json.dumps(camera))
            model = mujoco.MjModel.from_xml_path(str(reference / 'scene_multiview.xml'))
            data = mujoco.MjData(model)
            masks = root / 'masks'
            masks.mkdir()
            qs = np.array([-0.2, 0.0, 0.2])
            for frame, value in enumerate(qs):
                set_qpos(model, data, np.array([value]), np.array(['Rev1']))
                mujoco.mj_forward(model, data)
                points = np.asarray(data.xpos[[1]])
                uv, depth = project(points, np.eye(4), np.asarray(camera['K']))
                self.assertTrue(depth[0] > 0)
                x, y = np.rint(uv[0]).astype(int)
                mask = np.zeros((64, 64), dtype=np.uint16)
                mask[max(0, y - 1):min(64, y + 2), max(0, x - 1):min(64, x + 2)] = 2
                imageio.imwrite(masks / f'{frame:06d}.png', mask)
            np.savez(reference / 'motion.npz', q=qs[:, None],
                     joint_order=np.array(['Rev1']), link_order=np.array(['link_00', 'link_01']))
            output = root / 'decoded'
            report = decode_motion(reference, [masks], [camera_path], output,
                                   residual_threshold_px=2.0)
            decoded = np.load(output / 'decoded_motion.npz', allow_pickle=False)
            self.assertEqual(decoded['q'].shape, (3, 1))
            self.assertEqual(decoded['q_valid'].shape, (3, 1))
            self.assertGreaterEqual(report['success_fraction'], 2 / 3)
            self.assertTrue((output / 'motion_decode_quality.json').is_file())
            multi_output = root / 'decoded_multi'
            multi_report = decode_motion(reference, [masks, masks], [camera_path, camera_path],
                                         multi_output, residual_threshold_px=2.0)
            self.assertEqual(multi_report['view_count'], 2)
            self.assertIn('temporal synchronization assumed, not verified',
                          multi_report['frame_alignment'])
            partial_output = root / 'decoded_partial'
            partial_report = decode_motion(
                reference, [masks], [camera_path], partial_output,
                residual_threshold_px=2.0, active_joints=[])
            partial = np.load(partial_output / 'decoded_motion.npz', allow_pickle=False)
            self.assertEqual(partial_report['status'], 'review')
            self.assertFalse(partial['q_valid'].any())
            self.assertFalse(partial_report['full_dof_coverage'])

            bad_camera = dict(camera)
            bad_camera['T_world_camera'] = (np.diag([2.0, 1.0, 1.0, 1.0])).tolist()
            bad_camera_path = root / 'bad_camera.json'
            bad_camera_path.write_text(json.dumps(bad_camera))
            with self.assertRaisesRegex(ValueError, 'rigid transform'):
                decode_motion(reference, [masks], [bad_camera_path], root / 'bad_calibration')

            offset_camera = dict(camera, fps=12)
            offset_camera_path = root / 'different_fps.json'
            offset_camera_path.write_text(json.dumps(offset_camera))
            with self.assertRaisesRegex(ValueError, 'same fps'):
                decode_motion(reference, [masks, masks], [camera_path, offset_camera_path],
                              root / 'unsynchronized')

    def test_publishes_motion_tasks_and_checksums_inside_real_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / 'real'
            real.mkdir()
            (real / 'sample.json').write_text(json.dumps({
                'mode': 'real_video_annotation', 'sample_id': 'real-1',
            }))
            (real / 'quality.json').write_text(json.dumps({'status': 'needs_review'}))
            (real / 'review.json').write_text(json.dumps({
                'status': 'human_review_complete', 'reviewer': 'tester',
            }))
            # Reuse the first test's minimal reference and observation data.
            reference = root / 'reference'
            reference.mkdir()
            (reference / 'scene_multiview.xml').write_text('''<mujoco><worldbody>
              <body name="link_00" pos="0 0 4"><geom type="box" size="0.1 0.1 0.1"/>
              </body></worldbody></mujoco>''')
            camera = {'resolution_wh': [32, 32], 'fps': 5,
                      'K': [[30, 0, 16], [0, 30, 16], [0, 0, 1]],
                      'T_world_camera': np.eye(4).tolist()}
            camera_path = root / 'camera.json'; camera_path.write_text(json.dumps(camera))
            masks = root / 'masks'; masks.mkdir()
            mask = np.zeros((32, 32), dtype=np.uint16); mask[15:18, 15:18] = 1
            imageio.imwrite(masks / '000000.png', mask)
            np.savez(reference / 'motion.npz', q=np.zeros((1, 0)),
                     joint_order=np.array([], dtype='<U1'), link_order=np.array(['link_00']))
            output = real / 'decoded_motion'
            decode_motion(reference, [masks], [camera_path], output, residual_threshold_px=2.0)
            sample = json.loads((real / 'sample.json').read_text())
            quality = json.loads((real / 'quality.json').read_text())
            checksums = json.loads((real / 'checksums.json').read_text())
            self.assertEqual(sample['available_tasks'], ['rgb', 'part_id', 'human_review', 'motion', 'kinematics'])
            self.assertEqual(quality['motion_decode']['status'], 'motion_decode_passed')
            self.assertIn('decoded_motion/decoded_motion.npz', checksums)


if __name__ == '__main__':
    unittest.main()
