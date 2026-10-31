import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from pipeline.render.motion_decode import decode_motion
from pipeline.render.mujoco_runtime import set_qpos
from pipeline.render.package_dataset import package_dataset
from pipeline.render.review_server import ReviewApplication
from pipeline.render.stereo_annotation import project
from pipeline.render.task_queue import build_queue
from pipeline.render.validate_sample import validate_sample
from pipeline.render.video_annotation import import_multiview


class AnnotationEndToEndTest(unittest.TestCase):
    def test_video_import_review_decode_and_package_closed_loop(self):
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
            sim = mujoco.MjData(model)
            video_path = root / 'input.mp4'
            writer = imageio.get_writer(video_path, fps=10, codec='libx264', macro_block_size=1)
            masks = []
            q_values = (-0.2, 0.0, 0.2)
            try:
                for value in q_values:
                    set_qpos(model, sim, np.array([value]), np.array(['Rev1']))
                    mujoco.mj_forward(model, sim)
                    uv, depth = project(sim.xpos[[1]], np.eye(4), np.asarray(camera['K']))
                    self.assertGreater(depth[0], 0)
                    x, y = np.rint(uv[0]).astype(int)
                    self.assertTrue(3 <= x < 61 and 3 <= y < 61)
                    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
                    rgb[y - 2:y + 3, x - 2:x + 3] = 255
                    writer.append_data(rgb)
                    mask = np.zeros((64, 64), dtype='<u2')
                    mask[y - 2:y + 3, x - 2:x + 3] = 2
                    masks.append(mask)
            finally:
                writer.close()
            np.savez(reference / 'motion.npz', q=np.asarray(q_values)[:, None],
                     joint_order=np.array(['Rev1']),
                     link_order=np.array(['link_00', 'link_01']))

            output_root = root / 'dataset'
            sample = output_root / 'real' / 'known-motion'
            imported = import_multiview(
                {'cam01': video_path, 'cam02': video_path}, sample,
                cameras={'cam01': camera_path, 'cam02': camera_path},
                design_family_id='fixture-revolute',
            )
            self.assertEqual(imported['status'], 'needs_review')

            for view in ('cam01', 'cam02'):
                review = ReviewApplication(sample, f'rgb_{view}.mp4')
                review.save_part(2, 'moving link')
                for frame_index, mask in enumerate(masks):
                    review.save_mask(frame_index, 'reviewer', mask.tobytes())
                self.assertEqual(review.review()['status'], 'human_review_complete')

            decoded = decode_motion(
                reference,
                [sample / 'labels' / f'part_id_{view}' for view in ('cam01', 'cam02')],
                [sample / f'camera_{view}.json' for view in ('cam01', 'cam02')],
                sample / 'decoded_motion',
                residual_threshold_px=2.0,
            )
            self.assertEqual(decoded['status'], 'motion_decode_passed', decoded)
            self.assertEqual(decoded['view_count'], 2)
            validation = validate_sample(sample)
            self.assertEqual(validation['status'], 'accepted', validation)

            manifest = package_dataset(output_root)
            self.assertEqual(manifest['sample_count'], 1, manifest)
            entry = manifest['accepted'][0]
            self.assertEqual(entry['design_family_id'], 'fixture-revolute')
            self.assertIn('human_review', entry['tasks'])
            self.assertIn('motion', entry['tasks'])
            self.assertIn('kinematics', entry['tasks'])

    def test_partial_motion_keeps_accepted_segmentation_and_review_decode_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / 'dataset' / 'real' / 'partial-motion'
            video = root / 'clip.mp4'
            writer = imageio.get_writer(video, fps=2, codec='libx264', macro_block_size=1)
            for value in (0, 80):
                writer.append_data(np.full((8, 8, 3), value, dtype=np.uint8))
            writer.close()
            import_multiview({'cam01': video}, sample)
            app = ReviewApplication(sample)
            app.save_part(1, 'part')
            mask = np.ones((8, 8), dtype='<u2')
            for frame in range(2):
                app.save_mask(frame, 'reviewer', mask.tobytes())
            (sample / 'partial.npz').parent.mkdir(exist_ok=True)
            np.savez(sample / 'partial.npz', q=np.zeros((2, 1)),
                     q_valid=np.zeros((2, 1), dtype=bool))
            sample_data = json.loads((sample / 'sample.json').read_text())
            sample_data.update(reference_sample='reference', decoded_motion='partial.npz',
                               motion_decode_quality='partial_quality.json')
            (sample / 'sample.json').write_text(json.dumps(sample_data))
            (sample / 'partial_quality.json').write_text(json.dumps({
                'status': 'motion_decode_partial', 'frame_count': 2,
                'successful_frames': 2, 'full_dof_coverage': False,
            }))
            checksums = {
                str(path.relative_to(sample)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sample.rglob('*') if path.is_file() and path.name != 'checksums.json'
            }
            (sample / 'checksums.json').write_text(json.dumps(checksums))

            validation = validate_sample(sample)
            self.assertEqual(validation['status'], 'accepted', validation)
            queue = build_queue(root / 'dataset')
            decode_task = next(task for task in queue['tasks'] if task['stage'] == 'decode')
            self.assertEqual(decode_task['status'], 'review')

            manifest = package_dataset(root / 'dataset')
            self.assertEqual(manifest['sample_count'], 1, manifest)
            tasks = manifest['accepted'][0]['tasks']
            self.assertIn('human_review', tasks)
            self.assertNotIn('kinematics', tasks)


if __name__ == '__main__':
    unittest.main()
