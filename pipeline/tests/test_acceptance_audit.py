import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from pipeline.render.acceptance_audit import _runtime_replay, audit_acceptance


class AcceptanceAuditTest(unittest.TestCase):
    def test_runtime_replay_checks_q_signs_fk_and_camera_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory)
            (sample / 'scene.xml').write_text('''<mujoco><worldbody>
              <camera name="cam01" pos="0 0 3" fovy="38"/>
              <body name="link_00"><joint name="Rev1" type="hinge" axis="0 0 1"/>
                <geom type="sphere" size=".1"/></body>
            </worldbody></mujoco>''')
            model = mujoco.MjModel.from_xml_path(str(sample / 'scene.xml'))
            sim = mujoco.MjData(model)
            q = np.array([[-0.2], [0.0], [0.2]])
            poses = np.zeros((3, 1, 4, 4))
            for index, value in enumerate(q):
                sim.qpos[0] = value[0]
                mujoco.mj_forward(model, sim)
                poses[index, 0] = np.eye(4)
                poses[index, 0, :3, :3] = sim.xmat[1].reshape(3, 3)
                poses[index, 0, :3, 3] = sim.xpos[1]
            np.savez(sample / 'motion.npz', q=q, T_world_link=poses,
                     joint_order=np.array(['Rev1']), link_order=np.array(['link_00']))
            camera = np.eye(4)
            camera[:3, :3] = sim.cam_xmat[0].reshape(3, 3) @ np.diag([1, -1, -1])
            camera[:3, 3] = sim.cam_xpos[0]
            width, height = 100, 80
            focal = (height / 2) / np.tan(np.deg2rad(38 / 2))
            calibration = {
                'resolution_wh': [width, height], 'fovy_degrees': 38,
                'K': [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]],
                'T_world_camera': camera.tolist(),
            }
            (sample / 'camera_cam01.json').write_text(json.dumps(calibration))
            (sample / 'views.json').write_text(json.dumps({'cameras': {'cam01': {}}}))
            (sample / 'sample.json').write_text(json.dumps({
                'motion': 'motion.npz', 'scene': 'scene.xml', 'views': 'views.json',
            }))

            report = _runtime_replay(sample)
            self.assertEqual(report['status'], 'passed', report)
            self.assertEqual(report['zero_state_frame_count'], 1)
            self.assertTrue(report['positive_negative_perturbations'])
            self.assertLess(report['camera_extrinsic_error'], 1e-7)
            self.assertLess(report['camera_projection_error'], 1e-7)

    def test_acceptance_audit_reports_evidence_without_inflating_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data_root = tmp_path / 'raw'
            asset = data_root / 'asset_a'
            asset.mkdir(parents=True)
            (asset / 'assembly.json').write_text(json.dumps({'occurrences': {}, 'joints': {}}))
            output_root = tmp_path / 'output'
            report = audit_acceptance(data_root, output_root)

            self.assertEqual(report['summary']['raw_candidate_count'], 1)
            self.assertEqual(report['summary']['accepted_sample_count'], 0)
            self.assertEqual(report['overall_status'], 'pending')
            self.assertTrue(all(gate['status'] == 'pending' for gate in report['gates']))

    def test_four_view_gate_uses_four_view_target_not_candidate_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'raw'
            data_root.mkdir()
            output_root = root / 'output'
            sample_root = output_root / 'samples'
            samples = []
            for index in range(4):
                sample = sample_root / f'asset_{index}_multiview'
                sample.mkdir(parents=True)
                (sample / 'sample.json').write_text(json.dumps({
                    'sample_id': sample.name,
                    'views': {f'cam0{view}': {} for view in range(1, 5)},
                }))
                samples.append(sample)

            with patch('pipeline.render.acceptance_audit.validate_sample',
                       return_value={'status': 'accepted', 'reason_codes': []}):
                report = audit_acceptance(data_root, output_root)

            gate = next(item for item in report['gates'] if item['gate'] == 'P1-four-videos')
            self.assertEqual(gate['target'], 4)
            self.assertEqual(gate['actual'], 4)
            self.assertEqual(gate['status'], 'passed')

    def test_structural_acceptance_and_independent_replay_are_counted_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'raw'
            data_root.mkdir()
            sample = root / 'output' / 'samples' / 'fixture_multiview'
            sample.mkdir(parents=True)
            (sample / 'sample.json').write_text(json.dumps({
                'sample_id': sample.name,
                'views': {f'cam0{view}': {} for view in range(1, 5)},
            }))

            with patch('pipeline.render.acceptance_audit.validate_sample',
                       return_value={'status': 'accepted', 'reason_codes': []}), \
                    patch('pipeline.render.acceptance_audit._runtime_replay',
                          return_value={'status': 'failed', 'reason_codes': [
                              'positive_negative_perturbations_missing']}):
                report = audit_acceptance(data_root, root / 'output')

            summary = report['summary']
            self.assertEqual(summary['accepted_sample_count'], 1)
            self.assertEqual(summary['four_view_accepted_count'], 1)
            self.assertEqual(summary['runtime_replay_passed_sample_count'], 0)
            self.assertEqual(summary['runtime_replay_failed_sample_count'], 1)
            self.assertEqual(summary['runtime_replay_passed_family_count'], 0)
            p0 = next(item for item in report['gates'] if item['gate'] == 'P0')
            self.assertEqual(p0['actual'], 0)
            self.assertEqual(p0['status'], 'pending')

    def test_p4_requires_pseudo_label_evidence_and_human_review_for_every_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'raw'
            data_root.mkdir()
            real_root = root / 'output' / 'real'
            pseudo = real_root / 'pseudo_multiview'
            manual = real_root / 'manual_multiview'
            propagated = real_root / 'propagated_multiview'
            cases = ((pseudo, 'pseudo', None),
                     (manual, 'human_reference', None),
                     (propagated, 'human_reference', 'cam02'))
            for sample, origin, propagated_view in cases:
                sample.mkdir(parents=True)
                views = {'cam01': {'review': 'review.json'},
                         'cam02': {'review': 'review_cam02.json'}}
                (sample / 'sample.json').write_text(json.dumps({
                    'mode': 'real_video_annotation', 'sample_id': sample.name,
                    'views': views,
                }))
                (sample / 'quality.json').write_text(json.dumps({
                    'views': {view: {'labels': {'origin': origin,
                                                'source': 'human_reference'}}
                              for view in views},
                }))
                for view, spec in views.items():
                    review = {
                        'status': 'human_review_complete', 'reviewed_frames': [0],
                        'reviewer': 'reviewer',
                    }
                    if propagated_view:
                        review['pseudo_frames'] = {'1': {
                            'source': 'optical_flow_pseudo', 'seed_frame': 0,
                            'review_required': True,
                        }}
                    (sample / spec['review']).write_text(json.dumps(review))

            with patch('pipeline.render.acceptance_audit.validate_sample',
                       return_value={'status': 'accepted', 'reason_codes': []}):
                report = audit_acceptance(data_root, root / 'output')

            by_id = {item['sample_id']: item for item in report['real_samples']}
            self.assertTrue(by_id['pseudo_multiview']['has_pseudo_labels'])
            self.assertEqual(by_id['pseudo_multiview']['pseudo_label_views'], 2)
            self.assertFalse(by_id['manual_multiview']['has_pseudo_labels'])
            self.assertTrue(by_id['propagated_multiview']['has_pseudo_labels'])
            p4 = next(item for item in report['gates'] if item['gate'] == 'P4-real-video-closure')
            self.assertEqual(p4['actual'], 2)

    def test_generation_tier_names_do_not_count_as_native_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'raw'
            data_root.mkdir()
            output_root = root / 'output'
            sample_root = output_root / 'samples'
            for tier in ('G0', 'G1', 'G2'):
                sample = sample_root / f'fixture_{tier}_multiview'
                sample.mkdir(parents=True)
                (sample / 'sample.json').write_text(json.dumps({
                    'sample_id': sample.name, 'generation_tier': tier,
                    'template_id': f'claimed-{tier}',
                    'template_family': f'also-claimed-{tier}',
                }))

            with patch('pipeline.render.acceptance_audit.validate_sample',
                       return_value={'status': 'accepted', 'reason_codes': []}):
                report = audit_acceptance(data_root, output_root)

            gate = next(item for item in report['gates'] if item['gate'] == 'P3-native-templates')
            self.assertEqual(gate['actual'], 0)
            self.assertEqual(gate['status'], 'pending')
