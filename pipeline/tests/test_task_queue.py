import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
import numpy as np

from pipeline.render.task_queue import (
    build_queue, claim_task, complete_task, queue_report, reopen_task,
)


def _sample(root: Path, name: str = 'clip_a') -> Path:
    sample = root / 'real' / name
    (sample / 'labels' / 'part_id_cam01').mkdir(parents=True)
    (sample / 'sample.json').write_text(json.dumps({
        'schema_version': '1.0', 'mode': 'real_video_annotation',
        'views': {'cam01': {'video': 'rgb_cam01.mp4', 'part_id': 'labels/part_id_cam01'}},
    }))
    (sample / 'review.json').write_text(json.dumps({
        'status': 'needs_review', 'reviewed_frames': [],
    }))
    return sample


class TaskQueueBehaviorTest(unittest.TestCase):
    def test_tracking_completion_requires_full_frame_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'dataset'
            sample = _sample(root, 'clip_tracks')
            (sample / 'quality.json').write_text(json.dumps({'video': {'frame_count': 2}}))
            (sample / 'tracks_cam01.jsonl').write_text(
                '{"frame_index":0,"points":[]}\n{"frame_index":1,"points":[]}\n')
            data = json.loads((sample / 'sample.json').read_text())
            data['tracks'] = {'cam01': 'tracks_cam01.jsonl'}
            (sample / 'sample.json').write_text(json.dumps(data))
            queue_path = root / 'queue.json'
            queue = build_queue(root, queue_path)
            task = next(task for task in queue['tasks'] if task['stage'] == 'tracking')
            self.assertEqual(task['status'], 'done')

            (sample / 'tracks_cam01.jsonl').write_text('{"frame_index":0,"points":[]}\n')
            queue = build_queue(root, queue_path)
            task = next(task for task in queue['tasks'] if task['stage'] == 'tracking')
            self.assertEqual(task['status'], 'pending')
            self.assertEqual(task['reason_codes'], ['tracks_invalid_or_incomplete'])

    def test_decode_completion_requires_passed_quality_and_valid_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'dataset'
            sample = _sample(root, 'clip_decode')
            data = json.loads((sample / 'sample.json').read_text())
            data.update(reference_sample='reference', decoded_motion='decoded.npz',
                        motion_decode_quality='decode_quality.json')
            (sample / 'sample.json').write_text(json.dumps(data))
            np.savez(sample / 'decoded.npz', q=np.zeros((2, 1)), q_valid=np.ones((2, 1), dtype=bool))
            quality = {'status': 'motion_decode_passed', 'frame_count': 2, 'successful_frames': 2}
            (sample / 'decode_quality.json').write_text(json.dumps(quality))
            queue_path = root / 'queue.json'
            queue = build_queue(root, queue_path)
            task = next(task for task in queue['tasks'] if task['stage'] == 'decode')
            self.assertEqual(task['status'], 'done')

            quality['status'] = 'review'
            (sample / 'decode_quality.json').write_text(json.dumps(quality))
            queue = build_queue(root, queue_path)
            task = next(task for task in queue['tasks'] if task['stage'] == 'decode')
            self.assertEqual(task['status'], 'pending')
            self.assertEqual(task['reason_codes'], ['decoded_motion_missing_or_incomplete'])

    def test_queue_derives_tasks_and_cost_from_review_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'dataset'
            sample = _sample(root)
            queue_path = root / 'annotation_queue.json'
            queue = build_queue(root, queue_path, hourly_rate_usd=30)
            by_id = {task['task_id']: task for task in queue['tasks']}
            self.assertEqual(by_id['clip_a:cam01:review']['status'], 'pending')
            self.assertEqual(by_id['clip_a:cam01:tracking']['status'], 'pending')
            self.assertEqual(by_id['clip_a:cam01:decode']['status'], 'not_applicable')
            self.assertEqual(queue['summary']['task_count'], 3)

            self.assertEqual(claim_task(queue_path, 'clip_a:cam01:review', 'alice')['status'], 'claimed')
            with self.assertRaisesRegex(ValueError, 'source artifacts'):
                complete_task(queue_path, 'clip_a:cam01:review', duration_s=90)
            (sample / 'quality.json').write_text(json.dumps({'video': {'frame_count': 1}}))
            (sample / 'review.json').write_text(json.dumps({
                'status': 'human_review_complete', 'reviewed_frames': [0], 'reviewer': 'alice',
            }))
            (sample / 'labels' / 'part_id_cam01' / '000000.png').write_bytes(b'fixture')
            complete_task(queue_path, 'clip_a:cam01:review', duration_s=90, notes='checked')
            refreshed = build_queue(root, queue_path, hourly_rate_usd=30)
            review = next(task for task in refreshed['tasks'] if task['task_id'] == 'clip_a:cam01:review')
            self.assertEqual(review['status'], 'done')
            self.assertEqual(refreshed['summary']['total_duration_s'], 90)
            self.assertEqual(refreshed['summary']['estimated_cost_usd'], 0.75)

    def test_stale_done_review_task_reopens_when_source_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'dataset'
            _sample(root, 'clip_stale')
            queue_path = root / 'queue.json'
            build_queue(root, queue_path)
            queue = json.loads(queue_path.read_text())
            review = next(task for task in queue['tasks'] if task['task_id'] == 'clip_stale:cam01:review')
            review['status'] = 'done'
            queue_path.write_text(json.dumps(queue))

            refreshed = build_queue(root, queue_path)
            review = next(task for task in refreshed['tasks'] if task['task_id'] == 'clip_stale:cam01:review')
            self.assertEqual(review['status'], 'pending')
            self.assertEqual(review['reason_codes'], ['human_review_incomplete'])

    def test_failed_state_survives_queue_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'dataset'
            _sample(root, 'clip_b')
            queue_path = root / 'queue.json'
            build_queue(root, queue_path)
            claim_task(queue_path, 'clip_b:cam01:tracking', 'bob')
            complete_task(queue_path, 'clip_b:cam01:tracking', status='failed', duration_s=12)
            queue = build_queue(root, queue_path)
            tracking = next(task for task in queue['tasks'] if task['task_id'].endswith(':tracking'))
            self.assertEqual(tracking['status'], 'failed')
            self.assertEqual(tracking['duration_s'], 12)

    def test_reopen_preserves_attempt_history_and_accumulates_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'dataset'
            _sample(root, 'clip_retry')
            queue_path = root / 'queue.json'
            build_queue(root, queue_path, hourly_rate_usd=30)
            task_id = 'clip_retry:cam01:review'
            claim_task(queue_path, task_id, 'alice')
            complete_task(queue_path, task_id, status='failed', duration_s=12, notes='bad seed mask')

            reopened = reopen_task(queue_path, task_id, notes='seed mask replaced')
            self.assertEqual(reopened['status'], 'pending')
            self.assertEqual(reopened['attempt_history'][0]['worker'], 'alice')
            self.assertEqual(reopened['attempt_history'][0]['duration_s'], 12)
            self.assertEqual(reopened['notes'], 'seed mask replaced')
            self.assertEqual(queue_report(queue_path)['summary']['estimated_cost_usd'], 0.1)
            with self.assertRaisesRegex(ValueError, 'cannot be reopened'):
                reopen_task(queue_path, task_id)

            claim_task(queue_path, task_id, 'bob')
            complete_task(queue_path, task_id, status='failed', duration_s=8)
            self.assertEqual(queue_report(queue_path)['summary']['total_duration_s'], 20)

    def test_queue_includes_raw_assets_before_render(self):
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / 'raw'
            asset = data_root / 'asset_1'
            asset.mkdir(parents=True)
            (asset / 'assembly.json').write_text(json.dumps({
                'occurrences': {'root': {}}, 'joints': {'joint': {}},
            }))
            output_root = Path(directory) / 'dataset'
            queue = build_queue(output_root, data_root=data_root)
            tasks = {task['task_id']: task for task in queue['tasks']}
            self.assertEqual(tasks['asset_1:asset:qualification']['status'], 'done')
            self.assertEqual(tasks['asset_1:asset:annotate']['status'], 'pending')

    def test_queue_exposes_preview_review_and_reopens_stale_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'raw'
            asset = data_root / 'asset_1'
            asset.mkdir(parents=True)
            (asset / 'assembly.json').write_text(json.dumps({
                'occurrences': {'root': {}}, 'joints': {'joint': {}},
            }))
            output_root = root / 'output'
            preview = output_root / 'previews' / 'references' / 'asset_1'
            preview.mkdir(parents=True)
            (preview / 'contact_sheet.png').write_bytes(b'contact')
            (preview / 'motion.npz').write_bytes(b'motion')
            (preview / 'quality.json').write_text(json.dumps({
                'status': 'kinematic_preview', 'mujoco_fk_error': 0,
            }))
            from pipeline.render.auto_annotation import _preview_evidence
            report = [{'asset': 'asset_1', 'status': 'preview_ready'}]
            (output_root / 'previews' / 'preview_status.json').write_text(json.dumps(report))
            evidence = _preview_evidence(data_root, output_root, 'asset_1')
            report[0]['evidence'] = evidence
            (output_root / 'previews' / 'preview_status.json').write_text(json.dumps(report))
            queue = build_queue(output_root, data_root=data_root)
            review_id = 'asset_1:asset:preview_review'
            task = next(task for task in queue['tasks'] if task['task_id'] == review_id)
            self.assertEqual(task['status'], 'pending')

            from pipeline.render.auto_annotation import record_preview_decision
            record_preview_decision(data_root, output_root, 'asset_1', 'rejected',
                                    'alice', 'Motion not visually discernible')
            queue = build_queue(output_root, data_root=data_root)
            task = next(task for task in queue['tasks'] if task['task_id'] == review_id)
            self.assertEqual(task['status'], 'done')
            annotate = next(task for task in queue['tasks']
                            if task['task_id'] == 'asset_1:asset:annotate')
            self.assertEqual(annotate['status'], 'review')
            self.assertEqual(annotate['reason_codes'], ['preview_rejected'])

            (preview / 'contact_sheet.png').write_bytes(b'changed')
            queue = build_queue(output_root, data_root=data_root)
            task = next(task for task in queue['tasks'] if task['task_id'] == review_id)
            self.assertEqual(task['status'], 'pending')
            self.assertEqual(task['reason_codes'], ['preview_review_pending'])

    def test_publish_task_cannot_complete_before_sample_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'dataset'
            sample = root / 'samples' / 'fixture_multiview'
            sample.mkdir(parents=True)
            queue_path = root / 'queue.json'
            queue = build_queue(root, queue_path)
            publish_id = 'fixture_multiview:dataset:publish'
            publish = next(task for task in queue['tasks'] if task['task_id'] == publish_id)
            self.assertEqual(publish['status'], 'pending')
            claim_task(queue_path, publish_id, 'reviewer')
            with self.assertRaisesRegex(ValueError, 'source artifacts'):
                complete_task(queue_path, publish_id)

    def test_rejects_invalid_cost_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'dataset'
            _sample(root, 'clip_cost')
            queue_path = root / 'queue.json'
            with self.assertRaisesRegex(ValueError, 'hourly_rate_usd'):
                build_queue(root, queue_path, hourly_rate_usd=float('nan'))
            build_queue(root, queue_path)
            task_id = 'clip_cost:cam01:review'
            claim_task(queue_path, task_id, 'alice')
            with self.assertRaisesRegex(ValueError, 'duration_s'):
                complete_task(queue_path, task_id, status='failed', duration_s=-1)


class TaskQueueConcurrencyTest(unittest.TestCase):
    def test_concurrent_claims_assign_a_task_to_only_one_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'dataset'
            _sample(root)
            queue_path = root / 'queue.json'
            build_queue(root, queue_path)

            def claim(worker):
                try:
                    return claim_task(queue_path, 'clip_a:cam01:review', worker)['worker']
                except ValueError:
                    return None

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(claim, ('alice', 'bob')))
            self.assertEqual(len([value for value in results if value is not None]), 1)
            task = next(task for task in queue_report(queue_path)['tasks']
                        if task['task_id'] == 'clip_a:cam01:review')
            self.assertEqual(task['status'], 'claimed')
            self.assertIn(task['worker'], {'alice', 'bob'})
