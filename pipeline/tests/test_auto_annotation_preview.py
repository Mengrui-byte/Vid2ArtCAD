import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.render.auto_annotation import (
    _approved_preview, _preview_evidence, _template_scene, atomic_export_reference,
    build_config, record_preview_decision, reference_destination, run_dataset,
)


class AutoAnnotationPreviewTest(unittest.TestCase):
    def test_build_config_preserves_non_default_asset_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / 'custom_source' / 'demo'
            asset.mkdir(parents=True)
            (asset / 'assembly.json').write_text(json.dumps({
                'occurrences': {'root': {'name': 'base', 'is_grounded': True}},
                'joints': {'joint': {
                    'name': 'hinge',
                    'joint_motion': {'joint_type': 'RevoluteJointType'},
                }},
            }))

            config = build_config(asset, .5, .01, 2)

            self.assertEqual(config['asset_root'], str(asset.parent.resolve()))

    def test_template_scene_falls_back_to_existing_root_level_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / 'legacy_sample' / 'scene.xml'
            legacy.parent.mkdir()
            legacy.write_text('<mujoco/>')
            self.assertEqual(_template_scene(root), legacy)

            current = root / 'samples' / 'current_multiview' / 'scene.xml'
            current.parent.mkdir(parents=True)
            current.write_text('<mujoco/>')
            self.assertEqual(_template_scene(root), current)

    def test_reference_export_refuses_to_replace_existing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'reference'
            destination.mkdir()
            sentinel = destination / 'sentinel'
            sentinel.write_text('preserve')
            with patch('pipeline.render.auto_annotation.export') as export:
                with self.assertRaisesRegex(FileExistsError, 'choose a new version'):
                    atomic_export_reference(object(), destination, Path('template.xml'), True)
            self.assertEqual(sentinel.read_text(), 'preserve')
            export.assert_not_called()

    def test_reference_export_refuses_to_remove_stale_staging_data(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'reference'
            building = destination.with_name(destination.name + '.building')
            building.mkdir()
            sentinel = building / 'sentinel'
            sentinel.write_text('preserve')
            with patch('pipeline.render.auto_annotation.export') as export:
                with self.assertRaisesRegex(FileExistsError, 'inspect before retrying'):
                    atomic_export_reference(object(), destination, Path('template.xml'), True)
            self.assertEqual(sentinel.read_text(), 'preserve')
            export.assert_not_called()

    def test_incomplete_reference_is_preserved_and_formal_output_is_versioned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'demo'
            original.mkdir()
            sentinel = original / 'contact_sheet.png'
            sentinel.write_text('preview-only')

            destination = reference_destination(root, 'demo')

            self.assertEqual(destination, root / 'demo_v2')
            self.assertEqual(sentinel.read_text(), 'preview-only')

    def test_preview_is_isolated_from_published_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw'
            output = root / 'output'
            asset = raw / 'demo'
            asset.mkdir(parents=True)
            (asset / 'assembly.json').write_text(json.dumps({
                'occurrences': {'root': {'name': 'base', 'is_grounded': True}},
                'joints': {'joint': {
                    'name': 'hinge',
                    'joint_motion': {'joint_type': 'RevoluteJointType'},
                }},
            }))
            (output / 'references' / 'demo').mkdir(parents=True)
            (output / 'references' / 'demo' / 'sentinel').write_text('preserve')
            (output / 'manifest.jsonl').write_text('published manifest\n')
            (output / 'auto_annotation_status.json').write_text('published status\n')
            (output / 'review_queue.json').write_text('published queue\n')
            sample = output / 'samples' / 'approved_multiview'
            sample.mkdir(parents=True)
            (sample / 'scene.xml').write_text('<mujoco/>')

            def fake_export(asset, destination, **kwargs):
                reference = Path(destination)
                reference.mkdir(parents=True)
                (reference / 'contact_sheet.png').write_bytes(b'contact sheet')
                (reference / 'motion.npz').write_bytes(b'motion')
                (reference / 'quality.json').write_text(json.dumps({
                    'status': 'kinematic_preview', 'mujoco_fk_error': 0.0,
                }))

            with patch('pipeline.render.auto_annotation.export', side_effect=fake_export) as export, \
                    patch('pipeline.render.auto_annotation.Assembly', return_value=object()), \
                    patch('pipeline.render.auto_annotation.package_dataset') as package:
                report = run_dataset(raw, output, None, .5, .01, 2, True)

            self.assertEqual(report[0]['status'], 'preview_ready')
            self.assertTrue((output / 'references' / 'demo' / 'sentinel').exists())
            self.assertEqual((output / 'manifest.jsonl').read_text(), 'published manifest\n')
            self.assertEqual((output / 'auto_annotation_status.json').read_text(), 'published status\n')
            self.assertEqual((output / 'review_queue.json').read_text(), 'published queue\n')
            self.assertTrue((output / 'previews' / 'preview_status.json').exists())
            self.assertTrue((output / 'previews' / 'references' / 'demo').is_dir())
            self.assertIn('evidence', json.loads(
                (output / 'previews' / 'preview_status.json').read_text())[0])
            export.assert_called_once()
            package.assert_not_called()

    def test_preview_approval_is_auditable_and_invalidated_by_changed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw'
            output = root / 'output'
            asset = raw / 'demo'
            asset.mkdir(parents=True)
            source = asset / 'assembly.json'
            source.write_text('{"source": true}')
            preview = output / 'previews' / 'references' / 'demo'
            preview.mkdir(parents=True)
            (preview / 'contact_sheet.png').write_bytes(b'contact sheet')
            (preview / 'motion.npz').write_bytes(b'motion')
            (preview / 'quality.json').write_text(json.dumps({
                'status': 'kinematic_preview', 'mujoco_fk_error': 0.0,
            }))
            report = [{'asset': 'demo', 'status': 'preview_ready'}]
            (output / 'previews' / 'preview_status.json').write_text(json.dumps(report))
            report[0]['evidence'] = _preview_evidence(raw, output, 'demo')
            (output / 'previews' / 'preview_status.json').write_text(json.dumps(report))

            decision = record_preview_decision(raw, output, 'demo', 'approved',
                                               'reviewer-1', 'Motion and framing reviewed')

            self.assertEqual(decision['decision'], 'approved')
            self.assertEqual(_approved_preview(raw, output, 'demo'), (True, 'preview_approved'))
            (preview / 'contact_sheet.png').write_bytes(b'changed preview')
            self.assertEqual(_approved_preview(raw, output, 'demo'),
                             (False, 'preview_approval_stale'))

    def test_preview_approval_tracks_versioned_output_not_old_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, output = root / 'raw', root / 'output'
            (raw / 'demo').mkdir(parents=True)
            (raw / 'demo' / 'assembly.json').write_text('{}')
            base = output / 'previews' / 'references' / 'demo'
            version = output / 'previews' / 'references' / 'demo_v2'
            base.mkdir(parents=True)
            version.mkdir(parents=True)
            for folder, marker in ((base, b'old'), (version, b'current')):
                (folder / 'contact_sheet.png').write_bytes(marker)
                (folder / 'motion.npz').write_bytes(marker)
                (folder / 'quality.json').write_text(json.dumps({
                    'status': 'kinematic_preview', 'mujoco_fk_error': 0.0,
                }))
            report = [{'asset': 'demo', 'status': 'preview_ready',
                       'output': str(version)}]
            report[0]['evidence'] = _preview_evidence(raw, output, 'demo', report[0])
            status = output / 'previews' / 'preview_status.json'
            status.parent.mkdir(parents=True, exist_ok=True)
            status.write_text(json.dumps(report))

            record_preview_decision(raw, output, 'demo', 'approved',
                                    'reviewer-1', 'Versioned preview reviewed')
            (base / 'contact_sheet.png').write_bytes(b'changed old output')
            self.assertEqual(_approved_preview(raw, output, 'demo'),
                             (True, 'preview_approved'))
            (version / 'contact_sheet.png').write_bytes(b'changed current output')
            self.assertEqual(_approved_preview(raw, output, 'demo'),
                             (False, 'preview_approval_stale'))

    def test_preview_decision_requires_successful_fresh_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw'
            output = root / 'output'
            (raw / 'demo').mkdir(parents=True)
            (raw / 'demo' / 'assembly.json').write_text('{}')
            (output / 'previews').mkdir(parents=True)
            (output / 'previews' / 'preview_status.json').write_text(json.dumps([
                {'asset': 'demo', 'status': 'review_required'},
            ]))
            with self.assertRaisesRegex(ValueError, 'no successful preview'):
                record_preview_decision(raw, output, 'demo', 'approved',
                                        'reviewer-1', 'Looks good')

    def test_formal_export_is_blocked_until_preview_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, output = root / 'raw', root / 'output'
            asset = raw / 'demo'
            asset.mkdir(parents=True)
            (asset / 'assembly.json').write_text(json.dumps({
                'occurrences': {'root': {'name': 'base', 'is_grounded': True}},
                'joints': {'joint': {
                    'name': 'hinge',
                    'joint_motion': {'joint_type': 'RevoluteJointType'},
                }},
            }))
            sample = output / 'samples' / 'existing_multiview'
            sample.mkdir(parents=True)
            (sample / 'scene.xml').write_text('<mujoco/>')
            preview = output / 'previews' / 'references' / 'demo'
            preview.mkdir(parents=True)
            (preview / 'contact_sheet.png').write_bytes(b'contact sheet')
            (preview / 'motion.npz').write_bytes(b'motion')
            (preview / 'quality.json').write_text(json.dumps({
                'status': 'kinematic_preview', 'mujoco_fk_error': 0.0,
            }))
            preview_report = [{'asset': 'demo', 'status': 'preview_ready'}]
            preview_report[0]['evidence'] = _preview_evidence(raw, output, 'demo')
            status_path = output / 'previews' / 'preview_status.json'
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(json.dumps(preview_report))

            with patch('pipeline.render.auto_annotation.render_backend_error', return_value=None), \
                    patch('pipeline.render.auto_annotation.export') as export, \
                    patch('pipeline.render.auto_annotation.Assembly', return_value=object()), \
                    patch('pipeline.render.auto_annotation.package_dataset', return_value={'rejected': []}):
                report = run_dataset(raw, output, None, .5, .01, 2, False)

            self.assertEqual(report[0]['status'], 'awaiting_preview_review')
            export.assert_not_called()


if __name__ == '__main__':
    unittest.main()
