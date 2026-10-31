import json
import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from pipeline.render.package_dataset import _design_family, _family_splits, _split, package_dataset


class PackageDatasetTest(unittest.TestCase):
    def test_design_family_split_is_stable(self):
        base = _design_family(Path('design-a_multiview'), {'sample_id': 'design-a_multiview'})
        variant = _design_family(Path('variant'), {
            'sample_id': 'variant', 'design_family_id': base,
        })
        self.assertEqual(_split(base), _split(variant))

    def test_small_dataset_gets_all_splits_when_possible(self):
        splits = _family_splits({'design-a', 'design-b', 'design-c', 'design-d'})
        self.assertEqual(set(splits.values()), {'train', 'val', 'test'})

    def test_manifest_contains_only_quality_complete_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accepted = root / 'samples' / 'a_multiview'
            accepted.mkdir(parents=True)
            (accepted / 'quality.json').write_text(json.dumps({
                'status': 'multiview_replay_passed', 'videos': {'cam01': {'frame_count': 2}}
            }))
            (accepted / 'sample.json').write_text('{}')
            np.savez(accepted / 'motion.npz', q=np.zeros((2, 1)), timestamps=np.array([0., 1.]))
            writer = imageio.get_writer(accepted / 'rgb_cam01.mp4', fps=1, codec='libx264',
                                        macro_block_size=1)
            writer.append_data(np.zeros((2, 2, 3), dtype=np.uint8))
            writer.append_data(np.ones((2, 2, 3), dtype=np.uint8))
            writer.close()
            (accepted / 'labels').mkdir()
            (accepted / 'labels' / 'quality.json').write_text(json.dumps({
                'status': 'labels_derived', 'report': {'cameras': {'cam01': {'frame_count': 2}}}
            }))
            import hashlib
            files = [path for path in accepted.rglob('*') if path.is_file()]
            checksums = {
                str(path.relative_to(accepted)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files
            }
            (accepted / 'checksums.json').write_text(json.dumps(checksums))
            rejected = root / 'samples' / 'b_multiview'
            rejected.mkdir(parents=True)
            (rejected / 'quality.json').write_text(json.dumps({'status': 'review'}))
            (rejected / 'sample.json').write_text('{}')
            report = package_dataset(root)
            self.assertEqual(report['sample_count'], 1)
            self.assertEqual(len(report['rejected']), 1)
            self.assertTrue((root / 'manifest.jsonl').exists())

    def test_manifest_discovers_legacy_root_level_samples_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / 'samples' / 'same_multiview'
            current.mkdir(parents=True)
            (current / 'sample.json').write_text(json.dumps({'sample_id': 'same_multiview'}))
            (current / 'quality.json').write_text(json.dumps({'status': 'review'}))
            legacy = root / 'legacy'
            legacy.mkdir()
            (legacy / 'sample.json').write_text(json.dumps({'sample_id': 'same_multiview'}))
            (legacy / 'quality.json').write_text(json.dumps({'status': 'review'}))
            other = root / 'older_sample'
            other.mkdir()
            (other / 'sample.json').write_text(json.dumps({'sample_id': 'older_multiview'}))
            (other / 'quality.json').write_text(json.dumps({'status': 'review'}))

            report = package_dataset(root)

            self.assertEqual([item['sample_id'] for item in report['rejected']],
                             ['same_multiview', 'older_sample'])


if __name__ == '__main__':
    unittest.main()
