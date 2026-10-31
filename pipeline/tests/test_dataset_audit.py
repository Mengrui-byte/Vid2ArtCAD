import json
import tempfile
import unittest
from pathlib import Path

from pipeline.render.dataset_audit import audit_dataset


class DatasetAuditTest(unittest.TestCase):
    def test_reports_joint_types_and_missing_mesh_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / 'asset'
            asset.mkdir()
            (asset / 'assembly.json').write_text(json.dumps({
                'occurrences': {
                    'a': {'name': 'base', 'is_grounded': True},
                    'b': {'name': 'arm', 'is_grounded': False},
                },
                'joints': {
                    'j': {'joint_motion': {'joint_type': 'RevoluteJointType'}},
                },
            }))
            report = audit_dataset(root)
            self.assertEqual(report['asset_count'], 1)
            self.assertEqual(report['joint_type_totals']['RevoluteJointType'], 1)
            self.assertEqual(report['assets'][0]['status'], 'review_required')


if __name__ == '__main__':
    unittest.main()
