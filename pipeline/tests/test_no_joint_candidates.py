import json
import tempfile
import unittest
from pathlib import Path

from pipeline.render.no_joint_candidates import REVIEW_CONTRACT, build_report, inspect_asset


class NoJointCandidateTest(unittest.TestCase):
    def test_ranks_but_does_not_infer_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / 'fixture'
            asset.mkdir()
            source = {
                'joints': {},
                'occurrences': {
                    'base': {'name': 'Frame:1', 'bodies': {'base_body': {}}},
                    'lid': {'name': 'Hinged Lid:1', 'bodies': {'lid_body': {}}},
                },
                'contacts': [{
                    'entity_one': {'occurrence': 'base'},
                    'entity_two': {'occurrence': 'lid'},
                }],
                'holes': [{'type': 'RoundHole'}],
            }
            (asset / 'assembly.json').write_text(json.dumps(source))
            (asset / 'base_body.obj').write_text('mesh')
            (asset / 'lid_body.obj').write_text('mesh')

            record = inspect_asset(asset)
            self.assertEqual(record['status'], 'human_review_required')
            self.assertIsNone(record['motion_hypothesis'])
            self.assertEqual(record['cross_occurrence_contact_pair_count'], 1)
            report = build_report(root)
            self.assertEqual(report['summary']['candidate_count'], 1)
            self.assertEqual(report['review_contract'], REVIEW_CONTRACT)

    def test_skips_assets_with_joints_or_single_occurrence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / 'fixture'
            asset.mkdir()
            (asset / 'assembly.json').write_text(json.dumps({
                'joints': {'joint': {}},
                'occurrences': {'a': {}, 'b': {}},
            }))
            self.assertIsNone(inspect_asset(asset))

            (asset / 'assembly.json').write_text(json.dumps({
                'joints': {},
                'occurrences': {'only': {'name': 'Only component', 'bodies': {}}},
            }))
            self.assertIsNone(inspect_asset(asset))


if __name__ == '__main__':
    unittest.main()
