import tempfile
import unittest
from pathlib import Path

from pipeline.render.validate_sample import validate_sample


class ValidateSampleTest(unittest.TestCase):
    def test_missing_required_artifacts_are_review(self):
        with tempfile.TemporaryDirectory() as directory:
            report = validate_sample(Path(directory), check_video=False)
            self.assertEqual(report['status'], 'review')
            self.assertIn('missing_quality', report['reason_codes'])
            self.assertIn('missing_labels', report['reason_codes'])


if __name__ == '__main__':
    unittest.main()
