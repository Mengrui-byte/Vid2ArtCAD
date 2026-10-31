import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from pipeline.render.expand_views import _camera_pose, expand_views
from pipeline.render.stereo_annotation import project


class ExpandViewsTest(unittest.TestCase):
    def test_camera_pose_frames_complete_3d_envelope(self):
        base = np.eye(4)
        base[:3, :3] = np.diag([1.0, -1.0, -1.0])
        points = np.array([
            [-1.0, -0.5, -0.3], [1.0, -0.5, -0.3],
            [-1.0, 0.5, 0.3], [1.0, 0.5, 0.3],
        ])
        K = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1]])
        for angle in (0, 90, 180, 270):
            transform = _camera_pose(base, points, K, 640, 480, angle)
            uv, depth = project(points, transform, K)
            self.assertTrue(np.isfinite(transform).all())
            self.assertGreater(depth.min(), 0)
            self.assertTrue(np.all(uv.min(axis=0) >= 20))
            self.assertTrue(np.all(uv.max(axis=0) < np.array([640, 480]) - 20))

    def test_failed_staged_expansion_keeps_original_sample_untouched(self):
        with TemporaryDirectory() as directory:
            sample = Path(directory) / 'asset_multiview'
            sample.mkdir()
            marker = sample / 'original.txt'
            marker.write_text('original')
            with patch('pipeline.render.expand_views._expand_views_in_place',
                       side_effect=RuntimeError('render failed')):
                with self.assertRaisesRegex(RuntimeError, 'render failed'):
                    expand_views(sample)
            self.assertEqual(marker.read_text(), 'original')
            self.assertEqual(sorted(path.name for path in sample.parent.iterdir()),
                             ['asset_multiview'])

    def test_expansion_rejects_invalid_sample_without_replacing_it(self):
        with TemporaryDirectory() as directory:
            sample = Path(directory) / 'asset_multiview'
            sample.mkdir()
            marker = sample / 'original.txt'
            marker.write_text('original')
            with patch('pipeline.render.expand_views._expand_views_in_place',
                       return_value={'view_count': 4, 'added_views': ['cam03', 'cam04']}), \
                 patch('pipeline.render.validate_sample.validate_sample',
                       return_value={'status': 'review', 'reason_codes': ['bad_sample']}):
                with self.assertRaisesRegex(ValueError, 'failed validation'):
                    expand_views(sample)
            self.assertEqual(marker.read_text(), 'original')
            self.assertEqual(sorted(path.name for path in sample.parent.iterdir()),
                             ['asset_multiview'])


if __name__ == '__main__':
    unittest.main()
