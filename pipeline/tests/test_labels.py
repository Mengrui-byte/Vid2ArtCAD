import unittest
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from pipeline.render.labels import _segmentation_ids, _write_depth
from pipeline.render.compress_depth import _convert_file


class LabelDecodeTest(unittest.TestCase):
    def test_decodes_geom_id_first_even_when_geom_enum_is_a_real_id(self):
        geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        image = np.array([
            [[0, geom_type], [5, geom_type], [9, geom_type]],
            [[-1, -1], [2, geom_type], [3, 1]],
        ], dtype=np.int32)
        decoded = _segmentation_ids(image, {0, 2, 5, 9})
        self.assertEqual(decoded.tolist(), [[1, 6, 10], [0, 3, 0]])

    def test_decodes_type_first_format(self):
        geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        image = np.array([
            [[geom_type, 0], [geom_type, 2]],
            [[-1, -1], [1, 9]],
        ], dtype=np.int32)
        decoded = _segmentation_ids(image, {0, 2, 9})
        self.assertEqual(decoded.tolist(), [[1, 3], [0, 0]])

    def test_depth_is_losslessly_saved_as_compressed_npz(self):
        depth = np.array([[0.0, 1.25], [8.5, np.nan]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / '000001.npz'
            _write_depth(output, depth)
            self.assertTrue(output.is_file())
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(archive.files, ['depth'])
                self.assertEqual(archive['depth'].dtype, np.float32)
                np.testing.assert_equal(archive['depth'], depth)

    def test_existing_depth_migrates_without_changing_values(self):
        depth = np.linspace(0, 3, 1024, dtype=np.float32).reshape(32, 32)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / '000001.npy'
            np.save(source, depth)
            self.assertEqual(_convert_file(source), 'converted')
            self.assertFalse(source.exists())
            with np.load(source.with_suffix('.npz'), allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['depth'], depth)
            self.assertEqual(_convert_file(source), 'already_compressed')


if __name__ == '__main__':
    unittest.main()
