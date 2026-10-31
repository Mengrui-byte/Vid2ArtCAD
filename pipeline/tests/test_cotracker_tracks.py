import json
import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from pipeline.render.cotracker_tracks import track_cotracker


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def permute(self, *axes):
        return _Tensor(self.value.transpose(axes))

    def float(self):
        return self

    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, item):
        return _Tensor(self.value[item])


class _Torch:
    @staticmethod
    def from_numpy(value):
        return _Tensor(value)


class _Model:
    def __call__(self, video, queries=None):
        count = queries.value.shape[1]
        tracks = np.zeros((1, video.value.shape[1], count, 2), dtype=np.float32)
        tracks[:, :, :, 0] = 2
        tracks[:, :, :, 1] = 1
        visibility = np.ones((1, video.value.shape[1], count), dtype=np.float32)
        return _Tensor(tracks), _Tensor(visibility)


class CoTrackerTest(unittest.TestCase):
    def test_tracks_are_written_with_part_ids_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / 'sample'
            label_dir = sample / 'labels' / 'part_id_cam01'
            label_dir.mkdir(parents=True)
            writer = imageio.get_writer(sample / 'rgb.mp4', fps=2, codec='libx264', macro_block_size=1)
            writer.append_data(np.zeros((4, 6, 3), dtype=np.uint8))
            writer.append_data(np.ones((4, 6, 3), dtype=np.uint8))
            writer.close()
            mask = np.zeros((4, 6), dtype=np.uint16)
            mask[1, 1] = 513; mask[2, 4] = 514
            imageio.imwrite(label_dir / '000000.png', mask)
            (sample / 'sample.json').write_text(json.dumps({'mode': 'real_video_annotation'}))
            (sample / 'quality.json').write_text('{}')
            report = track_cotracker(sample, model=_Model(), torch_module=_Torch())
            self.assertEqual(report['query_count'], 2)
            lines = (sample / 'tracks_cam01.jsonl').read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])['points'][0]['part_id'], 513)
            sample_json = json.loads((sample / 'sample.json').read_text())
            self.assertEqual(sample_json['tracks']['cam01'], 'tracks_cam01.jsonl')
            self.assertTrue((sample / 'checksums.json').is_file())


if __name__ == '__main__':
    unittest.main()
