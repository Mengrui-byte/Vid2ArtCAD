import json
import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from pipeline.render.sam2_propagation import propagate_sam2


class _FakeLogits:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakePredictor:
    def init_state(self, video_path, offload_video_to_cpu=True):
        self.video_path = video_path
        return {}

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        self.seed = (frame_idx, obj_id, np.asarray(mask))

    def propagate_in_video(self, state):
        score = np.full((1, 4, 6), -1.0, dtype=np.float32)
        score[:, 1:3, 2:4] = 1.0
        yield 0, [513], _FakeLogits(score)
        yield 1, [513], _FakeLogits(score)

    def reset_state(self, state):
        pass


class Sam2PropagationTest(unittest.TestCase):
    def test_fake_sam2_writes_pseudo_masks_without_reviewing_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / 'sample'
            label_dir = sample / 'labels' / 'part_id_cam01'
            label_dir.mkdir(parents=True)
            writer = imageio.get_writer(sample / 'rgb.mp4', fps=2, codec='libx264', macro_block_size=1)
            writer.append_data(np.zeros((4, 6, 3), dtype=np.uint8))
            writer.append_data(np.ones((4, 6, 3), dtype=np.uint8))
            writer.close()
            seed = np.zeros((4, 6), dtype=np.uint16)
            seed[1:3, 2:4] = 513
            imageio.imwrite(label_dir / '000000.png', seed)
            (sample / 'review.json').write_text(json.dumps({
                'status': 'human_review_in_progress', 'video': 'rgb.mp4',
                'reviewed_frames': [0], 'reviewer': 'tester',
            }))
            (sample / 'checksums.json').write_text('{}')
            result = propagate_sam2(sample, 0, predictor=_FakePredictor())
            self.assertEqual(result['reviewed_frames'], [0])
            self.assertIn(1, result['written_frames'])
            restored = imageio.imread(label_dir / '000001.png')
            self.assertEqual(restored.dtype, np.uint16)
            self.assertIn(513, np.unique(restored))
            state = json.loads((sample / 'review.json').read_text())
            self.assertEqual(state['reviewed_frames'], [0])
            self.assertEqual(state['pseudo_frames']['1']['source'], 'sam2_pseudo')
            self.assertTrue(state['sam2_propagation_runs'][0]['reviewed_frames_unchanged'])


if __name__ == '__main__':
    unittest.main()
