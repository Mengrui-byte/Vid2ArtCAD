import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np

from pipeline.render.video_annotation import import_video
from pipeline.render.video_annotation import import_multiview
from pipeline.render.review_server import ReviewApplication
from pipeline.render.validate_sample import validate_sample
from pipeline.render.package_dataset import package_dataset


class VideoAnnotationTest(unittest.TestCase):
    def test_all_background_human_review_is_not_accepted_as_a_labeled_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'input.mp4'
            writer = imageio.get_writer(video, fps=1, codec='libx264', macro_block_size=1)
            writer.append_data(np.zeros((8, 10, 3), dtype=np.uint8))
            writer.close()
            sample = root / 'real' / 'blank-labels'
            import_video(video, sample)
            app = ReviewApplication(sample)
            app.save_mask(0, 'reviewer', np.zeros((8, 10), dtype='<u2').tobytes())

            report = validate_sample(sample)
            self.assertEqual(report['status'], 'review')
            self.assertIn('part_masks_have_no_foreground:cam01', report['reason_codes'])

    def test_multiview_import_requires_review_of_every_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = {}
            masks = {}
            cameras = {}
            for view in ('cam01', 'cam02'):
                video = root / f'{view}.mp4'
                writer = imageio.get_writer(video, fps=2, codec='libx264', macro_block_size=1)
                writer.append_data(np.zeros((4, 6, 3), dtype=np.uint8))
                writer.append_data(np.full((4, 6, 3), 100, dtype=np.uint8))
                writer.close()
                mask_dir = root / f'masks_{view}'; mask_dir.mkdir()
                imageio.imwrite(mask_dir / '000000.png', np.full((4, 6), 513, dtype=np.uint16))
                imageio.imwrite(mask_dir / '000001.png', np.full((4, 6), 513, dtype=np.uint16))
                camera = root / f'camera_{view}.json'
                camera.write_text(json.dumps({
                    'resolution_wh': [6, 4], 'fps': 2,
                    'K': [[1, 0, 3], [0, 1, 2], [0, 0, 1]],
                    'T_world_camera': np.eye(4).tolist(),
                }))
                videos[view] = video; masks[view] = mask_dir; cameras[view] = camera
            output_root = root / 'output'; sample = output_root / 'real' / 'multi'
            report = import_multiview(videos, sample, masks, cameras)
            self.assertEqual(report['frame_count'], 2)
            self.assertEqual(validate_sample(sample)['status'], 'review')
            for view in ('cam01', 'cam02'):
                app = ReviewApplication(sample, f'rgb_{view}.mp4')
                app.save_part(513, 'moving part')
                for frame in range(2):
                    app.save_mask(frame, 'tester', np.full((4, 6), 513, dtype='<u2').tobytes())
            validation = validate_sample(sample)
            self.assertEqual(validation['status'], 'accepted', validation)
            quality = json.loads((sample / 'quality.json').read_text())
            self.assertEqual(quality['labels']['source'], 'human_reference')
            self.assertEqual(quality['views']['cam01']['labels']['origin'], 'pseudo')
            self.assertEqual(quality['views']['cam01']['labels']['source'], 'human_reference')
            for view in ('cam01', 'cam02'):
                self.assertEqual(quality['views'][view]['labels']['origin'], 'pseudo')
            self.assertEqual(quality['views']['cam02']['labels']['source'], 'human_reference')

            # Equal mask counts must not hide missing or misnumbered frames.
            mask_path = sample / 'labels' / 'part_id_cam02' / '000001.png'
            mask_path.rename(mask_path.with_name('000002.png'))
            checksums = {
                str(path.relative_to(sample)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(sample.rglob('*'))
                if path.is_file() and path.name != 'checksums.json'
            }
            (sample / 'checksums.json').write_text(json.dumps(checksums))
            invalid = validate_sample(sample)
            self.assertEqual(invalid['status'], 'review')
            self.assertIn('mask_frame_index_coverage_incomplete:cam02', invalid['reason_codes'])
            mask_path.with_name('000002.png').rename(mask_path)
            checksums = {
                str(path.relative_to(sample)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(sample.rglob('*'))
                if path.is_file() and path.name != 'checksums.json'
            }
            (sample / 'checksums.json').write_text(json.dumps(checksums))

            manifest = package_dataset(output_root)
            self.assertEqual(manifest['sample_count'], 1, manifest)
            self.assertEqual(manifest['accepted'][0]['videos']['cam02']['frame_count'], 2)

    def test_imports_video_and_uint16_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'input.mp4'
            writer = imageio.get_writer(video, fps=2, codec='libx264', macro_block_size=1)
            writer.append_data(np.zeros((4, 4, 3), dtype=np.uint8))
            writer.append_data(np.ones((4, 4, 3), dtype=np.uint8))
            writer.close()
            masks = root / 'masks'
            masks.mkdir()
            imageio.imwrite(masks / '000000.png', np.ones((4, 4), dtype=np.uint16))
            imageio.imwrite(masks / '000001.png', np.full((4, 4), 2, dtype=np.uint16))
            report = import_video(video, root / 'sample', masks=masks)
            self.assertEqual(report['status'], 'pseudo_labels_imported')
            self.assertEqual(report['video']['frame_count'], 2)
            self.assertEqual(len(list((root / 'sample' / 'labels' / 'part_id_cam01').glob('*.png'))), 2)

    def test_rejects_mask_resolution_mismatch_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'input.mp4'
            writer = imageio.get_writer(video, fps=2, codec='libx264', macro_block_size=1)
            writer.append_data(np.zeros((4, 4, 3), dtype=np.uint8))
            writer.close()
            masks = root / 'masks'
            masks.mkdir()
            imageio.imwrite(masks / '000000.png', np.ones((3, 4), dtype=np.uint16))
            output = root / 'sample'
            with self.assertRaisesRegex(ValueError, 'mask resolution'):
                import_video(video, output, masks=masks)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name('sample.building').exists())

    def test_rejects_color_masks_instead_of_dropping_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'input.mp4'
            writer = imageio.get_writer(video, fps=2, codec='libx264', macro_block_size=1)
            writer.append_data(np.zeros((4, 4, 3), dtype=np.uint8))
            writer.close()
            masks = root / 'masks'
            masks.mkdir()
            imageio.imwrite(masks / '000000.png', np.ones((4, 4, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, '2D integer'):
                import_video(video, root / 'sample', masks=masks)


if __name__ == '__main__':
    unittest.main()
