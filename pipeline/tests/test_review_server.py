import json
import hashlib
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import imageio.v2 as imageio
import numpy as np

from pipeline.render.review_server import ReviewApplication, make_handler
from pipeline.render.package_dataset import package_dataset
from pipeline.render.validate_sample import validate_sample
from pipeline.render.video_annotation import import_multiview


class ReviewServerTest(unittest.TestCase):
    def test_concurrent_view_reviewers_preserve_shared_sample_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = {}
            for view, value in (('cam01', 40), ('cam02', 120)):
                video = root / f'{view}.mp4'
                writer = imageio.get_writer(video, fps=1, codec='libx264', macro_block_size=1)
                writer.append_data(np.full((4, 6, 3), value, dtype=np.uint8))
                writer.close()
                videos[view] = video
            sample = root / 'samples' / 'concurrent-review'
            import_multiview(videos, sample, design_family_id='fixture-family')
            reviewers = [
                ReviewApplication(sample, video_name=f'rgb_{view}.mp4')
                for view in ('cam01', 'cam02')
            ]

            def annotate(app, part_id, part_name):
                app.save_part(part_id, part_name)
                mask = np.full((4, 6), part_id, dtype='<u2')
                return app.save_mask(0, f'reviewer-{app.view}', mask.tobytes())

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(
                    lambda args: annotate(*args),
                    ((reviewers[0], 513, 'left link'), (reviewers[1], 514, 'right link')),
                ))

            mapping = json.loads((sample / 'human_part_mapping.json').read_text())
            self.assertEqual(
                [(part['part_id'], part['name']) for part in mapping['parts']],
                [(513, 'left link'), (514, 'right link')],
            )
            quality = json.loads((sample / 'quality.json').read_text())
            self.assertEqual(quality['human_review']['status'], 'human_review_complete')
            self.assertEqual(quality['human_review_cam02']['status'], 'human_review_complete')
            self.assertEqual(len(results), 2)
            checksums = json.loads((sample / 'checksums.json').read_text())
            self.assertEqual(
                set(checksums),
                {str(path.relative_to(sample)) for path in sample.rglob('*')
                 if path.is_file() and path.name != 'checksums.json'},
            )
            validation = validate_sample(sample)
            self.assertEqual(validation['status'], 'accepted', validation)

    def test_named_part_ids_persist_in_mapping_and_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / 'sample'
            sample.mkdir()
            video = sample / 'rgb.mp4'
            writer = imageio.get_writer(video, fps=1, codec='libx264', macro_block_size=1)
            writer.append_data(np.zeros((4, 4, 3), dtype=np.uint8))
            writer.close()
            app = ReviewApplication(sample)
            server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_address[1]}'
            try:
                initial = json.loads(urlopen(base + '/api/parts').read())
                self.assertEqual(initial['parts'], [])
                request = Request(
                    base + '/api/parts',
                    data=json.dumps({'part_id': 17, 'name': 'upper arm'}).encode(),
                    method='POST', headers={'Content-Type': 'application/json'},
                )
                saved = json.loads(urlopen(request).read())
                self.assertEqual(saved['parts'], [
                    {'part_id': 17, 'name': 'upper arm', 'source': 'human'},
                ])
                mapping = json.loads((sample / 'human_part_mapping.json').read_text())
                self.assertEqual(mapping, saved)
                checksums = json.loads((sample / 'checksums.json').read_text())
                self.assertIn('human_part_mapping.json', checksums)

                duplicate = Request(
                    base + '/api/parts',
                    data=json.dumps({'part_id': 18, 'name': 'upper arm'}).encode(),
                    method='POST', headers={'Content-Type': 'application/json'},
                )
                with self.assertRaises(Exception) as error:
                    urlopen(duplicate)
                self.assertEqual(error.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_optical_flow_propagation_stays_pseudo_until_manually_saved(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest('install the propagation extra to run optical-flow tests')
        from pipeline.render.video_propagation import propagate

        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / 'sample'
            label_dir = sample / 'labels' / 'part_id_cam01'
            label_dir.mkdir(parents=True)
            video = sample / 'rgb.mp4'
            frames = []
            for x in (4, 6, 8):
                frame = np.zeros((32, 32, 3), dtype=np.uint8)
                frame[10:20, x:x + 8] = 255
                frames.append(frame)
            writer = imageio.get_writer(video, fps=5, codec='libx264', macro_block_size=1)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            seed = np.zeros((32, 32), dtype=np.uint16)
            seed[10:20, 4:12] = 513
            imageio.imwrite(label_dir / '000000.png', seed)
            (sample / 'review.json').write_text(json.dumps({
                'status': 'human_review_in_progress', 'video': 'rgb.mp4',
                'reviewed_frames': [0], 'reviewer': 'tester',
            }))
            report = propagate(sample, seed_frame=0, radius=2, max_width=32)
            self.assertIn(1, report['written_frames'])
            self.assertEqual(report['reviewed_frames'], [0])
            propagated = imageio.imread(label_dir / '000001.png')
            self.assertEqual(propagated.dtype, np.uint16)
            self.assertIn(513, np.unique(propagated))
            state = json.loads((sample / 'review.json').read_text())
            self.assertNotIn(1, state['reviewed_frames'])
            self.assertTrue(state['pseudo_frames']['1']['review_required'])
            checksums = json.loads((sample / 'checksums.json').read_text())
            self.assertIn('labels/part_id_cam01/000001.png', checksums)

            second_video = sample / 'rgb_cam02.mp4'
            second_writer = imageio.get_writer(second_video, fps=5, codec='libx264', macro_block_size=1)
            for frame in frames:
                second_writer.append_data(frame)
            second_writer.close()
            second_labels = sample / 'labels' / 'part_id_cam02'
            second_labels.mkdir(parents=True)
            imageio.imwrite(second_labels / '000000.png', seed)
            (sample / 'review_cam02.json').write_text(json.dumps({
                'status': 'human_review_in_progress', 'video': 'rgb_cam02.mp4',
                'view': 'cam02', 'reviewed_frames': [0], 'reviewer': 'tester',
            }))
            second = propagate(sample, seed_frame=0, radius=2, max_width=32, view='cam02')
            self.assertEqual(second['view'], 'cam02')
            self.assertTrue((second_labels / '000001.png').is_file())
            self.assertFalse((sample / 'labels' / 'part_id_cam01' / '000002.png').exists())
            with self.assertRaisesRegex(ValueError, 'max_width'):
                propagate(sample, seed_frame=0, radius=2, max_width=0, view='cam02')

    def test_browser_review_saves_uint16_masks_and_marks_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / 'real' / 'fixture'
            sample.mkdir(parents=True)
            video = sample / 'rgb.mp4'
            writer = imageio.get_writer(video, fps=2, codec='libx264', macro_block_size=1)
            writer.append_data(np.zeros((4, 6, 3), dtype=np.uint8))
            writer.append_data(np.full((4, 6, 3), 100, dtype=np.uint8))
            writer.close()
            (sample / 'quality.json').write_text(json.dumps({
                'status': 'pseudo_labels_imported',
                'video': {'frame_count': 2, 'fps': 2.0, 'resolution_wh': [6, 4]},
            }))
            (sample / 'sample.json').write_text(json.dumps({
                'mode': 'real_video_annotation', 'sample_id': sample.name,
                'design_family_id': 'fixture-family', 'video': 'rgb.mp4',
            }))
            app = ReviewApplication(sample)
            self.assertIn('human_review_incomplete', validate_sample(sample)['reason_codes'])
            with self.assertRaisesRegex(ValueError, 'reviewer name is required'):
                app.save_mask(0, '', np.zeros((4, 6), dtype='<u2').tobytes())
            app.save_part(513, 'moving part A')
            app.save_part(514, 'moving part B')
            server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_address[1]}'
            try:
                page = urlopen(base + '/').read().decode()
                self.assertIn('保存当前帧', page)
                response = urlopen(Request(base + '/media', headers={'Range': 'bytes=0-15'}))
                self.assertEqual(response.status, 206)
                self.assertEqual(len(response.read()), 16)
                for frame, label in ((0, 513), (1, 514)):
                    mask = np.full((4, 6), label, dtype='<u2')
                    request = Request(
                        f'{base}/api/mask?frame={frame}&reviewer=tester',
                        data=mask.tobytes(), method='POST',
                        headers={'Content-Type': 'application/octet-stream'},
                    )
                    state = json.loads(urlopen(request).read())
                self.assertEqual(state['status'], 'human_review_complete')
                validation = validate_sample(sample)
                self.assertEqual(validation['status'], 'accepted', validation)
                mapping_path = sample / 'human_part_mapping.json'
                original_mapping = mapping_path.read_bytes()
                mapping = json.loads(original_mapping)
                mapping['parts'] = [part for part in mapping['parts'] if part['part_id'] == 513]
                mapping_path.write_text(json.dumps(mapping))
                checksums = {
                    str(path.relative_to(sample)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(sample.rglob('*'))
                    if path.is_file() and path.name != 'checksums.json'
                }
                (sample / 'checksums.json').write_text(json.dumps(checksums))
                unmapped = validate_sample(sample)
                self.assertIn('part_ids_missing_names:cam01', unmapped['reason_codes'])
                mapping_path.write_bytes(original_mapping)
                checksums = {
                    str(path.relative_to(sample)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(sample.rglob('*'))
                    if path.is_file() and path.name != 'checksums.json'
                }
                (sample / 'checksums.json').write_text(json.dumps(checksums))
                restored = imageio.imread(sample / 'labels' / 'part_id_cam01' / '000000.png')
                self.assertEqual(restored.dtype, np.uint16)
                self.assertTrue(np.all(restored == 513))
                quality = json.loads((sample / 'quality.json').read_text())
                self.assertEqual(quality['labels']['source'], 'human_reference')
                checksums = json.loads((sample / 'checksums.json').read_text())
                self.assertIn('review.json', checksums)
                self.assertIn('labels/part_id_cam01/000001.png', checksums)
                packaged = package_dataset(sample.parent.parent)
                self.assertEqual(packaged['sample_count'], 1, packaged)
                self.assertEqual(packaged['accepted'][0]['design_family_id'], 'fixture-family')
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
