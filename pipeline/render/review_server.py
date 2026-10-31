"""Local human-review UI for video-aligned uint16 part masks."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import imageio.v2 as imageio
import numpy as np

from pipeline.render.file_lock import file_lock


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def _refresh_changed_checksums(sample: Path, changed: list[Path]) -> None:
    manifest_path = sample / 'checksums.json'
    if manifest_path.exists():
        checksums = json.loads(manifest_path.read_text())
    else:
        checksums = {}
        for path in sorted(sample.rglob('*')):
            if path.is_file() and path.name != 'checksums.json':
                checksums[str(path.relative_to(sample))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in changed:
        checksums[str(path.relative_to(sample))] = hashlib.sha256(path.read_bytes()).hexdigest()
    _atomic_json(manifest_path, checksums)


class ReviewApplication:
    def __init__(self, sample: Path, video_name: str = 'rgb.mp4'):
        self.sample = sample.resolve()
        requested = self.sample / video_name
        if not requested.is_file() and video_name == 'rgb.mp4' and (self.sample / 'rgb_cam01.mp4').is_file():
            requested = self.sample / 'rgb_cam01.mp4'
        self.video = requested.resolve()
        if self.video.parent != self.sample or not self.video.is_file():
            raise FileNotFoundError(f'video must be a file directly inside the sample: {video_name}')
        self.view = self.video.stem.removeprefix('rgb_') if self.video.stem.startswith('rgb_') else 'cam01'
        reader = imageio.get_reader(self.video)
        try:
            metadata = reader.get_meta_data()
            self.frame_count = sum(1 for _ in reader)
            self.fps = float(metadata.get('fps', 0))
            width, height = map(int, metadata['size'])
        finally:
            reader.close()
        if self.frame_count < 1 or self.fps <= 0 or width < 1 or height < 1:
            raise ValueError('video must have frames, positive fps, and valid dimensions')
        self.width, self.height = width, height
        self.label_dir = self.sample / 'labels' / f'part_id_{self.view}'
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.review_path = self.sample / ('review.json' if self.view == 'cam01' else f'review_{self.view}.json')
        self.part_mapping_path = self.sample / 'human_part_mapping.json'
        self.sample_lock_path = self.sample.parent / f'.{self.sample.name}.annotation.lock'
        self.quality_path = self.sample / 'quality.json'
        self.lock = threading.RLock()
        with file_lock(self.sample_lock_path):
            if not self.review_path.exists():
                _atomic_json(self.review_path, {
                    'schema_version': '1.0', 'status': 'needs_review',
                    'video': self.video.name, 'view': self.view,
                    'reviewed_frames': [], 'reviewer': None,
                })
                _refresh_changed_checksums(self.sample, [self.review_path])

    def part_mapping(self) -> dict:
        if not self.part_mapping_path.exists():
            return {
                'schema_version': '1.0',
                'convention': '0 background; 65535 ignore; uint16 PNG',
                'parts': [],
            }
        value = json.loads(self.part_mapping_path.read_text())
        if not isinstance(value.get('parts'), list):
            raise ValueError('human part mapping must contain a parts list')
        return value

    def save_part(self, part_id: int, name: str) -> dict:
        if not 1 <= part_id < 65535:
            raise ValueError('part ID must be between 1 and 65534')
        name = name.strip()
        if not name:
            raise ValueError('part name is required')
        with self.lock, file_lock(self.sample_lock_path):
            mapping = self.part_mapping()
            parts = mapping['parts']
            duplicate_name = next((part for part in parts
                                   if part.get('name', '').casefold() == name.casefold()
                                   and int(part.get('part_id', -1)) != part_id), None)
            if duplicate_name:
                raise ValueError(f'part name already belongs to ID {duplicate_name["part_id"]}')
            existing = next((part for part in parts
                             if int(part.get('part_id', -1)) == part_id), None)
            if existing:
                existing['name'] = name
            else:
                parts.append({'part_id': part_id, 'name': name, 'source': 'human'})
            parts.sort(key=lambda part: int(part['part_id']))
            mapping.update({
                'schema_version': '1.0',
                'convention': '0 background; 65535 ignore; uint16 PNG',
                'parts': parts,
            })
            _atomic_json(self.part_mapping_path, mapping)
            _refresh_changed_checksums(self.sample, [self.part_mapping_path])
            return mapping

    def mask_path(self, frame: int) -> Path:
        if not 0 <= frame < self.frame_count:
            raise ValueError('frame index is outside the video')
        return self.label_dir / f'{frame:06d}.png'

    def review(self) -> dict:
        if self.review_path.exists():
            value = json.loads(self.review_path.read_text())
        else:
            value = {'status': 'needs_review', 'reviewed_frames': [], 'reviewer': None}
        value['reviewed_count'] = len(set(value.get('reviewed_frames', [])))
        value['frame_count'] = self.frame_count
        return value

    def save_mask(self, frame: int, reviewer: str, raw: bytes) -> dict:
        expected_bytes = self.width * self.height * 2
        if len(raw) != expected_bytes:
            raise ValueError(f'mask payload must be exactly {expected_bytes} bytes')
        array = np.frombuffer(raw, dtype='<u2').reshape(self.height, self.width)
        with self.lock, file_lock(self.sample_lock_path):
            state = json.loads(self.review_path.read_text())
            reviewer_name = reviewer.strip() or state.get('reviewer')
            if not reviewer_name:
                raise ValueError('reviewer name is required before saving a reviewed frame')
            path = self.mask_path(frame)
            temporary = path.with_name(path.stem + '.tmp.png')
            imageio.imwrite(temporary, array)
            temporary.replace(path)
            reviewed = set(int(index) for index in state.get('reviewed_frames', []))
            reviewed.add(frame)
            complete = len(reviewed) == self.frame_count
            state.update({
                'schema_version': '1.0',
                'status': 'human_review_complete' if complete else 'human_review_in_progress',
                'video': self.video.name,
                'reviewed_frames': sorted(reviewed),
                'reviewer': reviewer_name,
            })
            _atomic_json(self.review_path, state)
            if self.quality_path.exists():
                quality = json.loads(self.quality_path.read_text())
            else:
                quality = {'schema_version': '1.0'}
            review_key = 'human_review' if self.view == 'cam01' else f'human_review_{self.view}'
            quality[review_key] = {
                'status': state['status'],
                'reviewed_frames': len(reviewed),
                'frame_count': self.frame_count,
                'reviewer': state['reviewer'],
            }
            if complete:
                if self.view == 'cam01':
                    quality.setdefault('labels', {})['source'] = 'human_reference'
                    if self.view in quality.get('views', {}):
                        quality['views'][self.view].setdefault('labels', {})['source'] = 'human_reference'
                else:
                    quality.setdefault('views', {}).setdefault(self.view, {}).setdefault('labels', {})['source'] = 'human_reference'
            _atomic_json(self.quality_path, quality)
            _refresh_changed_checksums(self.sample, [path, self.review_path, self.quality_path])
            return self.review()


def make_handler(app: ReviewApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'Vid2ArtCADReview/1.0'

        def log_message(self, _format, *_args):
            return

        def send_bytes(self, status: int, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, value: dict):
            self.send_bytes(status, json.dumps(value, ensure_ascii=False).encode(), 'application/json')

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == '/':
                html = Path(__file__).with_name('review.html').read_bytes()
                return self.send_bytes(200, html, 'text/html; charset=utf-8')
            if parsed.path == '/api/info':
                return self.send_json(200, {
                    'video': self.path_video, 'view': app.view,
                    'frame_count': app.frame_count,
                    'fps': app.fps, 'width': app.width, 'height': app.height,
                    'review': app.review(),
                })
            if parsed.path == '/api/review':
                return self.send_json(200, app.review())
            if parsed.path == '/api/parts':
                try:
                    return self.send_json(200, app.part_mapping())
                except (OSError, ValueError, TypeError) as exc:
                    return self.send_json(409, {'error': str(exc)})
            if parsed.path == '/api/mask':
                try:
                    frame = int(query.get('frame', ['-1'])[0])
                    path = app.mask_path(frame)
                except (ValueError, TypeError):
                    return self.send_json(400, {'error': 'invalid frame index'})
                if not path.exists():
                    empty = np.zeros((app.height, app.width), dtype='<u2')
                    return self.send_bytes(200, empty.tobytes(), 'application/octet-stream')
                mask = np.asarray(imageio.imread(path))
                if mask.shape != (app.height, app.width) or mask.dtype != np.uint16:
                    return self.send_json(409, {'error': 'stored mask shape or dtype is invalid'})
                return self.send_bytes(200, mask.astype('<u2', copy=False).tobytes(),
                                       'application/octet-stream')
            if parsed.path == '/media':
                return self.send_video()
            return self.send_json(404, {'error': 'not found'})

        @property
        def path_video(self):
            return app.video.name

        def send_video(self):
            size = app.video.stat().st_size
            range_header = self.headers.get('Range')
            start, end = 0, size - 1
            status = 200
            if range_header:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', range_header.strip())
                if not match:
                    return self.send_json(416, {'error': 'invalid byte range'})
                left, right = match.groups()
                if left:
                    start = int(left)
                    end = int(right) if right else end
                elif right:
                    start = max(0, size - int(right))
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.end_headers()
                    return
                end = min(end, size - 1)
                status = 206
            with app.video.open('rb') as stream:
                stream.seek(start)
                body = stream.read(end - start + 1)
            self.send_response(status)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(len(body)))
            if status == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == '/api/propagate':
                try:
                    request = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
                    with app.lock:
                        from pipeline.render.video_propagation import propagate
                        result = propagate(app.sample, int(request['seed_frame']),
                                           int(request.get('radius', 30)), view=app.view)
                    return self.send_json(200, result)
                except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
                    return self.send_json(400, {'error': str(exc)})
            if parsed.path == '/api/parts':
                try:
                    request = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
                    mapping = app.save_part(int(request['part_id']), str(request.get('name', '')))
                    return self.send_json(200, mapping)
                except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
                    return self.send_json(400, {'error': str(exc)})
            if parsed.path != '/api/mask':
                return self.send_json(404, {'error': 'not found'})
            try:
                frame = int(query.get('frame', ['-1'])[0])
                reviewer = query.get('reviewer', [''])[0]
                content_length = int(self.headers.get('Content-Length', '-1'))
                if content_length < 0 or content_length > app.width * app.height * 2:
                    raise ValueError('invalid request body size')
                raw = self.rfile.read(content_length)
                state = app.save_mask(frame, reviewer, raw)
            except (ValueError, TypeError, OSError) as exc:
                return self.send_json(400, {'error': str(exc)})
            self.send_json(200, state)

    return Handler


def serve(sample: Path, video: str = 'rgb.mp4', host: str = '127.0.0.1', port: int = 8765):
    app = ReviewApplication(sample, video)
    server = ThreadingHTTPServer((host, port), make_handler(app))
    print(f'Review UI: http://{host}:{server.server_address[1]}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--video', default='rgb.mp4')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    serve(args.sample, args.video, args.host, args.port)
