"""Optional CoTracker point tracks with part-id provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def _refresh_checksums(sample: Path) -> None:
    checksums = {}
    for path in sorted(sample.rglob('*')):
        if path.is_file() and path.name != 'checksums.json':
            checksums[str(path.relative_to(sample))] = hashlib.sha256(path.read_bytes()).hexdigest()
    _atomic_json(sample / 'checksums.json', checksums)


def _load_model(device: str | None):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('CoTracker requires PyTorch; install the optional tracking environment') from exc
    try:
        model = torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline')
    except Exception as exc:
        raise RuntimeError('could not load CoTracker3 from the official torch hub repository') from exc
    target = device or ('cuda' if torch.cuda.is_available() else
                        'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
                        else 'cpu')
    return model.to(target), torch, target


def _read_video(path: Path) -> np.ndarray:
    reader = imageio.get_reader(path)
    frames = []
    try:
        for frame in reader:
            frame = np.asarray(frame)
            if frame.ndim != 3 or frame.shape[2] < 3:
                raise ValueError(f'video frame must be RGB: {frame.shape}')
            frames.append(frame[:, :, :3])
    finally:
        reader.close()
    if not frames:
        raise ValueError('video contains no frames')
    return np.stack(frames)


def _sample_queries(mask: np.ndarray | None, frame_index: int, points_per_part: int,
                    grid_size: int, frame_shape: tuple[int, int] | None = None) -> tuple[np.ndarray, list[int]]:
    if mask is None:
        height, width = frame_shape or (256, 256)
        ys, xs = np.mgrid[0:height:grid_size * 1j, 0:width:grid_size * 1j]
        points = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
        return np.column_stack([np.full(len(points), frame_index), points]), [0] * len(points)
    labels = []
    points = []
    for part_id in sorted(int(value) for value in np.unique(mask) if 0 < int(value) < 65535):
        ys, xs = np.nonzero(mask == part_id)
        if not len(xs):
            continue
        count = min(points_per_part, len(xs))
        indices = np.linspace(0, len(xs) - 1, count, dtype=int)
        points.extend([[frame_index, float(xs[index]), float(ys[index])] for index in indices])
        labels.extend([part_id] * count)
    if not points:
        raise ValueError('seed mask contains no foreground part IDs')
    return np.asarray(points, dtype=np.float32), labels


def track_cotracker(sample: Path, view: str = 'cam01', query_frame: int = 0,
                    points_per_part: int = 16, grid_size: int = 10,
                    device: str | None = None, model=None, torch_module=None) -> dict:
    """Track seed-mask points and publish an auditable JSONL track artifact."""
    sample = sample.resolve()
    video_name = 'rgb.mp4' if view == 'cam01' and (sample / 'rgb.mp4').exists() else f'rgb_{view}.mp4'
    video = sample / video_name
    if not video.is_file():
        raise FileNotFoundError(video)
    frames = _read_video(video)
    if not 0 <= query_frame < len(frames):
        raise ValueError('query_frame is outside the video')
    label_dir = sample / 'labels' / f'part_id_{view}'
    seed_path = label_dir / f'{query_frame:06d}.png'
    seed_mask = np.asarray(imageio.imread(seed_path)) if seed_path.exists() else None
    queries, part_ids = _sample_queries(seed_mask, query_frame, points_per_part, grid_size,
                                        frames.shape[1:3])
    if model is None:
        model, torch_module, target_device = _load_model(device)
    else:
        if torch_module is None:
            try:
                import torch as torch_module
            except ImportError as exc:
                raise RuntimeError('injected CoTracker model requires PyTorch') from exc
        target_device = device or 'cpu'
    video_tensor = torch_module.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(target_device)
    query_tensor = torch_module.from_numpy(queries)[None].to(target_device)
    try:
        result = model(video_tensor, queries=query_tensor)
    except TypeError:
        result = model(video_tensor, grid_size=grid_size, grid_query_frame=query_frame)
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise ValueError('CoTracker output must contain tracks and visibility')
    tracks = result[0].detach().float().cpu().numpy()
    visibility = result[1].detach().float().cpu().numpy()
    tracks = tracks[0]
    visibility = visibility[0]
    if visibility.ndim == 3:
        visibility = visibility[..., 0]
    if tracks.shape[:2] != (len(frames), len(part_ids)):
        raise ValueError(f'CoTracker track shape {tracks.shape} does not match video/query count')
    track_path = sample / f'tracks_{view}.jsonl'
    with track_path.open('w') as stream:
        for frame_index in range(len(frames)):
            points = []
            for track_id, part_id in enumerate(part_ids):
                points.append({
                    'track_id': track_id, 'part_id': part_id,
                    'xy': [float(tracks[frame_index, track_id, 0]),
                           float(tracks[frame_index, track_id, 1])],
                    'visible': bool(visibility[frame_index, track_id] > 0.5),
                    'visibility_score': float(visibility[frame_index, track_id]),
                })
            stream.write(json.dumps({
                'frame_index': frame_index, 'query_frame': query_frame,
                'points': points,
            }, ensure_ascii=False) + '\n')
    report = {
        'schema_version': '1.0', 'status': 'tracks_generated', 'method': 'cotracker3_offline',
        'view': view, 'video': video_name, 'frame_count': len(frames),
        'query_frame': query_frame, 'query_count': len(part_ids),
        'part_ids': sorted(set(part_ids)), 'points_per_part': points_per_part,
        'device': target_device, 'source': 'pseudo',
        'tracks': track_path.name, 'review_required': True,
    }
    report_path = sample / f'cotracker_{view}.json'
    _atomic_json(report_path, report)
    sample_json_path = sample / 'sample.json'
    if sample_json_path.exists():
        sample_json = json.loads(sample_json_path.read_text())
        sample_json.setdefault('tracks', {})[view] = track_path.name
        sample_json.setdefault('tracking_reports', {})[view] = report_path.name
        _atomic_json(sample_json_path, sample_json)
    quality_path = sample / 'quality.json'
    if quality_path.exists():
        quality = json.loads(quality_path.read_text())
        quality.setdefault('tracking', {})[view] = report
        _atomic_json(quality_path, quality)
    _refresh_checksums(sample)
    return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Track part seed points with CoTracker3')
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--view', default='cam01')
    parser.add_argument('--query-frame', type=int, default=0)
    parser.add_argument('--points-per-part', type=int, default=16)
    parser.add_argument('--grid-size', type=int, default=10)
    parser.add_argument('--device')
    args = parser.parse_args()
    print(json.dumps(track_cotracker(args.sample, args.view, args.query_frame,
                                     args.points_per_part, args.grid_size, args.device),
                     ensure_ascii=False, indent=2))
