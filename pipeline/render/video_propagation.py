"""Propagate reviewed video masks as explicitly unreviewed optical-flow suggestions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from pipeline.render.file_lock import file_lock


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def _update_checksums(sample: Path, changed: list[Path]) -> None:
    manifest = sample / 'checksums.json'
    hashes = json.loads(manifest.read_text()) if manifest.exists() else {}
    for path in changed:
        hashes[str(path.relative_to(sample))] = hashlib.sha256(path.read_bytes()).hexdigest()
    _atomic_json(manifest, hashes)


def propagate(sample: Path, seed_frame: int, radius: int = 30,
              max_width: int = 640, consistency_threshold: float = 1.5,
              view: str = 'cam01') -> dict:
    sample = sample.resolve()
    lock_path = sample.parent / f'.{sample.name}.annotation.lock'
    with file_lock(lock_path):
        return _propagate_locked(sample, seed_frame, radius, max_width, consistency_threshold, view)


def _propagate_locked(sample: Path, seed_frame: int, radius: int = 30,
                      max_width: int = 640, consistency_threshold: float = 1.5,
                      view: str = 'cam01') -> dict:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Optical-flow propagation requires `pip install -e '.[propagation]'`") from exc

    sample = sample.resolve()
    if not view or any(char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for char in view):
        raise ValueError('invalid view name')
    state_path = sample / ('review.json' if view == 'cam01' else f'review_{view}.json')
    state = json.loads(state_path.read_text())
    reviewed = {int(index) for index in state.get('reviewed_frames', [])}
    if seed_frame not in reviewed:
        raise ValueError('seed frame must be saved as a human-reviewed frame first')
    if radius < 1 or radius > 300:
        raise ValueError('radius must be between 1 and 300 frames')
    if isinstance(max_width, bool) or not isinstance(max_width, int) or max_width < 1:
        raise ValueError('max_width must be a positive integer')
    if not np.isfinite(consistency_threshold) or consistency_threshold <= 0:
        raise ValueError('consistency_threshold must be finite and positive')
    video_path = sample / state.get('video', 'rgb.mp4' if view == 'cam01' else f'rgb_{view}.mp4')
    label_dir = sample / 'labels' / f'part_id_{view}'
    seed_path = label_dir / f'{seed_frame:06d}.png'
    seed = np.asarray(imageio.imread(seed_path))
    if seed.ndim != 2 or seed.dtype != np.uint16 or not np.any((seed > 0) & (seed < 65535)):
        raise ValueError('seed frame must contain at least one uint16 part label')

    reader = imageio.get_reader(video_path)
    try:
        meta = reader.get_meta_data()
        frames = []
        first, last = max(0, seed_frame - radius), seed_frame + radius
        for index, rgb in enumerate(reader):
            if index > last:
                break
            if index < first:
                continue
            rgb = np.asarray(rgb)
            if rgb.shape[:2] != seed.shape or rgb.ndim != 3 or rgb.shape[2] < 3:
                raise ValueError(f'video frame {index} shape does not match seed mask')
            gray = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2GRAY)
            if gray.shape[1] > max_width:
                height = max(1, round(gray.shape[0] * max_width / gray.shape[1]))
                gray = cv2.resize(gray, (max_width, height), interpolation=cv2.INTER_AREA)
            frames.append((index, gray))
    finally:
        reader.close()
    if not frames or frames[0][0] > seed_frame or frames[-1][0] < seed_frame:
        raise ValueError('could not decode seed frame and requested propagation range')

    scale_shape = frames[0][1].shape
    mask = cv2.resize(seed, (scale_shape[1], scale_shape[0]), interpolation=cv2.INTER_NEAREST)
    height, width = scale_shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    entries, written, skipped = [], [], []
    existing_pseudo = state.get('pseudo_frames', {})
    # Propagate independently from the seed in each temporal direction.
    for direction in (1, -1):
        current = mask.copy()
        indices = range(seed_frame, frames[-1][0]) if direction == 1 else range(seed_frame, frames[0][0], -1)
        frame_map = {index: gray for index, gray in frames}
        for index in indices:
            target_index = index + direction
            source_gray, target_gray = frame_map[index], frame_map[target_index]
            backward = cv2.calcOpticalFlowFarneback(target_gray, source_gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            forward = cv2.calcOpticalFlowFarneback(source_gray, target_gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            map_x, map_y = xx + backward[:, :, 0], yy + backward[:, :, 1]
            warped = cv2.remap(current, map_x, map_y, cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=65535)
            sampled = cv2.remap(forward, map_x, map_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            error = np.linalg.norm(backward + sampled, axis=2)
            valid = (error <= consistency_threshold) & (map_x >= 0) & (map_x < width - 1) & (map_y >= 0) & (map_y < height - 1)
            # Ignore uncertain pixels; never reinterpret them as background.
            warped[~valid] = 65535
            confidence = float(np.mean(valid))
            target_path = label_dir / f'{target_index:06d}.png'
            if target_index in reviewed or (target_path.exists() and str(target_index) not in existing_pseudo):
                skipped.append(target_index)
            else:
                full_mask = cv2.resize(warped, (seed.shape[1], seed.shape[0]), interpolation=cv2.INTER_NEAREST)
                temporary = target_path.with_name(target_path.stem + '.tmp.png')
                imageio.imwrite(temporary, full_mask.astype(np.uint16))
                temporary.replace(target_path)
                written.append(target_index)
                existing_pseudo[str(target_index)] = {
                    'source': 'optical_flow_pseudo', 'seed_frame': seed_frame,
                    'confidence': round(confidence, 6), 'review_required': True,
                }
                entries.append(target_path)
            current = warped
    state['pseudo_frames'] = existing_pseudo
    state.setdefault('propagation_runs', []).append({
        'method': 'opencv_farneback_forward_backward', 'opencv_version': cv2.__version__,
        'seed_frame': seed_frame, 'radius': radius, 'view': view,
        'consistency_threshold_px': consistency_threshold,
        'reviewed_frames_unchanged': True,
    })
    _atomic_json(state_path, state)
    _update_checksums(sample, entries + [state_path])
    return {'written_frames': sorted(written), 'skipped_frames': sorted(set(skipped)),
            'reviewed_frames': sorted(reviewed), 'view': view,
            'status': state.get('status', 'human_review_in_progress')}
