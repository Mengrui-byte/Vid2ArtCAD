"""Import real RGB video and externally produced pseudo-labels safely."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from pipeline.render.stereo_annotation import digest, dump


def _inspect_video(video: Path) -> dict:
    reader = imageio.get_reader(video)
    try:
        metadata = reader.get_meta_data()
        frame_count = 0
        frame_shape = None
        for frame in reader:
            frame = np.asarray(frame)
            if frame.ndim != 3 or frame.shape[2] < 3:
                raise ValueError(f'video frame must be RGB: {frame.shape}')
            shape = frame.shape[:2]
            if frame_shape is None:
                frame_shape = shape
            elif shape != frame_shape:
                raise ValueError('video resolution changes across frames')
            frame_count += 1
    finally:
        reader.close()
    if frame_count <= 0:
        raise ValueError('input video contains no frames')
    fps = float(metadata.get('fps', 0.0))
    size = metadata.get('size')
    if not size or fps <= 0:
        raise ValueError('input video metadata must contain size and positive fps')
    width, height = map(int, size)
    if frame_shape != (height, width):
        raise ValueError(f'video metadata resolution {(width, height)} != decoded {(frame_shape[1], frame_shape[0])}')
    return {'frame_count': frame_count, 'fps': fps, 'resolution_wh': [width, height]}


def _validate_masks(masks: Path, metadata: dict) -> list[Path]:
    if not masks.is_dir():
        raise FileNotFoundError(masks)
    files = sorted(path for path in masks.iterdir()
                   if path.suffix.lower() in {'.png', '.tif', '.tiff'})
    if len(files) != metadata['frame_count']:
        raise ValueError(f'mask count {len(files)} != video frame count {metadata["frame_count"]}')
    expected = {f'{index:06d}' for index in range(metadata['frame_count'])}
    if {path.stem for path in files} != expected:
        raise ValueError('mask filenames must cover contiguous 000000..N frame indices')
    for mask_path in files:
        mask = np.asarray(imageio.imread(mask_path))
        if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f'mask must be a 2D integer image: {mask_path}')
        if mask.shape != (metadata['resolution_wh'][1], metadata['resolution_wh'][0]):
            raise ValueError(f'mask resolution {mask.shape[::-1]} != video resolution {tuple(metadata["resolution_wh"])}: {mask_path}')
        if mask.size and int(mask.max()) > np.iinfo(np.uint16).max:
            raise ValueError(f'mask value exceeds uint16: {mask_path}')
    return files


def import_multiview(videos: dict[str, Path], output: Path,
                     masks: dict[str, Path] | None = None,
                     cameras: dict[str, Path] | None = None,
                     tracks: dict[str, Path] | None = None,
                     source_label: str = 'pseudo', design_family_id: str | None = None) -> dict:
    """Import synchronized real views into one reviewable sample.

    ``cam01`` is the default review view; every view gets independent masks,
    checksums, and human-review state. Video frame counts and fps must agree.
    """
    if output.exists():
        raise FileExistsError(output)
    if source_label not in {'measured', 'pseudo', 'human_reference'}:
        raise ValueError('source_label must be measured, pseudo, or human_reference')
    if not videos:
        raise ValueError('at least one named video view is required')
    if 'cam01' not in videos:
        raise ValueError('videos must include cam01')
    for name, video in videos.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError(f'invalid view name: {name}')
        if not video.is_file():
            raise FileNotFoundError(video)
    metadata = {name: _inspect_video(video) for name, video in videos.items()}
    reference = metadata['cam01']
    for name, value in metadata.items():
        if value['frame_count'] != reference['frame_count']:
            raise ValueError(f'view {name} frame count does not match cam01')
        if not np.isclose(value['fps'], reference['fps'], atol=1e-6, rtol=0):
            raise ValueError(f'view {name} fps does not match cam01')
    masks = masks or {}
    cameras = cameras or {}
    tracks = tracks or {}
    mask_files = {name: _validate_masks(path, metadata[name]) for name, path in masks.items()}
    for name, camera in cameras.items():
        if name not in videos:
            raise ValueError(f'camera provided for unknown view: {name}')
        if not camera.is_file():
            raise FileNotFoundError(camera)
    for name, track in tracks.items():
        if name not in videos:
            raise ValueError(f'tracks provided for unknown view: {name}')
        if not track.is_file():
            raise FileNotFoundError(track)

    stage = output.with_name(output.name + '.building')
    if stage.exists():
        raise FileExistsError(f'incomplete staging directory exists: {stage}')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=True)
    try:
        view_report = {}
        view_specs = {}
        for name, video in videos.items():
            video_name = f'rgb_{name}.mp4'
            shutil.copy2(video, stage / video_name)
            camera_name = None
            if name in cameras:
                camera_name = f'camera_{name}.json'
                shutil.copy2(cameras[name], stage / camera_name)
            label_dir_name = f'labels/part_id_{name}'
            if name in mask_files:
                label_dir = stage / label_dir_name
                label_dir.mkdir(parents=True)
                for index, mask_path in enumerate(mask_files[name]):
                    imageio.imwrite(label_dir / f'{index:06d}.png',
                                    np.asarray(imageio.imread(mask_path)).astype(np.uint16))
            tracks_name = None
            if name in tracks:
                tracks_name = f'tracks_{name}.jsonl'
                shutil.copy2(tracks[name], stage / tracks_name)
            review_name = 'review.json' if name == 'cam01' else f'review_{name}.json'
            view_specs[name] = {
                'video': video_name, 'camera': camera_name,
                'part_id': label_dir_name,
                'tracks': tracks_name, 'review': review_name,
            }
            view_report[name] = {
                **metadata[name], 'video': video_name,
                'labels': {'part_id': name in mask_files,
                           'source': source_label if name in mask_files else None,
                           'origin': source_label if name in mask_files else None,
                           'convention': '0 background; 65535 ignore; uint16 PNG'},
                'camera': camera_name is not None, 'tracks': tracks_name is not None,
            }
            dump(stage / review_name, {
                'schema_version': '1.0', 'status': 'needs_review',
                'video': video_name, 'view': name, 'reviewed_frames': [], 'reviewer': None,
            })
        status = 'pseudo_labels_imported' if mask_files else 'needs_review'
        dump(stage / 'quality.json', {
            'schema_version': '1.0', 'status': status, 'source_label': source_label,
            'views': view_report, 'synchronized': True,
            'frame_count': reference['frame_count'], 'fps': reference['fps'],
        })
        dump(stage / 'sample.json', {
            'schema_version': '1.0', 'mode': 'real_video_annotation',
            'sample_id': output.name, 'design_family_id': design_family_id or output.name,
            'video': 'rgb_cam01.mp4', 'camera': view_specs['cam01']['camera'],
            'part_id': view_specs['cam01']['part_id'], 'views': view_specs,
            'quality': 'quality.json', 'review_views': {name: spec['review'] for name, spec in view_specs.items()},
        })
        dump(stage / 'checksums.json', {
            str(path.relative_to(stage)): digest(path)
            for path in sorted(stage.rglob('*'))
            if path.is_file() and path.name != 'checksums.json'
        })
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {'schema_version': '1.0', 'status': status, 'views': view_report,
            'frame_count': reference['frame_count'], 'fps': reference['fps']}


def import_video(video: Path, output: Path, masks: Path | None = None,
                 tracks: Path | None = None, camera: Path | None = None,
                 source_label: str = 'pseudo', design_family_id: str | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if source_label not in {'measured', 'pseudo', 'human_reference'}:
        raise ValueError('source_label must be measured, pseudo, or human_reference')
    metadata = _inspect_video(video)
    frame_count = metadata['frame_count']
    fps = metadata['fps']
    width, height = metadata['resolution_wh']

    mask_files = []
    if masks is not None:
        mask_files = _validate_masks(masks, metadata)
    if tracks is not None and not tracks.is_file():
        raise FileNotFoundError(tracks)
    if tracks is not None:
        seen_frames = set()
        for line_number, line in enumerate(tracks.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError(f'invalid tracks JSONL at line {line_number}') from exc
            if not isinstance(record, dict):
                raise ValueError(f'track record must be an object at line {line_number}')
            if 'frame_index' in record:
                frame_index = record['frame_index']
                if not isinstance(frame_index, int) or not 0 <= frame_index < frame_count:
                    raise ValueError(f'invalid track frame_index at line {line_number}')
                if frame_index in seen_frames:
                    raise ValueError(f'duplicate track frame_index at line {line_number}')
                seen_frames.add(frame_index)
    if camera is not None and not camera.is_file():
        raise FileNotFoundError(camera)

    stage = output.with_name(output.name + '.building')
    if stage.exists():
        raise FileExistsError(f'incomplete staging directory exists: {stage}')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=True)
    try:
        shutil.copy2(video, stage / 'rgb.mp4')
        if camera is not None:
            shutil.copy2(camera, stage / 'camera.json')
        label_count = 0
        if masks is not None:
            label_dir = stage / 'labels' / 'part_id_cam01'
            label_dir.mkdir(parents=True)
            for index, mask_path in enumerate(mask_files):
                mask = np.asarray(imageio.imread(mask_path))
                imageio.imwrite(label_dir / f'{index:06d}.png', mask.astype(np.uint16))
                label_count += 1
        if tracks is not None:
            shutil.copy2(tracks, stage / 'tracks.jsonl')
        status = 'pseudo_labels_imported' if masks is not None else 'needs_review'
        report = {
            'schema_version': '1.0',
            'status': status,
            'source_label': source_label,
            'video': {'frame_count': frame_count, 'fps': fps, 'resolution_wh': [width, height]},
            'labels': {'part_id': label_count == frame_count if masks is not None else False,
                       'source': source_label if masks is not None else None,
                       'origin': source_label if masks is not None else None,
                       'convention': '0 background; 65535 ignore; uint16 PNG'},
            'tracks': tracks is not None,
            'camera': camera is not None,
        }
        dump(stage / 'quality.json', report)
        dump(stage / 'sample.json', {
            'schema_version': '1.0', 'mode': 'real_video_annotation',
            'sample_id': output.name,
            'design_family_id': design_family_id or output.name,
            'video': 'rgb.mp4', 'camera': 'camera.json' if camera else None,
            'part_id': 'labels/part_id_cam01',
            'tracks': 'tracks.jsonl' if tracks else None, 'quality': 'quality.json',
        })
        dump(stage / 'checksums.json', {
            str(path.relative_to(stage)): digest(path)
            for path in sorted(stage.rglob('*'))
            if path.is_file() and path.name != 'checksums.json'
        })
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--masks', type=Path)
    parser.add_argument('--tracks', type=Path)
    parser.add_argument('--camera', type=Path)
    parser.add_argument('--source-label', default='pseudo')
    parser.add_argument('--design-family-id')
    args = parser.parse_args()
    print(json.dumps(import_video(args.video, args.output, args.masks, args.tracks,
                                   args.camera, args.source_label, args.design_family_id),
                     ensure_ascii=False, indent=2))
