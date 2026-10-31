"""Independent validation for a published multiview annotation sample."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _check_sample_checksums(sample: Path, reasons: list[str]) -> None:
    checksums_path = sample / 'checksums.json'
    try:
        checksums = json.loads(checksums_path.read_text())
        if not checksums:
            reasons.append('empty_checksums')
        actual_files = {
            str(path.relative_to(sample))
            for path in sample.rglob('*')
            if path.is_file() and path.name != 'checksums.json'
        }
        if set(checksums) != actual_files:
            reasons.append('checksum_coverage_incomplete')
        for relative, expected in checksums.items():
            path = sample / relative
            if not path.resolve().is_relative_to(sample.resolve()) or not path.is_file():
                reasons.append(f'checksum_path_invalid:{relative}')
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                reasons.append(f'checksum_mismatch:{relative}')
    except (OSError, ValueError, TypeError):
        reasons.append('checksums_invalid')


def _read_video_metadata(path: Path) -> tuple[int, float, tuple[int, int] | None]:
    reader = imageio.get_reader(path)
    count = 0
    shape = None
    try:
        fps = float(reader.get_meta_data().get('fps', 0))
        for frame in reader:
            frame = np.asarray(frame)
            if frame.ndim != 3 or frame.shape[2] < 3:
                raise ValueError('video frame shape invalid')
            if shape is None:
                shape = frame.shape[:2]
            elif shape != frame.shape[:2]:
                raise ValueError('video resolution changes')
            count += 1
    finally:
        reader.close()
    return count, fps, shape


def _check_tracks(path: Path, frame_count: int, reasons: list[str], code: str) -> None:
    try:
        records = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get('points'), list):
                raise ValueError(f'invalid track record at line {line_number}')
            records.append(record)
        if len(records) != frame_count or [r.get('frame_index') for r in records] != list(range(frame_count)):
            reasons.append(f'track_frame_coverage_incomplete:{code}')
    except (OSError, ValueError, TypeError):
        reasons.append(f'tracks_invalid:{code}')


def _human_part_ids(sample: Path, reasons: list[str]) -> set[int]:
    path = sample / 'human_part_mapping.json'
    try:
        mapping = json.loads(path.read_text())
        parts = mapping.get('parts')
        if not isinstance(parts, list):
            raise ValueError('parts must be a list')
        ids = []
        names = []
        for part in parts:
            part_id = part.get('part_id')
            name = part.get('name')
            if isinstance(part_id, bool) or not isinstance(part_id, int) or not 1 <= part_id < 65535:
                raise ValueError('part_id must be an integer from 1 to 65534')
            if not isinstance(name, str) or not name.strip():
                raise ValueError('part names must be non-empty strings')
            ids.append(part_id)
            names.append(name.strip().casefold())
        if len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise ValueError('part IDs and names must be unique')
        return set(ids)
    except (OSError, ValueError, TypeError, AttributeError):
        reasons.append('human_part_mapping_missing_or_invalid')
        return set()


def _check_mask_part_ids(mask_ids: set[int], mapped_ids: set[int], reasons: list[str], code: str) -> None:
    observed = mask_ids - {0, 65535}
    if observed - mapped_ids:
        reasons.append(f'part_ids_missing_names:{code}')


def _decoded_motion_report(sample: Path, sample_data: dict, frame_count: int,
                           reasons: list[str]) -> dict | None:
    relative = sample_data.get('decoded_motion')
    if not relative:
        return None
    decoded = (sample / str(relative)).resolve()
    quality_path = (sample / str(sample_data.get('motion_decode_quality', ''))).resolve()
    if not decoded.is_relative_to(sample.resolve()) or not quality_path.is_relative_to(sample.resolve()):
        reasons.append('decoded_motion_path_invalid')
        return None
    try:
        with np.load(decoded, allow_pickle=False) as arrays:
            q = arrays['q']
            valid = arrays['q_valid']
            if q.ndim != 2 or q.shape[0] != frame_count or not np.isfinite(q).all():
                reasons.append('decoded_motion_invalid')
            if valid.shape != q.shape:
                reasons.append('decoded_motion_validity_shape_mismatch')
        return json.loads(quality_path.read_text())
    except (OSError, ValueError, KeyError):
        reasons.append('decoded_motion_missing_or_invalid')
        return None


def _validate_multiview_video_annotation(sample: Path, sample_data: dict, quality: dict) -> dict:
    reasons = []
    mapped_ids = _human_part_ids(sample, reasons)
    views = sample_data.get('views')
    if not isinstance(views, dict) or not views:
        return {'status': 'review', 'sample_id': sample.name,
                'reason_codes': ['multiview_metadata_invalid']}
    reference_count = None
    reference_fps = None
    for name, spec in sorted(views.items()):
        if not isinstance(spec, dict):
            reasons.append(f'view_metadata_invalid:{name}')
            continue
        video = sample / str(spec.get('video') or '')
        if not video.is_file():
            reasons.append(f'missing_video:{name}')
            continue
        try:
            count, fps, shape = _read_video_metadata(video)
        except (OSError, ValueError):
            reasons.append(f'video_invalid:{name}')
            continue
        if count <= 0 or fps <= 0 or shape is None:
            reasons.append(f'video_metadata_invalid:{name}')
        if reference_count is None:
            reference_count, reference_fps = count, fps
        elif count != reference_count or not np.isclose(fps, reference_fps, atol=1e-6, rtol=0):
            reasons.append(f'view_synchronization_mismatch:{name}')
        recorded = quality.get('views', {}).get(name, {})
        if recorded.get('frame_count') != count:
            reasons.append(f'video_frame_count_metadata_mismatch:{name}')
        if not np.isclose(float(recorded.get('fps', 0)), fps, atol=1e-6, rtol=0):
            reasons.append(f'video_fps_metadata_mismatch:{name}')
        if shape and recorded.get('resolution_wh') != [shape[1], shape[0]]:
            reasons.append(f'video_resolution_metadata_mismatch:{name}')
        review_path = sample / str(spec.get('review') or ('review.json' if name == 'cam01' else f'review_{name}.json'))
        try:
            review = json.loads(review_path.read_text())
        except (OSError, ValueError):
            reasons.append(f'review_metadata_missing:{name}')
            review = {}
        if review.get('status') != 'human_review_complete':
            reasons.append(f'human_review_incomplete:{name}')
        if not str(review.get('reviewer') or '').strip():
            reasons.append(f'reviewer_missing:{name}')
        reviewed = set(review.get('reviewed_frames', []))
        if reviewed != set(range(count)):
            reasons.append(f'reviewed_frame_coverage_incomplete:{name}')
        label_dir = sample / str(spec.get('part_id') or '')
        files = sorted(label_dir.glob('*.png')) if label_dir.is_dir() else []
        if len(files) != count:
            reasons.append(f'mask_frame_count_mismatch:{name}')
        expected_names = {f'{index:06d}.png' for index in range(count)}
        if {path.name for path in files} != expected_names:
            reasons.append(f'mask_frame_index_coverage_incomplete:{name}')
        if shape:
            mask_ids = set()
            for path in files:
                try:
                    mask = np.asarray(imageio.imread(path))
                    if mask.dtype != np.uint16 or mask.shape != shape:
                        reasons.append(f'invalid_mask:{name}:{path.name}')
                        break
                    mask_ids.update(int(value) for value in np.unique(mask))
                except (OSError, ValueError):
                    reasons.append(f'unreadable_mask:{name}:{path.name}')
                    break
            if not (mask_ids - {0, 65535}):
                reasons.append(f'part_masks_have_no_foreground:{name}')
            _check_mask_part_ids(mask_ids, mapped_ids, reasons, name)
        label_source = (quality.get('labels', {}).get('source') if name == 'cam01'
                        else quality.get('views', {}).get(name, {}).get('labels', {}).get('source'))
        if label_source != 'human_reference':
            reasons.append(f'human_reference_label_source_missing:{name}')
        if spec.get('tracks'):
            _check_tracks(sample / str(spec['tracks']), count, reasons, name)
        elif isinstance(sample_data.get('tracks'), dict) and sample_data['tracks'].get(name):
            _check_tracks(sample / str(sample_data['tracks'][name]), count, reasons, name)
    motion_report = _decoded_motion_report(sample, sample_data, reference_count or 0, reasons)
    _check_sample_checksums(sample, reasons)
    return {'status': 'accepted' if not reasons else 'review', 'sample_id': sample.name,
            'reason_codes': reasons, 'frame_count': reference_count,
            'fps': reference_fps, 'view_count': len(views),
            'available_tasks': ['rgb', 'part_id', 'human_review'] if not reasons else [],
            'motion_status': motion_report.get('status') if motion_report else None,
            'motion_reason_codes': (['decoded_motion_quality_incomplete']
                                    if motion_report and motion_report.get('status') != 'motion_decode_passed'
                                    else [])}


def validate_video_annotation(sample: Path) -> dict:
    reasons = []
    try:
        sample_data = json.loads((sample / 'sample.json').read_text())
        quality = json.loads((sample / 'quality.json').read_text())
        review = json.loads((sample / 'review.json').read_text())
    except (OSError, ValueError):
        return {'status': 'review', 'sample_id': sample.name,
                'reason_codes': ['missing_or_invalid_video_annotation_metadata']}
    if sample_data.get('mode') != 'real_video_annotation':
        reasons.append('wrong_sample_mode')
    if sample_data.get('views'):
        return _validate_multiview_video_annotation(sample, sample_data, quality)
    mapped_ids = _human_part_ids(sample, reasons)
    video = sample / str(sample_data.get('video') or 'rgb.mp4')
    if not video.is_file():
        reasons.append('missing_video')
        return {'status': 'review', 'sample_id': sample.name, 'reason_codes': reasons}
    reader = imageio.get_reader(video)
    count = 0
    shape = None
    try:
        metadata = reader.get_meta_data()
        fps = float(metadata.get('fps', 0))
        for frame in reader:
            frame = np.asarray(frame)
            if frame.ndim != 3 or frame.shape[2] < 3:
                reasons.append('video_frame_shape_invalid')
                break
            if shape is None:
                shape = frame.shape[:2]
            elif shape != frame.shape[:2]:
                reasons.append('video_resolution_changes')
                break
            count += 1
    finally:
        reader.close()
    if count <= 0 or fps <= 0:
        reasons.append('video_metadata_invalid')
    recorded_video = quality.get('video', {})
    if recorded_video.get('frame_count') != count:
        reasons.append('video_frame_count_metadata_mismatch')
    if not np.isclose(float(recorded_video.get('fps', 0)), fps, atol=1e-6, rtol=0):
        reasons.append('video_fps_metadata_mismatch')
    if shape and recorded_video.get('resolution_wh') != [shape[1], shape[0]]:
        reasons.append('video_resolution_metadata_mismatch')
    if review.get('status') != 'human_review_complete':
        reasons.append('human_review_incomplete')
    if not str(review.get('reviewer') or '').strip():
        reasons.append('reviewer_missing')
    reviewed = set(review.get('reviewed_frames', []))
    if len(reviewed) != count or reviewed != set(range(count)):
        reasons.append('reviewed_frame_coverage_incomplete')
    label_dir = sample / 'labels' / 'part_id_cam01'
    mask_files = sorted(label_dir.glob('*.png')) if label_dir.is_dir() else []
    if len(mask_files) != count:
        reasons.append('mask_frame_count_mismatch')
    expected_names = {f'{index:06d}.png' for index in range(count)}
    if {path.name for path in mask_files} != expected_names:
        reasons.append('mask_frame_index_coverage_incomplete')
    mask_ids = set()
    for path in mask_files:
        try:
            mask = np.asarray(imageio.imread(path))
            if mask.dtype != np.uint16 or mask.shape != shape:
                reasons.append(f'invalid_mask:{path.name}')
                break
            mask_ids.update(int(value) for value in np.unique(mask))
        except (OSError, ValueError):
            reasons.append(f'unreadable_mask:{path.name}')
            break
    _check_mask_part_ids(mask_ids, mapped_ids, reasons, 'cam01')
    if not (mask_ids - {0, 65535}):
        reasons.append('part_masks_have_no_foreground:cam01')
    checksums_path = sample / 'checksums.json'
    try:
        checksums = json.loads(checksums_path.read_text())
        if not checksums:
            reasons.append('empty_checksums')
        actual_files = {
            str(path.relative_to(sample))
            for path in sample.rglob('*')
            if path.is_file() and path.name != 'checksums.json'
        }
        if set(checksums) != actual_files:
            reasons.append('checksum_coverage_incomplete')
        for relative, expected in checksums.items():
            path = sample / relative
            if not path.resolve().is_relative_to(sample.resolve()) or not path.is_file():
                reasons.append(f'checksum_path_invalid:{relative}')
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                reasons.append(f'checksum_mismatch:{relative}')
    except (OSError, ValueError, TypeError):
        reasons.append('checksums_invalid')
    if quality.get('human_review', {}).get('status') != 'human_review_complete':
        reasons.append('quality_review_status_incomplete')
    if quality.get('labels', {}).get('source') != 'human_reference':
        reasons.append('human_reference_label_source_missing')
    if sample_data.get('tracks'):
        track_path = sample / str(sample_data['tracks']) if isinstance(sample_data['tracks'], str) else None
        if track_path is not None:
            _check_tracks(track_path, count, reasons, 'cam01')
    decoded_motion_report = _decoded_motion_report(sample, sample_data, count, reasons)
    return {
        'status': 'accepted' if not reasons else 'review',
        'sample_id': sample.name,
        'reason_codes': reasons,
        'frame_count': count,
        'fps': fps,
        'resolution_wh': [shape[1], shape[0]] if shape else None,
        'available_tasks': ['rgb', 'part_id', 'human_review'] if not reasons else [],
        'motion_status': decoded_motion_report.get('status') if decoded_motion_report else None,
        'motion_reason_codes': (['decoded_motion_quality_incomplete']
                                if decoded_motion_report and decoded_motion_report.get('status') != 'motion_decode_passed'
                                else []),
    }


def validate_sample(sample: Path, check_video: bool = True) -> dict:
    sample_json = sample / 'sample.json'
    if sample_json.exists():
        try:
            if json.loads(sample_json.read_text()).get('mode') == 'real_video_annotation':
                return validate_video_annotation(sample)
        except (OSError, ValueError):
            pass
    reasons = []
    quality_path = sample / 'quality.json'
    labels_path = sample / 'labels' / 'quality.json'
    motion_path = sample / 'motion.npz'
    if not quality_path.exists():
        reasons.append('missing_quality')
    if not labels_path.exists():
        reasons.append('missing_labels')
    if not motion_path.exists():
        reasons.append('missing_motion')
    if reasons:
        return {'status': 'review', 'sample_id': sample.name, 'reason_codes': reasons}

    quality = json.loads(quality_path.read_text())
    labels = json.loads(labels_path.read_text())
    motion = np.load(motion_path, allow_pickle=False)
    q = motion['q']
    timestamps = motion['timestamps']
    if quality.get('status') not in {'multiview_replay_passed', 'edit_replay_passed'}:
        reasons.append('quality_status_not_passed')
    if labels.get('status') != 'labels_derived':
        reasons.append('labels_status_not_derived')
    if q.ndim != 2 or len(q) != len(timestamps) or not np.isfinite(q).all():
        reasons.append('motion_shape_or_finite_check_failed')
    if len(timestamps) < 2 or not np.all(np.diff(timestamps) > 0):
        reasons.append('timestamps_not_strictly_increasing')

    video_report = {}
    expected_videos = quality.get('videos', {})
    for camera, metadata in expected_videos.items():
        video = sample / f'rgb_{camera}.mp4'
        if not video.exists():
            reasons.append(f'missing_video:{camera}')
            continue
        if not check_video:
            continue
        reader = imageio.get_reader(video)
        count = 0
        try:
            fps = float(reader.get_meta_data().get('fps', 0))
            for frame in reader:
                if np.asarray(frame).ndim != 3:
                    reasons.append(f'video_frame_shape:{camera}')
                    break
                count += 1
        finally:
            reader.close()
        video_report[camera] = {'frame_count': count, 'fps': fps}
        if count != metadata.get('frame_count') or count != len(q):
            reasons.append(f'video_frame_count:{camera}')

    label_cameras = labels.get('report', {}).get('cameras', {})
    for camera in expected_videos:
        if label_cameras.get(camera, {}).get('frame_count') != len(q):
            reasons.append(f'label_frame_count:{camera}')
    return {
        'status': 'accepted' if not reasons else 'review',
        'sample_id': sample.name,
        'reason_codes': reasons,
        'frame_count': int(len(q)),
        'video_report': video_report,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()
    print(json.dumps(validate_sample(args.sample, not args.no_video), ensure_ascii=False, indent=2))
