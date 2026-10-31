"""Persistent annotation task queue and cost accounting."""

from __future__ import annotations

import json
import math
import time
import numpy as np
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from pipeline.render.validate_sample import validate_sample
from pipeline.render.file_lock import file_lock


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict:
    if not path.exists():
        return {'schema_version': '1.0', 'hourly_rate_usd': 25.0, 'tasks': []}
    return json.loads(path.read_text())


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


@contextmanager
def _queue_lock(path: Path):
    """Serialize queue read-modify-write cycles across threads and processes."""
    with file_lock(path.with_name(path.name + '.lock')):
        yield


def _review_status(sample: Path, view: str) -> tuple[str, list[str]]:
    path = sample / ('review.json' if view == 'cam01' else f'review_{view}.json')
    if not path.exists():
        return 'pending', ['review_metadata_missing']
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return 'pending', ['review_metadata_invalid']
    if value.get('status') != 'human_review_complete':
        return 'pending', ['human_review_incomplete']
    try:
        sample_data = json.loads((sample / 'sample.json').read_text())
        quality = json.loads((sample / 'quality.json').read_text())
    except (OSError, ValueError):
        return 'pending', ['review_metadata_invalid']
    if not str(value.get('reviewer') or '').strip():
        return 'pending', ['reviewer_missing']
    views = quality.get('views', {})
    frame_count = (views.get(view, {}).get('frame_count') if isinstance(views, dict) else None)
    if frame_count is None and view == 'cam01':
        frame_count = quality.get('video', {}).get('frame_count')
    if not isinstance(frame_count, int) or frame_count <= 0:
        return 'pending', ['video_frame_count_missing']
    try:
        reviewed = {int(frame) for frame in value.get('reviewed_frames', [])}
    except (TypeError, ValueError):
        return 'pending', ['reviewed_frame_indices_invalid']
    if reviewed != set(range(frame_count)):
        return 'pending', ['reviewed_frame_coverage_incomplete']
    spec = _view_specs(sample, sample_data).get(view, {})
    label_dir = sample / str(spec.get('part_id') or f'labels/part_id_{view}')
    files = {path.name for path in label_dir.glob('*.png')} if label_dir.is_dir() else set()
    expected = {f'{index:06d}.png' for index in range(frame_count)}
    if files != expected:
        return 'pending', ['mask_frame_coverage_incomplete']
    return 'done', []


def _view_specs(sample: Path, sample_data: dict) -> dict[str, dict]:
    if isinstance(sample_data.get('views'), dict):
        return sample_data['views']
    return {'cam01': {
        'part_id': sample_data.get('part_id', 'labels/part_id_cam01'),
        'video': sample_data.get('video', 'rgb.mp4'),
    }}


def _sample_file(sample: Path, relative: object) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    path = (sample / relative).resolve()
    try:
        path.relative_to(sample.resolve())
    except ValueError:
        return None
    return path


def _tracking_artifact_valid(sample: Path, data: dict, view: str, spec: dict) -> bool:
    tracks = data.get('tracks', {})
    relative = tracks.get(view) if isinstance(tracks, dict) else tracks
    path = _sample_file(sample, relative or spec.get('tracks'))
    quality_path = _sample_file(sample, 'quality.json')
    try:
        quality = json.loads(quality_path.read_text())
        views = quality.get('views', {})
        frame_count = views.get(view, {}).get('frame_count') if isinstance(views, dict) else None
        if frame_count is None and view == 'cam01':
            frame_count = quality.get('video', {}).get('frame_count')
        frame_count = int(frame_count)
        if frame_count < 1 or path is None:
            return False
        records = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get('points'), list):
                return False
            records.append(record)
        return (len(records) == frame_count and
                [item.get('frame_index') for item in records] == list(range(frame_count)))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _decode_artifact_valid(sample: Path, data: dict) -> bool:
    decoded = _sample_file(sample, data.get('decoded_motion'))
    quality = _sample_file(sample, data.get('motion_decode_quality'))
    if decoded is None or quality is None:
        return False
    try:
        report = json.loads(quality.read_text())
        with np.load(decoded, allow_pickle=False) as arrays:
            q = arrays['q']
            valid = arrays['q_valid']
            return (report.get('status') == 'motion_decode_passed' and
                    int(report.get('frame_count', -1)) == q.shape[0] and
                    int(report.get('successful_frames', -1)) == q.shape[0] and
                    q.ndim == 2 and q.shape[0] > 0 and np.isfinite(q).all() and
                    valid.shape == q.shape and valid.astype(bool).all())
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _decode_artifact_state(sample: Path, data: dict) -> tuple[str, list[str]]:
    decoded = _sample_file(sample, data.get('decoded_motion'))
    quality = _sample_file(sample, data.get('motion_decode_quality'))
    if decoded is None or quality is None:
        return 'pending', ['decoded_motion_missing_or_incomplete']
    try:
        report = json.loads(quality.read_text())
        with np.load(decoded, allow_pickle=False) as arrays:
            q, valid = arrays['q'], arrays['q_valid']
            if (q.ndim != 2 or not q.shape[0] or not np.isfinite(q).all()
                    or valid.shape != q.shape):
                return 'review', ['decoded_motion_invalid_or_incomplete']
        if _decode_artifact_valid(sample, data):
            return 'done', []
        if report.get('status') == 'motion_decode_partial':
            return 'review', ['motion_decode_partial_or_unobservable']
        if report.get('status') == 'review':
            return 'pending', ['decoded_motion_missing_or_incomplete']
        return 'review', ['decoded_motion_missing_or_incomplete']
    except (OSError, ValueError, TypeError, KeyError):
        return 'review', ['decoded_motion_invalid_or_incomplete']


def _synthetic_sample_complete(sample: Path) -> bool:
    try:
        quality = json.loads((sample / 'quality.json').read_text())
        labels = json.loads((sample / 'labels' / 'quality.json').read_text())
        return quality.get('status') == 'multiview_replay_passed' and labels.get('status') == 'labels_derived'
    except (OSError, ValueError, TypeError):
        return False


def _task(task_id: str, sample: Path, view: str, stage: str, status: str,
          reasons: list[str], previous: dict | None = None,
          sample_id: str | None = None) -> dict:
    value = dict(previous or {})
    previous_status = value.get('status')
    if status == 'pending' and previous_status in {'claimed', 'review', 'failed'}:
        status = previous_status
    elif status == 'pending' and previous_status == 'done':
        _archive_attempt(value)
    elif status == 'done' and previous_status == 'claimed':
        status = 'claimed'
    value.update({
        'task_id': task_id, 'sample_id': sample_id or sample.name, 'sample_path': str(sample),
        'view': view, 'stage': stage, 'status': status, 'reason_codes': reasons,
    })
    value.setdefault('worker', None)
    value.setdefault('created_at', _now())
    value.setdefault('started_at', None)
    value.setdefault('completed_at', None)
    value.setdefault('duration_s', 0.0)
    value.setdefault('notes', None)
    return value


def _derive_tasks(output_root: Path, previous: dict[str, dict], data_root: Path | None = None) -> list[dict]:
    tasks = []
    if data_root is not None and data_root.is_dir():
        status_path = output_root / 'auto_annotation_status.json'
        status_by_asset = {
            item.get('asset'): item for item in (_load_json(status_path) if status_path.is_file() else [])
            if isinstance(item, dict) and item.get('asset')
        }
        for asset in sorted(path for path in data_root.iterdir()
                            if path.is_dir() and (path / 'assembly.json').is_file()):
            asset_id = asset.name
            sample = output_root / 'samples' / f'{asset_id}_multiview'
            item = status_by_asset.get(asset_id, {})
            run_status = item.get('status')
            try:
                assembly = json.loads((asset / 'assembly.json').read_text())
                joint_count = len(assembly.get('joints') or {})
                occurrence_count = len(assembly.get('occurrences') or {})
                qualification_status = 'done' if joint_count and occurrence_count else 'not_applicable'
                qualification_reasons = [] if qualification_status == 'done' else ['no_kinematic_joints_or_occurrences']
            except (OSError, ValueError, TypeError):
                qualification_status = 'failed'
                qualification_reasons = ['assembly_json_invalid']
            key = f'{asset_id}:asset:qualification'
            tasks.append(_task(key, asset, 'asset', 'qualification', qualification_status,
                               qualification_reasons, previous.get(key), asset_id))
            preview_path = output_root / 'previews' / 'preview_status.json'
            preview_records = _load_json(preview_path)
            preview = next((record for record in reversed(preview_records)
                            if record.get('asset') == asset_id), None)
            preview_rejected = False
            if preview and preview.get('status') == 'preview_ready':
                from pipeline.render.auto_annotation import (
                    _latest_preview_decision, _preview_evidence,
                )
                decision = _latest_preview_decision(output_root, asset_id)
                fresh = False
                if decision:
                    try:
                        fresh = decision.get('evidence') == _preview_evidence(
                            data_root, output_root, asset_id)
                    except (OSError, ValueError, TypeError):
                        pass
                preview_rejected = bool(fresh and decision.get('decision') == 'rejected')
                review_status = 'done' if fresh else 'pending'
                review_reasons = [] if fresh else ['preview_review_pending']
                if preview_rejected:
                    review_reasons = ['preview_rejected']
                key = f'{asset_id}:asset:preview_review'
                preview_task = _task(key, asset, 'asset', 'preview_review', review_status,
                                     review_reasons, previous.get(key), asset_id)
                preview_task['preview_output_root'] = str(output_root)
                preview_task['data_root'] = str(data_root.resolve())
                tasks.append(preview_task)
            if _synthetic_sample_complete(sample):
                annotate_status, annotate_reasons = 'done', []
            elif run_status in {'failed', 'label_backfill_failed'}:
                annotate_status, annotate_reasons = 'failed', [run_status]
            elif run_status == 'awaiting_preview_review':
                annotate_status, annotate_reasons = 'pending', ['preview_review_pending']
            elif run_status == 'review_required':
                annotate_status, annotate_reasons = 'review', ['preview_rejected_or_source_review_required']
            elif preview_rejected:
                annotate_status, annotate_reasons = 'review', ['preview_rejected']
            elif run_status == 'blocked_render_backend':
                annotate_status, annotate_reasons = 'pending', ['render_backend_blocked']
            elif run_status == 'skipped_no_active_joints':
                annotate_status, annotate_reasons = 'not_applicable', ['no_active_joints']
            else:
                annotate_status, annotate_reasons = 'pending', ['annotation_not_generated']
            key = f'{asset_id}:asset:annotate'
            tasks.append(_task(key, sample, 'asset', 'annotate', annotate_status,
                               annotate_reasons, previous.get(key), asset_id))
    real_root = output_root / 'real'
    if real_root.is_dir():
        for sample in sorted(path for path in real_root.iterdir() if path.is_dir()):
            sample_json_path = sample / 'sample.json'
            if not sample_json_path.exists():
                continue
            try:
                sample_data = json.loads(sample_json_path.read_text())
            except (OSError, ValueError):
                continue
            if sample_data.get('mode') != 'real_video_annotation':
                continue
            for view, spec in sorted(_view_specs(sample, sample_data).items()):
                review_status, reasons = _review_status(sample, view)
                key = f'{sample.name}:{view}:review'
                tasks.append(_task(key, sample, view, 'review', review_status, reasons, previous.get(key)))
                tracks = sample_data.get('tracks', {})
                track_path = tracks.get(view) if isinstance(tracks, dict) else tracks
                if not track_path:
                    track_path = spec.get('tracks')
                has_mask = (sample / 'labels' / f'part_id_{view}').is_dir()
                tracking_status = ('done' if _tracking_artifact_valid(sample, sample_data, view, spec)
                                   else ('pending' if has_mask else 'not_applicable'))
                tracking_reasons = [] if tracking_status == 'done' else (
                    ['tracks_missing'] if has_mask else ['part_id_mask_missing'])
                if has_mask and track_path and not _tracking_artifact_valid(sample, sample_data, view, spec):
                    tracking_reasons = ['tracks_invalid_or_incomplete']
                key = f'{sample.name}:{view}:tracking'
                tasks.append(_task(key, sample, view, 'tracking', tracking_status,
                                   tracking_reasons, previous.get(key)))
                has_reference = bool(sample_data.get('reference_sample') or sample_data.get('reference'))
                decoded_path = sample_data.get('decoded_motion')
                decode_status, decode_reasons = (_decode_artifact_state(sample, sample_data)
                                                 if has_reference else
                                                 ('not_applicable', ['reference_sample_missing']))
                key = f'{sample.name}:{view}:decode'
                tasks.append(_task(key, sample, view, 'decode', decode_status,
                                   decode_reasons, previous.get(key)))
    samples_root = output_root / 'samples'
    if samples_root.is_dir():
        for sample in sorted(samples_root.glob('*_multiview')):
            key = f'{sample.name}:dataset:publish'
            validation = validate_sample(sample, check_video=False)
            status = 'done' if validation['status'] == 'accepted' else 'pending'
            reasons = [] if status == 'done' else validation.get('reason_codes', ['validation_failed'])
            tasks.append(_task(key, sample, 'dataset', 'publish', status, reasons, previous.get(key)))
    return tasks


def _load_json(path: Path) -> list:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, list) else []
    except (OSError, ValueError, TypeError):
        return []


def _summary(queue: dict) -> dict:
    tasks = queue.get('tasks', [])
    counts = {}
    for task in tasks:
        counts[task['status']] = counts.get(task['status'], 0) + 1
    total_seconds = sum(
        sum(float(attempt.get('duration_s') or 0) for attempt in task.get('attempt_history', []))
        + float(task.get('duration_s') or 0)
        for task in tasks
    )
    return {
        'task_count': len(tasks), 'status_counts': counts,
        'total_duration_s': total_seconds,
        'total_duration_h': total_seconds / 3600,
        'estimated_cost_usd': total_seconds / 3600 * float(queue.get('hourly_rate_usd', 25.0)),
        'hourly_rate_usd': float(queue.get('hourly_rate_usd', 25.0)),
    }


def build_queue(output_root: Path, queue_path: Path | None = None,
                hourly_rate_usd: float = 25.0, data_root: Path | None = None) -> dict:
    hourly_rate_usd = float(hourly_rate_usd)
    if not math.isfinite(hourly_rate_usd) or hourly_rate_usd < 0:
        raise ValueError('hourly_rate_usd must be a finite nonnegative number')
    output_root = output_root.resolve()
    queue_path = queue_path or output_root / 'annotation_queue.json'
    with _queue_lock(queue_path):
        current = _load(queue_path)
        previous = {task['task_id']: task for task in current.get('tasks', [])}
        queue = {
            'schema_version': '1.0', 'generated_at': _now(),
            'dataset_root': str(output_root), 'data_root': str(data_root.resolve()) if data_root else None,
            'hourly_rate_usd': hourly_rate_usd,
            'tasks': _derive_tasks(output_root, previous, data_root),
        }
        queue['summary'] = _summary(queue)
        _write(queue_path, queue)
        return queue


def claim_task(queue_path: Path, task_id: str, worker: str) -> dict:
    if not worker.strip():
        raise ValueError('worker is required')
    with _queue_lock(queue_path):
        queue = _load(queue_path)
        for task in queue.get('tasks', []):
            if task['task_id'] == task_id:
                if task['status'] != 'pending':
                    raise ValueError(f'task is not pending: {task["status"]}')
                task.update({'status': 'claimed', 'worker': worker.strip(), 'started_at': _now()})
                queue['updated_at'] = _now(); queue['summary'] = _summary(queue)
                _write(queue_path, queue)
                return task
    raise KeyError(f'unknown task: {task_id}')


def complete_task(queue_path: Path, task_id: str, status: str = 'done',
                  duration_s: float | None = None, notes: str | None = None) -> dict:
    if status not in {'done', 'review', 'failed'}:
        raise ValueError('status must be done, review, or failed')
    with _queue_lock(queue_path):
        queue = _load(queue_path)
        for task in queue.get('tasks', []):
            if task['task_id'] == task_id:
                if task['status'] not in {'claimed', 'pending'}:
                    raise ValueError(f'task cannot be completed from {task["status"]}')
                if status == 'done' and not _task_done_gate(task):
                    raise ValueError('task cannot be marked done until its source artifacts pass the task gate')
                if duration_s is None and task.get('started_at'):
                    try:
                        started = datetime.fromisoformat(task['started_at']).timestamp()
                        duration_s = max(0.0, time.time() - started)
                    except ValueError:
                        duration_s = 0.0
                duration_s = float(duration_s or 0)
                if not math.isfinite(duration_s) or duration_s < 0:
                    raise ValueError('duration_s must be a finite nonnegative number')
                task.update({'status': status, 'completed_at': _now(),
                             'duration_s': duration_s, 'notes': notes})
                queue['updated_at'] = _now(); queue['summary'] = _summary(queue)
                _write(queue_path, queue)
                return task
    raise KeyError(f'unknown task: {task_id}')


def _archive_attempt(task: dict) -> None:
    if any(task.get(key) for key in ('worker', 'started_at', 'completed_at', 'duration_s', 'notes')):
        task.setdefault('attempt_history', []).append({
            key: task.get(key) for key in
            ('status', 'worker', 'started_at', 'completed_at', 'duration_s', 'notes')
        })
    task.update({'worker': None, 'started_at': None, 'completed_at': None,
                 'duration_s': 0.0, 'notes': None})


def reopen_task(queue_path: Path, task_id: str, notes: str | None = None) -> dict:
    with _queue_lock(queue_path):
        queue = _load(queue_path)
        for task in queue.get('tasks', []):
            if task['task_id'] == task_id:
                if task['status'] not in {'review', 'failed'}:
                    raise ValueError(f'task cannot be reopened from {task["status"]}')
                _archive_attempt(task)
                task.update({
                    'status': 'pending', 'worker': None, 'started_at': None,
                    'completed_at': None, 'duration_s': 0.0, 'notes': notes,
                    'reopened_at': _now(),
                })
                queue['updated_at'] = _now()
                queue['summary'] = _summary(queue)
                _write(queue_path, queue)
                return task
    raise KeyError(f'unknown task: {task_id}')


def _task_done_gate(task: dict) -> bool:
    sample = Path(task['sample_path'])
    stage, view = task['stage'], task['view']
    if stage == 'review':
        return _review_status(sample, view)[0] == 'done'
    if stage == 'tracking':
        try:
            data = json.loads((sample / 'sample.json').read_text())
        except (OSError, ValueError):
            return False
        spec = _view_specs(sample, data).get(view, {})
        return _tracking_artifact_valid(sample, data, view, spec)
    if stage == 'decode':
        try:
            data = json.loads((sample / 'sample.json').read_text())
        except (OSError, ValueError):
            return False
        return bool(data.get('reference_sample') or data.get('reference')) and _decode_artifact_valid(sample, data)
    if stage == 'publish':
        return validate_sample(sample, check_video=False)['status'] == 'accepted'
    if stage == 'annotate':
        return _synthetic_sample_complete(sample)
    if stage == 'qualification':
        try:
            assembly = json.loads((sample / 'assembly.json').read_text())
            return bool(assembly.get('joints') and assembly.get('occurrences'))
        except (OSError, ValueError, TypeError):
            return False
    if stage == 'preview_review':
        from pipeline.render.auto_annotation import _latest_preview_decision, _preview_evidence
        output_root = Path(task['preview_output_root'])
        data_root = Path(task['data_root'])
        decision = _latest_preview_decision(output_root, task['sample_id'])
        if not decision or decision.get('decision') not in {'approved', 'rejected'}:
            return False
        try:
            return decision.get('evidence') == _preview_evidence(
                data_root, output_root, task['sample_id'])
        except (OSError, ValueError, TypeError):
            return False
    return False


def queue_report(queue_path: Path) -> dict:
    queue = _load(queue_path)
    queue['summary'] = _summary(queue)
    return queue
