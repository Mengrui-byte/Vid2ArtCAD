"""Estimate kinematic motion from part-id mask centroids and calibrated views."""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.optimize import least_squares

from pipeline.render.mujoco_runtime import set_qpos
from pipeline.render.file_lock import file_lock
from pipeline.render.stereo_annotation import project


def _numeric_frames(directory: Path) -> list[Path]:
    files = sorted(directory.glob('*.png'))
    if not files:
        raise ValueError(f'no PNG masks found in {directory}')
    expected = list(range(len(files)))
    actual = []
    for path in files:
        try:
            actual.append(int(path.stem))
        except ValueError as exc:
            raise ValueError(f'mask filename must be a numeric frame index: {path.name}') from exc
    if actual != expected:
        raise ValueError(f'mask frames must be contiguous 000000..{len(files) - 1:06d}')
    return files


def _centroids(mask: np.ndarray, label_to_link: dict[int, int]) -> dict[int, np.ndarray]:
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError('part-id masks must be 2D integer images')
    result = {}
    for label in np.unique(mask):
        label = int(label)
        link = label_to_link.get(label)
        if link is None or label in (0, 65535):
            continue
        ys, xs = np.nonzero(mask == label)
        if len(xs):
            result[link] = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)
    return result


def _load_label_map(path: Path | None, part_count: int) -> dict[int, int]:
    if path is None:
        return {index + 1: index for index in range(part_count)}
    value = json.loads(path.read_text())
    if isinstance(value, dict) and 'label_to_link' in value:
        value = value['label_to_link']
    if not isinstance(value, dict):
        raise ValueError('label map must be a JSON object such as {"513": 0}')
    result = {int(label): int(link) for label, link in value.items()}
    if any(label <= 0 or label >= 65535 or link < 0 or link >= part_count
           for label, link in result.items()):
        raise ValueError('label map contains an out-of-range label or link index')
    return result


def _validate_camera(camera: dict, camera_path: Path) -> dict:
    try:
        width, height = camera['resolution_wh']
        K = np.asarray(camera['K'], dtype=np.float64)
        T = np.asarray(camera['T_world_camera'], dtype=np.float64)
        fps = float(camera['fps'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'incomplete camera calibration in {camera_path}') from exc
    if (isinstance(width, bool) or isinstance(height, bool)
            or int(width) != width or int(height) != height or width <= 0 or height <= 0):
        raise ValueError(f'invalid image resolution in {camera_path}')
    if K.shape != (3, 3) or T.shape != (4, 4):
        raise ValueError(f'invalid calibration matrix shape in {camera_path}')
    if not np.isfinite(K).all() or not np.isfinite(T).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f'non-finite calibration or non-positive fps in {camera_path}')
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1], atol=1e-9):
        raise ValueError(f'invalid pinhole intrinsics in {camera_path}')
    rotation = T[:3, :3]
    if (not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)):
        raise ValueError(f'T_world_camera must be a rigid transform in {camera_path}')
    distortion = np.asarray(camera.get('distortion', np.zeros(5)), dtype=np.float64)
    if not np.isfinite(distortion).all() or np.any(np.abs(distortion) > 1e-12):
        raise ValueError(f'camera images must be undistorted for pinhole decoding: {camera_path}')
    return {'K': K, 'T': T, 'width': int(width), 'height': int(height),
            'fps': fps}


def _load_view(mask_dir: Path, camera_path: Path, label_map: dict[int, int]) -> tuple[list[dict[int, np.ndarray]], dict]:
    camera = json.loads(camera_path.read_text())
    if not isinstance(camera, dict):
        raise ValueError(f'camera calibration must be a JSON object: {camera_path}')
    calibration = _validate_camera(camera, camera_path)
    width, height = calibration['width'], calibration['height']
    masks = _numeric_frames(mask_dir)
    observations = []
    for path in masks:
        mask = np.asarray(imageio.imread(path))
        if mask.shape != (height, width):
            raise ValueError(f'{path} shape {mask.shape[::-1]} != camera resolution {(width, height)}')
        observations.append(_centroids(mask, label_map))
    return observations, {**calibration, 'mask_dir': str(mask_dir)}


def _refresh_sample_metadata(sample: Path, output: Path, quality: dict) -> None:
    """Publish decoded motion only when the caller explicitly targets a real sample."""
    sample_json_path = sample / 'sample.json'
    if not sample_json_path.is_file():
        return
    sample_json = json.loads(sample_json_path.read_text())
    if sample_json.get('mode') != 'real_video_annotation':
        return
    relative = output.relative_to(sample)
    sample_json['decoded_motion'] = str(relative / 'decoded_motion.npz')
    sample_json['motion_decode_quality'] = str(relative / 'motion_decode_quality.json')
    sample_json.setdefault('available_tasks', ['rgb', 'part_id', 'human_review'])
    if quality.get('status') == 'motion_decode_passed':
        for task in ('motion', 'kinematics'):
            if task not in sample_json['available_tasks']:
                sample_json['available_tasks'].append(task)
    sample_json_path.write_text(json.dumps(sample_json, ensure_ascii=False, indent=2) + '\n')
    quality_path = sample / 'quality.json'
    sample_quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}
    sample_quality['motion_decode'] = quality
    quality_path.write_text(json.dumps(sample_quality, ensure_ascii=False, indent=2) + '\n')
    checksums = {}
    for path in sorted(sample.rglob('*')):
        if path.is_file() and path.name != 'checksums.json':
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            checksums[str(path.relative_to(sample))] = digest
    (sample / 'checksums.json').write_text(json.dumps(checksums, ensure_ascii=False, indent=2) + '\n')


def _project_links(data: mujoco.MjData, geom_groups: list[list[int]], camera: dict) -> np.ndarray:
    result = np.full((len(geom_groups), 2), np.nan, dtype=np.float64)
    for link, geom_ids in enumerate(geom_groups):
        if not geom_ids:
            continue
        points = np.asarray(data.geom_xpos[geom_ids], dtype=np.float64)
        uv, depth = project(points, camera['T'], camera['K'])
        valid = np.isfinite(uv).all(axis=1) & (depth > 1e-8)
        if np.any(valid):
            # A visible mask centroid is better approximated by the projected
            # geometry centers than by the kinematic joint/body origin.
            result[link] = np.mean(uv[valid], axis=0)
    return result


def decode_motion(reference: Path, mask_dirs: list[Path], camera_paths: list[Path], output: Path,
                  label_map_path: Path | None = None, residual_threshold_px: float = 40.0,
                  active_joints: list[str] | None = None) -> dict:
    sample = output.parent.resolve()
    lock_path = sample.parent / f'.{sample.name}.annotation.lock'
    with file_lock(lock_path):
        return _decode_motion_locked(reference, mask_dirs, camera_paths, output,
                                     label_map_path, residual_threshold_px, active_joints)


def _decode_motion_locked(reference: Path, mask_dirs: list[Path], camera_paths: list[Path], output: Path,
                          label_map_path: Path | None = None,
                          residual_threshold_px: float = 40.0,
                          active_joints: list[str] | None = None) -> dict:
    """Decode q(t) without modifying the reference sample.

    ``mask_dirs`` and ``camera_paths`` are parallel lists. Part labels map to
    link indices by ``label - 1`` unless an explicit JSON label map is given.
    """
    if len(mask_dirs) != len(camera_paths) or not mask_dirs:
        raise ValueError('at least one mask directory and matching camera are required')
    if output.exists():
        raise FileExistsError(output)
    reference = reference.resolve()
    scene_path = reference / 'scene_multiview.xml'
    motion_path = reference / 'motion.npz'
    if not scene_path.exists() or not motion_path.exists():
        raise FileNotFoundError('reference must contain scene_multiview.xml and motion.npz')
    reference_motion = np.load(motion_path, allow_pickle=False)
    link_order = reference_motion['link_order']
    joint_order = reference_motion['joint_order']
    reference_q = np.asarray(reference_motion['q'], dtype=np.float64)
    if (reference_q.ndim != 2 or reference_q.shape[1] != len(joint_order)
            or not np.isfinite(reference_q).all()):
        raise ValueError('reference q must be a finite 2D array matching joint_order')
    q_size = int(reference_q.shape[1])
    joint_names = [str(name) for name in joint_order]
    if len(set(joint_names)) != len(joint_names):
        raise ValueError('reference joint_order must contain unique names')
    if active_joints is None:
        motion_range = np.ptp(reference_q, axis=0) if len(reference_q) else np.zeros(q_size)
        active_indices = np.flatnonzero(motion_range > 1e-8)
        active_selection = 'inferred_from_reference_motion'
    else:
        unknown = sorted(set(active_joints) - set(joint_names))
        if unknown:
            raise ValueError(f'active joints absent from reference joint_order: {unknown}')
        if len(set(active_joints)) != len(active_joints):
            raise ValueError('active_joints must not contain duplicates')
        active_indices = np.asarray([joint_names.index(name) for name in active_joints], dtype=int)
        active_selection = 'explicit'
    runtime_path = reference / 'joint_runtime.json'
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else None
    label_map = _load_label_map(label_map_path, len(link_order))
    views = [_load_view(mask_dir, camera_path, label_map)
             for mask_dir, camera_path in zip(mask_dirs, camera_paths)]
    frame_count = len(views[0][0])
    if any(len(observations) != frame_count for observations, _ in views):
        raise ValueError('all mask views must contain the same frame count')
    cameras = [camera for _, camera in views]
    # Views may use different calibrated resolutions; each residual remains
    # in its own pixel coordinate system.
    fps = cameras[0]['fps']
    if any(not np.isclose(camera['fps'], fps, atol=1e-6, rtol=0) for camera in cameras[1:]):
        raise ValueError('all calibrated views must use the same fps; frames are paired by index')

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
                for name in link_order]
    if any(body_id < 0 for body_id in body_ids):
        raise ValueError('reference motion contains a link absent from scene_multiview.xml')
    body_to_link = {body_id: link for link, body_id in enumerate(body_ids)}
    geom_groups = [[] for _ in body_ids]
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        while body_id not in body_to_link and body_id > 0:
            body_id = int(model.body_parentid[body_id])
        if body_id in body_to_link:
            geom_groups[body_to_link[body_id]].append(geom_id)
    q = np.zeros((frame_count, q_size), dtype=np.float64)
    q_valid = np.zeros_like(q, dtype=bool)
    poses = np.zeros((frame_count, len(body_ids), 4, 4), dtype=np.float64)
    poses[:, :, 3, 3] = 1.0
    records = []
    current = np.zeros(q_size, dtype=np.float64)
    residuals = []
    success_count = 0
    decoded_frame_count = 0

    def evaluate(values: np.ndarray, observations: list[dict[int, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        set_qpos(model, data, values, joint_order, runtime)
        mujoco.mj_forward(model, data)
        residual = []
        valid_observations = []
        for view_index, observed in enumerate(observations):
            projected = _project_links(data, geom_groups, cameras[view_index])
            for link, point in observed.items():
                if link >= len(projected) or not np.isfinite(projected[link]).all():
                    continue
                residual.extend(projected[link] - point)
                valid_observations.append((view_index, link))
        return np.asarray(residual, dtype=np.float64), np.asarray(valid_observations, dtype=np.int64)

    for frame_index in range(frame_count):
        observations = [view[0][frame_index] for view in views]
        visible = sum(len(value) for value in observations)
        if visible < 1:
            records.append({'frame_index': frame_index, 'status': 'unobservable',
                            'visible_observations': 0, 'rmse_px': None})
            q[frame_index] = current
            continue
        # Keep temporal continuity. A frame with one visible part can still be
        # useful for a one-DOF tree, but its success is reported explicitly.
        if len(active_indices) == 0:
            residual, pairs = evaluate(current, observations)
            class _FixedResult:
                x = current
                jac = np.empty((len(residual), 0))
                success = True
                message = 'no degrees of freedom'
            result = _FixedResult()
        else:
            def selected_residual(values: np.ndarray) -> np.ndarray:
                candidate = current.copy()
                candidate[active_indices] = values
                return evaluate(candidate, observations)[0]

            result = least_squares(
                selected_residual,
                current[active_indices],
                loss='soft_l1', f_scale=3.0, max_nfev=80,
                xtol=1e-7, ftol=1e-7, gtol=1e-7)
            current[active_indices] = result.x
            residual, pairs = evaluate(current, observations)
        rmse = float(np.sqrt(np.mean(residual ** 2))) if residual.size else None
        if len(active_indices) == 0:
            current = np.asarray(result.x, dtype=np.float64)
        q[frame_index] = current
        rank = int(np.linalg.matrix_rank(result.jac)) if len(active_indices) else 0
        observable = rank == len(active_indices)
        has_decoded_dof = active_indices.size > 0 or q_size == 0
        frame_ok = bool(has_decoded_dof and result.success and observable and rmse is not None and
                         rmse <= residual_threshold_px)
        q_valid[frame_index, active_indices] = frame_ok
        decoded_frame_count += int(frame_ok)
        if frame_ok and q_valid[frame_index].all():
            success_count += 1
        residuals.append(rmse if rmse is not None else float('inf'))
        mujoco.mj_forward(model, data)
        poses[frame_index, :, :3, :3] = np.asarray(data.xmat[body_ids]).reshape(-1, 3, 3)
        poses[frame_index, :, :3, 3] = data.xpos[body_ids]
        records.append({'frame_index': frame_index, 'status': 'decoded' if frame_ok else 'review',
                        'visible_observations': visible, 'used_observations': int(len(pairs)),
                        'rmse_px': rmse, 'optimizer_success': bool(result.success),
                        'active_dof_count': int(len(active_indices)),
                        'observable_rank': rank,
                        'optimizer_message': str(result.message)})

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + '.building')
    stage.mkdir(parents=True)
    try:
        np.savez_compressed(stage / 'decoded_motion.npz',
                            timestamps=np.arange(frame_count, dtype=np.float64) / fps,
                            q=q, q_valid=q_valid, T_world_link=poses,
                            joint_order=joint_order, link_order=link_order)
        with (stage / 'decoded_tracks.jsonl').open('w') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        p95 = float(np.percentile(np.asarray(residuals), 95)) if residuals else float('inf')
        all_dofs_selected = len(active_indices) == q_size
        full_success = decoded_frame_count == frame_count and all_dofs_selected
        quality = {
            'schema_version': '1.0',
            'status': ('motion_decode_passed' if full_success else
                       'motion_decode_partial' if decoded_frame_count == frame_count else 'review'),
            'source': 'pseudo',
            'reference_sample': str(reference),
            'view_count': len(views), 'frame_count': frame_count, 'fps': fps,
            'frame_alignment': ('single_view' if len(views) == 1 else
                                'paired_by_frame_index; temporal synchronization assumed, not verified'),
            'successful_frames': decoded_frame_count, 'fully_valid_frames': success_count,
            'success_fraction': decoded_frame_count / frame_count,
            'joint_order': joint_names,
            'active_joints': [joint_names[index] for index in active_indices],
            'active_joint_selection': active_selection,
            'full_dof_coverage': all_dofs_selected,
            'residual_rmse_p95_px': p95, 'residual_threshold_px': residual_threshold_px,
            'available_tasks': ['motion', 'kinematics'] if full_success else [],
            'review_required': not full_success,
            'observability_policy': 'only selected coordinates with full-rank per-frame Jacobians are marked valid',
            'failure_policy': 'unobservable or high-residual frames remain review; q is not promoted to ground truth',
        }
        (stage / 'motion_decode_quality.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2) + '\n')
        stage.rename(output)
        sample = output.parent
        if (sample / 'sample.json').is_file():
            review_path = sample / 'review.json'
            if review_path.exists():
                review = json.loads(review_path.read_text())
                if review.get('status') == 'human_review_complete':
                    quality['source'] = 'human_reference'
                    (output / 'motion_decode_quality.json').write_text(
                        json.dumps(quality, ensure_ascii=False, indent=2) + '\n')
            _refresh_sample_metadata(sample, output, quality)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return quality


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Decode kinematic motion from calibrated part-id masks')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--masks', type=Path, action='append', required=True)
    parser.add_argument('--camera', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label-map', type=Path)
    parser.add_argument('--residual-threshold-px', type=float, default=40.0)
    parser.add_argument('--joint', action='append', dest='active_joints',
                        help='joint to decode; repeat to select multiple (default: infer from reference motion)')
    args = parser.parse_args()
    print(json.dumps(decode_motion(args.reference, args.masks, args.camera, args.output,
                                   args.label_map, args.residual_threshold_px, args.active_joints),
                     ensure_ascii=False, indent=2))
