"""Evidence-based audit for the end-to-end annotation acceptance gates."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import mujoco
import numpy as np

from pipeline.render.mujoco_runtime import set_qpos
from pipeline.render.validate_sample import validate_sample


TARGETS = {'p0': 3, 'p1_candidates': 20, 'p1_videos': 4, 'p2_samples': 100,
           'p3_templates': 3, 'p4_real_videos': 20}


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _sample_dirs(output_root: Path) -> list[Path]:
    root = output_root / 'samples'
    return sorted(path for path in root.glob('*_multiview') if path.is_dir()) if root.is_dir() else []


def _real_dirs(output_root: Path) -> list[Path]:
    root = output_root / 'real'
    return sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []


def _views(sample: Path, sample_data: dict) -> dict:
    value = sample_data.get('views')
    if isinstance(value, str):
        value = _read_json(sample / value)
    if not isinstance(value, dict):
        return {}
    cameras = value.get('cameras')
    return cameras if isinstance(cameras, dict) else value


def _view_has_pseudo_labels(sample: Path, sample_data: dict, quality: dict,
                            view: str, spec: dict) -> bool:
    if sample_data.get('views'):
        labels = quality.get('views', {}).get(view, {}).get('labels', {})
    else:
        labels = quality.get('labels', {})
    origin = labels.get('origin') or labels.get('source')
    if origin == 'pseudo':
        return True
    review_name = spec.get('review') or ('review.json' if view == 'cam01' else f'review_{view}.json')
    review = _read_json(sample / str(review_name))
    pseudo_frames = review.get('pseudo_frames', {})
    return isinstance(pseudo_frames, dict) and any(
        isinstance(record, dict)
        and record.get('source') in {'sam2_pseudo', 'optical_flow_pseudo'}
        for record in pseudo_frames.values()
    )


def _family(sample: Path, sample_data: dict) -> str | None:
    if sample_data.get('design_family_id'):
        return str(sample_data['design_family_id'])
    sample_id = str(sample_data.get('sample_id', sample.name))
    return sample_id.removesuffix('_multiview') if sample_id.endswith('_multiview') else sample_id


def _gate(name: str, target: int, actual: int, reasons: list[str], evidence: list[str]) -> dict:
    return {
        'gate': name, 'target': target, 'actual': actual,
        'status': 'passed' if actual >= target and not reasons else 'pending',
        'reasons': reasons, 'evidence': evidence,
    }


def _runtime_replay(sample: Path) -> dict:
    """Independently replay stored states and compare FK and camera calibrations."""
    reasons = []
    try:
        data = _read_json(sample / 'sample.json')
        motion_path = sample / str(data.get('motion') or 'motion.npz')
        scene_path = sample / str(data.get('scene') or 'scene_multiview.xml')
        with np.load(motion_path, allow_pickle=False) as motion:
            q = motion['q']
            poses = motion['T_world_link']
            joint_order = motion['joint_order']
            link_order = motion['link_order']
        if q.ndim != 2 or poses.shape[:2] != (len(q), len(link_order)) or len(q) == 0:
            raise ValueError('motion array shapes are inconsistent')
        if not np.isfinite(q).all() or not np.isfinite(poses).all():
            raise ValueError('motion arrays contain non-finite values')

        runtime_name = data.get('joint_runtime', 'joint_runtime.json')
        runtime = _read_json(sample / str(runtime_name)) if runtime_name else None
        model = mujoco.MjModel.from_xml_path(str(scene_path.resolve()))
        sim = mujoco.MjData(model)
        body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
                    for name in link_order]
        if any(body_id < 0 for body_id in body_ids):
            raise ValueError('motion link missing from MuJoCo scene')
        zero_frames = np.flatnonzero(np.all(np.isclose(q, 0.0, atol=1e-9), axis=1))
        active = np.ptp(q, axis=0) > 1e-9
        perturbations = bool(np.all(
            np.any(q[:, active] < -1e-9, axis=0) & np.any(q[:, active] > 1e-9, axis=0)
        )) if np.any(active) else False
        if not len(zero_frames):
            reasons.append('q_zero_state_missing')
        if not perturbations:
            reasons.append('positive_negative_perturbations_missing')

        pose_error = 0.0
        rotation_error = 0.0
        for index, values in enumerate(q):
            set_qpos(model, sim, values, joint_order, runtime)
            mujoco.mj_forward(model, sim)
            pose_error = max(pose_error, float(np.max(np.abs(sim.xpos[body_ids] - poses[index, :, :3, 3]))))
            rotations = sim.xmat[body_ids].reshape(-1, 3, 3)
            rotation_error = max(rotation_error, float(np.max(np.abs(rotations - poses[index, :, :3, :3]))))
        if pose_error > 1e-7 or rotation_error > 1e-7:
            reasons.append('backend_replay_mismatch')

        extrinsic_error = 0.0
        projection_error = 0.0
        camera_count = 0
        flip = np.diag([1.0, -1.0, -1.0])
        views = _views(sample, data)
        for view in views:
            calibration_path = sample / f'camera_{view}.json'
            if not calibration_path.is_file():
                continue
            calibration = _read_json(calibration_path)
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, view)
            if camera_id < 0 or not calibration.get('T_world_camera'):
                reasons.append(f'camera_calibration_unverifiable:{view}')
                continue
            expected = np.asarray(calibration['T_world_camera'], dtype=float)
            actual = np.eye(4)
            actual[:3, :3] = sim.cam_xmat[camera_id].reshape(3, 3) @ flip
            actual[:3, 3] = sim.cam_xpos[camera_id]
            extrinsic = float(np.max(np.abs(actual - expected)))
            width, height = calibration['resolution_wh']
            intrinsics = np.asarray(calibration['K'], dtype=float)
            fovy = float(calibration['fovy_degrees'])
            implied_fovy = float(np.rad2deg(2.0 * np.arctan((height / 2.0) / intrinsics[1, 1])))
            intrinsic_error = max(
                abs(float(model.cam_fovy[camera_id]) - fovy),
                abs(implied_fovy - fovy),
                abs(float(intrinsics[0, 0] - intrinsics[1, 1])),
                abs(float(intrinsics[0, 2]) - (width - 1) / 2.0),
                abs(float(intrinsics[1, 2]) - (height - 1) / 2.0),
            )
            extrinsic_error = max(extrinsic_error, extrinsic)
            projection_error = max(projection_error, extrinsic, intrinsic_error)
            camera_count += 1
        if camera_count == 0 or projection_error > 1e-7:
            reasons.append('camera_projection_calibration_mismatch')
        return {
            'status': 'passed' if not reasons else 'failed', 'reason_codes': reasons,
            'zero_state_frame_count': int(len(zero_frames)),
            'positive_negative_perturbations': perturbations,
            'pose_error_m': pose_error, 'rotation_error': rotation_error,
            'camera_count': camera_count, 'camera_extrinsic_error': extrinsic_error,
            'camera_projection_error': projection_error,
        }
    except (OSError, ValueError, KeyError, TypeError, IndexError, mujoco.FatalError) as exc:
        return {'status': 'failed', 'reason_codes': ['runtime_replay_error'], 'error': str(exc)}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, relative: object) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path


def _native_template_evidence(data_root: Path, output_root: Path,
                              accepted_samples: list[dict]) -> tuple[list[str], list[dict]]:
    tiers = ('G0', 'G1', 'G2')
    candidates = {}
    sample_template = {}
    evidence = []
    for item in accepted_samples:
        sample = output_root / 'samples' / item['sample_id']
        data = _read_json(sample / 'sample.json')
        descriptor = data.get('native_template')
        if not isinstance(descriptor, dict):
            continue
        template_id = descriptor.get('template_id')
        manifest_path = _inside(sample, descriptor.get('manifest'))
        manifest = _read_json(manifest_path) if manifest_path and manifest_path.is_file() else {}
        source = manifest.get('native_source')
        source_path = _inside(data_root, source.get('path')) if isinstance(source, dict) else None
        tier = data.get('generation_tier')
        valid = (
            isinstance(template_id, str) and bool(template_id.strip()) and
            manifest.get('template_id') == template_id and
            manifest.get('source_format') in {'f3d', 'f3z'} and
            source_path is not None and source_path.is_file() and
            source_path.suffix.lower() == f".{manifest.get('source_format')}" and
            isinstance(source, dict) and source.get('sha256') == _file_sha256(source_path) and
            tier in tiers
        )
        if not valid:
            continue
        template_id = template_id.strip()
        candidates.setdefault(template_id, set()).add(tier)
        sample_template[sample.resolve()] = template_id
        evidence.append({'template_id': template_id, 'sample_id': item['sample_id'],
                         'generation_tier': tier, 'manifest': str(manifest_path)})

    edit_evidence = set()
    edits_root = output_root / 'edits'
    edit_records = sorted(edits_root.rglob('edit.json')) if edits_root.is_dir() else []
    for edit_path in edit_records:
        edit_dir = edit_path.parent
        edit = _read_json(edit_path)
        source_value = edit.get('source_sample')
        if not isinstance(source_value, str) or not source_value:
            continue
        source_path = Path(source_value)
        if not source_path.is_absolute():
            source_path = (Path.cwd() / source_path).resolve()
        template_id = sample_template.get(source_path.resolve())
        if template_id is None:
            continue
        quality = _read_json(edit_dir / 'quality.json')
        if quality.get('status') != 'edit_replay_passed':
            continue
        try:
            validation = validate_sample(edit_dir, check_video=False)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if validation.get('status') != 'accepted':
            continue
        sync_ok = True
        for key in ('joint_sync', 'anchor_sync'):
            check = edit.get(key)
            error = check.get('max_error_m') if isinstance(check, dict) else None
            if (not isinstance(error, (int, float)) or isinstance(error, bool) or
                    not math.isfinite(error) or error > 1e-6 or check.get('status') != 'passed'):
                sync_ok = False
                break
        outputs = edit.get('synchronized_outputs')
        if not isinstance(outputs, list) or not outputs:
            sync_ok = False
        else:
            for relative in outputs:
                path = _inside(edit_dir, relative)
                if path is None or not path.exists():
                    sync_ok = False
                    break
        if sync_ok:
            edit_evidence.add(template_id)
            evidence.append({'template_id': template_id, 'edit_pair': str(edit_path),
                             'joint_sync_error_m': edit['joint_sync']['max_error_m'],
                             'anchor_sync_error_m': edit['anchor_sync']['max_error_m']})

    qualified = sorted(template_id for template_id, found_tiers in candidates.items()
                       if set(tiers).issubset(found_tiers) and template_id in edit_evidence)
    return qualified, evidence


def audit_acceptance(data_root: Path, output_root: Path, report_path: Path | None = None,
                    check_video: bool = False) -> dict:
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    raw_assets = sorted(path for path in data_root.iterdir()
                        if path.is_dir() and (path / 'assembly.json').is_file()) \
        if data_root.is_dir() else []
    samples = _sample_dirs(output_root)
    real_samples = _real_dirs(output_root)

    sample_reports = []
    for sample in samples:
        sample_data = _read_json(sample / 'sample.json')
        validation = validate_sample(sample, check_video=check_video)
        views = _views(sample, sample_data)
        view_count = len(views)
        runtime = _runtime_replay(sample)
        sample_reports.append({
            'sample_id': sample.name,
            'design_family_id': _family(sample, sample_data),
            'view_count': view_count,
            'validation_status': validation.get('status'),
            'validation_reasons': validation.get('reason_codes', []),
            'has_provenance': (sample / 'provenance.json').is_file(),
            'has_quality': (sample / 'quality.json').is_file(),
            'has_available_tasks': bool(_read_json(sample / 'quality.json').get('available_tasks')),
            'runtime_replay': runtime,
        })
    accepted_samples = [item for item in sample_reports if item['validation_status'] == 'accepted']
    four_view_samples = [item for item in accepted_samples if item['view_count'] >= TARGETS['p1_videos']]
    families = sorted({item['design_family_id'] for item in accepted_samples if item['design_family_id']})

    real_reports = []
    for sample in real_samples:
        data = _read_json(sample / 'sample.json')
        quality = _read_json(sample / 'quality.json')
        views = _views(sample, data) or {'cam01': data}
        complete_reviews = 0
        pseudo_label_views = 0
        for view, spec in views.items():
            spec = spec if isinstance(spec, dict) else {}
            review_path = sample / str(
                spec.get('review') or ('review.json' if view == 'cam01' else f'review_{view}.json'))
            if _read_json(review_path).get('status') == 'human_review_complete':
                complete_reviews += 1
            if _view_has_pseudo_labels(sample, data, quality, view, spec):
                pseudo_label_views += 1
        real_reports.append({
            'sample_id': sample.name,
            'view_count': len(views),
            'complete_review_views': complete_reviews,
            'pseudo_label_views': pseudo_label_views,
            'has_pseudo_labels': pseudo_label_views == len(views),
            'validation_status': validate_sample(sample, check_video=check_video).get('status'),
        })
    real_closed = [item for item in real_reports
                   if item['validation_status'] == 'accepted'
                   and item['complete_review_views'] == item['view_count']
                   and item['has_pseudo_labels']]

    template_ids, template_evidence = _native_template_evidence(
        data_root, output_root, accepted_samples)

    runtime_passed = [item for item in accepted_samples
                      if item['runtime_replay']['status'] == 'passed']
    runtime_failed = [item for item in accepted_samples
                      if item['runtime_replay']['status'] != 'passed']
    runtime_families = sorted({item['design_family_id'] for item in runtime_passed
                               if item['design_family_id']})
    p0_reasons = [] if len(runtime_families) >= TARGETS['p0'] else [
        'independently_replayed_mechanism_count_below_target']
    p0_evidence = [str(output_root / 'samples' / item['sample_id']) for item in runtime_passed]
    gates = [
        _gate('P0', TARGETS['p0'], len(runtime_families), p0_reasons, p0_evidence),
        _gate('P1-candidates', TARGETS['p1_candidates'], len(raw_assets),
              [] if len(raw_assets) >= TARGETS['p1_candidates'] else ['fusion_candidate_count_below_target'],
              [str(data_root)]),
        _gate('P1-four-videos', TARGETS['p1_videos'], len(four_view_samples),
              [] if len(four_view_samples) >= TARGETS['p1_videos'] else ['accepted_assets_with_four_views_below_target'],
              [str(output_root / 'samples')]),
        _gate('P2-accepted-assets', TARGETS['p2_samples'], len(accepted_samples),
              [] if len(accepted_samples) >= TARGETS['p2_samples'] else ['accepted_asset_count_below_target'],
              [str(output_root / 'dataset_manifest.json')]),
        _gate('P3-native-templates', TARGETS['p3_templates'], len(template_ids),
              [] if len(template_ids) >= TARGETS['p3_templates'] else ['native_template_evidence_missing'],
              [str(output_root / 'samples')]),
        _gate('P4-real-video-closure', TARGETS['p4_real_videos'], len(real_closed),
              [] if len(real_closed) >= TARGETS['p4_real_videos'] else ['real_video_human_review_count_below_target'],
              [str(output_root / 'real')]),
    ]
    report = {
        'schema_version': '1.0', 'data_root': str(data_root), 'output_root': str(output_root),
        'targets': TARGETS, 'summary': {
            'raw_candidate_count': len(raw_assets), 'sample_count': len(samples),
            'accepted_sample_count': len(accepted_samples), 'four_view_accepted_count': len(four_view_samples),
            'runtime_replay_passed_sample_count': len(runtime_passed),
            'runtime_replay_failed_sample_count': len(runtime_failed),
            'runtime_replay_passed_family_count': len(runtime_families),
            'design_family_count': len(families), 'real_sample_count': len(real_samples),
            'real_closed_loop_count': len(real_closed), 'native_template_count': len(template_ids),
        },
        'gates': gates, 'samples': sample_reports, 'real_samples': real_reports,
        'design_families': families, 'native_templates': sorted(template_ids),
        'native_template_evidence': template_evidence,
        'overall_status': 'passed' if all(gate['status'] == 'passed' for gate in gates) else 'pending',
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Audit Vid2ArtCAD acceptance gates')
    parser.add_argument('--data-root', type=Path, default=Path('data/raw/batch_001'))
    parser.add_argument('--output-root', type=Path, default=Path('annoation'))
    parser.add_argument('--output', type=Path, default=Path('annoation/acceptance_audit.json'))
    parser.add_argument('--check-video', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit_acceptance(args.data_root, args.output_root, args.output, args.check_video),
                     ensure_ascii=False, indent=2))
