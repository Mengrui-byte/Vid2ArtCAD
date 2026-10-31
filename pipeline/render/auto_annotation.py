"""Discover dataset assemblies and export reviewed two-view annotation clips."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline.render.batch_tree_annotation import Assembly, export
from pipeline.render.labels import export_labels
from pipeline.render.multiview_annotation import run as multiview
from pipeline.render.package_dataset import package_dataset
from pipeline.render.file_lock import file_lock
from pipeline.render.stereo_annotation import digest, dump

REFERENCE_FILES = {
    'assembly.urdf', 'motion.npz', 'timestamps.npy', 'part_mapping.json',
    'loop_constraints.json', 'motion_description.json', 'motion_description.txt',
    'quality.json', 'scene.xml', 'camera.json', 'camera.npz', 'rgb.mp4',
}
SAMPLE_FILES = {
    'sample.json', 'quality.json', 'views.json', 'assembly.urdf', 'motion.npz',
    'timestamps.npy', 'scene_multiview.xml', 'rgb_cam01.mp4', 'rgb_cam02.mp4',
    'camera_cam01.json', 'camera_cam01.npz', 'camera_cam02.json', 'camera_cam02.npz',
    'checksums.json',
}
PREVIEW_CODE_FILES = (
    'auto_annotation.py', 'batch_tree_annotation.py', 'verified_lamp.py',
    'multiview_annotation.py', 'labels.py', '../kinematics/source.py',
    '../kinematics/solver.py', '../fusion/adapter.py',
)


def published_sample(path: Path) -> bool:
    if not path.is_dir() or not SAMPLE_FILES.issubset({p.name for p in path.iterdir()}):
        return False
    try:
        quality = json.loads((path / 'quality.json').read_text())
        return quality.get('status') == 'multiview_replay_passed' and all(
            (path / f'rgb_{camera}.mp4').is_file() for camera in quality.get('videos', {}))
    except (OSError, ValueError):
        return False


def complete_sample(path: Path) -> bool:
    if not published_sample(path):
        return False
    labels = path / 'labels' / 'quality.json'
    try:
        report = json.loads(labels.read_text())
        return report.get('status') == 'labels_derived'
    except (OSError, ValueError):
        return False


def backfill_sample_labels(path: Path) -> dict:
    """Upgrade an older published sample without re-rendering its RGB videos."""
    report = export_labels(path)
    sample_path = path / 'sample.json'
    sample = json.loads(sample_path.read_text())
    sample['labels'] = 'labels/quality.json'
    sample['label_report'] = report
    dump(sample_path, sample)
    quality_path = path / 'quality.json'
    quality = json.loads(quality_path.read_text())
    quality.setdefault('available_tasks', ['rgb', 'motion', 'kinematics', 'depth', 'part_id', 'tracks'])
    dump(quality_path, quality)
    dump(path / 'checksums.json', {
        str(item.relative_to(path)): digest(item)
        for item in sorted(path.rglob('*')) if item.is_file() and item.name != 'checksums.json'
    })
    return report


def _template_scene(output_root: Path) -> Path | None:
    samples = sorted(output_root.glob('samples/*_multiview/scene.xml'))
    if samples:
        return samples[0]
    legacy = sorted(path for path in output_root.glob('*/scene.xml')
                    if path.parent.name not in {'previews', 'references', 'samples', 'real'})
    return legacy[0] if legacy else None


def complete_reference(path: Path) -> bool:
    if not path.is_dir() or not REFERENCE_FILES.issubset({p.name for p in path.iterdir()}):
        return False
    try:
        return json.loads((path / 'quality.json').read_text()).get('status') == 'kinematic_preview'
    except (OSError, ValueError):
        return False


def reference_destination(root: Path, asset_id: str) -> Path:
    """Reuse a complete reference or preserve incomplete data under a new version."""
    base = root / asset_id
    if not base.exists() or complete_reference(base):
        return base
    version = 2
    while True:
        candidate = root / f'{asset_id}_v{version}'
        if not candidate.exists() or complete_reference(candidate):
            return candidate
        version += 1


def _preview_code_fingerprint() -> dict[str, str]:
    render_root = Path(__file__).resolve().parent
    return {name: digest(render_root / name) for name in PREVIEW_CODE_FILES}


def _preview_reference_dir(output_root: Path, asset_id: str,
                           preview: dict | None = None) -> Path:
    output = (preview or {}).get('output')
    if output:
        candidate = Path(output)
        if not candidate.is_absolute():
            candidate = candidate if candidate.exists() else output_root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(output_root.resolve())
        except ValueError as exc:
            raise ValueError('preview output must be inside output root') from exc
        return candidate
    return (output_root / 'previews' / 'references' / asset_id).resolve()


def _latest_preview_status(output_root: Path, asset_id: str) -> dict | None:
    try:
        report = json.loads((output_root / 'previews' / 'preview_status.json').read_text())
    except (OSError, ValueError, TypeError):
        return None
    return next((item for item in reversed(report)
                 if item.get('asset') == asset_id and item.get('status') == 'preview_ready'), None)


def _preview_evidence(data_root: Path, output_root: Path, asset_id: str,
                      preview: dict | None = None) -> dict:
    preview_root = _preview_reference_dir(output_root, asset_id, preview)
    required = ('contact_sheet.png', 'motion.npz', 'quality.json')
    if not all((preview_root / name).is_file() for name in required):
        raise FileNotFoundError(f'preview evidence is incomplete for {asset_id}')
    quality = json.loads((preview_root / 'quality.json').read_text())
    if quality.get('status') != 'kinematic_preview' or float(quality.get('mujoco_fk_error', float('inf'))) > 1e-8:
        raise ValueError(f'preview quality gate failed for {asset_id}')
    source_assembly = data_root / asset_id / 'assembly.json'
    if not source_assembly.is_file():
        raise FileNotFoundError(source_assembly)
    return {
        'asset_id': asset_id,
        'preview_output': str(preview_root.resolve().relative_to(output_root.resolve())),
        'source_assembly_sha256': digest(source_assembly),
        'contact_sheet_sha256': digest(preview_root / 'contact_sheet.png'),
        'motion_sha256': digest(preview_root / 'motion.npz'),
        'quality_sha256': digest(preview_root / 'quality.json'),
        'pipeline_sha256': _preview_code_fingerprint(),
    }


def _latest_preview_decision(output_root: Path, asset_id: str) -> dict | None:
    path = output_root / 'previews' / 'review_decisions.json'
    try:
        decisions = json.loads(path.read_text()).get('decisions', [])
    except (OSError, ValueError, TypeError):
        return None
    return next((item for item in reversed(decisions) if item.get('asset_id') == asset_id), None)


def record_preview_decision(data_root: Path, output_root: Path, asset_id: str,
                            decision: str, reviewer: str, notes: str) -> dict:
    if decision not in {'approved', 'rejected'}:
        raise ValueError('decision must be approved or rejected')
    if not reviewer.strip() or not notes.strip():
        raise ValueError('reviewer and notes are required')
    status_path = output_root / 'previews' / 'preview_status.json'
    try:
        report = json.loads(status_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError('preview status is missing or invalid; run preview first') from exc
    preview = next((item for item in reversed(report) if item.get('asset') == asset_id), None)
    if not preview or preview.get('status') != 'preview_ready':
        raise ValueError(f'asset has no successful preview: {asset_id}')
    evidence = _preview_evidence(data_root, output_root, asset_id, preview)
    if preview.get('evidence') != evidence:
        raise ValueError('preview evidence is stale; rerun preview before reviewing')
    entry = {'asset_id': asset_id, 'decision': decision, 'reviewer': reviewer.strip(),
             'notes': notes.strip(), 'evidence': evidence}
    path = output_root / 'previews' / 'review_decisions.json'
    with file_lock(path.with_name(path.name + '.lock')):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError, TypeError):
            data = {'schema_version': '1.0', 'decisions': []}
        data.setdefault('decisions', []).append(entry)
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)
    return entry


def _approved_preview(data_root: Path, output_root: Path, asset_id: str) -> tuple[bool, str]:
    decision = _latest_preview_decision(output_root, asset_id)
    if decision is None:
        return False, 'preview_review_pending'
    if decision.get('decision') != 'approved':
        return False, 'preview_rejected'
    try:
        preview = _latest_preview_status(output_root, asset_id)
        if preview is None:
            return False, 'preview_evidence_missing_or_invalid'
        evidence = _preview_evidence(data_root, output_root, asset_id, preview)
    except (OSError, ValueError, TypeError):
        return False, 'preview_evidence_missing_or_invalid'
    if decision.get('evidence') != evidence:
        return False, 'preview_approval_stale'
    return True, 'preview_approved'


def render_backend_error() -> str | None:
    """Probe the renderer before starting a batch, so failures are actionable."""
    probe = (
        'import mujoco; '
        "m=mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>'); "
        'r=mujoco.Renderer(m,16,16); r.close()'
    )
    result = subprocess.run(
        [sys.executable, '-c', probe], capture_output=True, text=True,
        check=False,
    )
    if result.returncode == 0:
        return None
    lines = (result.stderr or result.stdout).strip().splitlines()
    useful = [line for line in lines if 'CGLError:' in line and 'raise ' not in line]
    if not useful:
        useful = [line for line in lines if 'Error:' in line or 'error' in line.lower()]
    return (useful[0] if useful else (lines[-1] if lines else
            f'render probe exited with code {result.returncode}')).strip()


def atomic_export_reference(asset, destination: Path, template_scene: Path, preview: bool) -> None:
    building = destination.with_name(destination.name + '.building')
    if destination.exists():
        raise FileExistsError(f'Reference output exists; choose a new version: {destination}')
    if building.exists():
        raise FileExistsError(f'Reference staging directory exists; inspect before retrying: {building}')
    try:
        export(asset, building, preview=preview, template_scene=template_scene)
        building.rename(destination)
    except Exception:
        if building.exists():
            shutil.rmtree(building)
        raise


def active_joint_plan(asset_dir: Path, max_active: int, revolute: float, slider: float):
    data = json.loads((asset_dir / 'assembly.json').read_text())
    joints = data.get('joints') or {}
    plan = []
    for joint in joints.values():
        kind = joint.get('joint_motion', {}).get('joint_type')
        if kind == 'RevoluteJointType':
            plan.append((joint['name'], revolute))
        elif kind == 'SliderJointType':
            plan.append((joint['name'], slider))
        elif kind == 'CylindricalJointType':
            plan.append((joint['name'], [revolute, slider]))
        elif kind == 'PinSlotJointType':
            plan.append((joint['name'], [revolute, slider]))
        elif kind == 'BallJointType':
            plan.append((joint['name'], [0.0, 0.0, revolute]))
        elif kind == 'PlanarJointType':
            plan.append((joint['name'], [slider, slider, revolute]))
    return plan[:max_active]


def build_config(asset_dir: Path, revolute: float, slider: float, max_active: int):
    data = json.loads((asset_dir / 'assembly.json').read_text())
    occurrences = list(data['occurrences'].values())
    grounded = [i for i, item in enumerate(occurrences) if item.get('is_grounded')]
    plan = active_joint_plan(asset_dir, max_active, revolute, slider)
    if not plan:
        return None
    root = grounded[0] if grounded else 0
    root_name = occurrences[root]['name']
    joint_names = [name for name, _ in plan]
    return dict(
        id=asset_dir.name,
        asset_root=str(asset_dir.parent.resolve()),
        title=occurrences[root].get('name', asset_dir.name),
        root=root,
        drive=dict(plan),
        description=f'{root_name}保持固定，{ "、".join(joint_names) }依次运动，随后组合运动并复位。')


def run_dataset(data_root: Path, output_root: Path, assets: list[str] | None,
                revolute: float, slider: float, max_active: int, preview: bool,
                strict: bool = False):
    output_root.mkdir(parents=True, exist_ok=True)
    report = []
    review_queue = []
    backend_error = None if preview else render_backend_error()
    if backend_error:
        print(f'RENDER_BACKEND_BLOCKED {backend_error}', flush=True)
    candidates = sorted(path for path in data_root.iterdir()
                        if path.is_dir() and (path / 'assembly.json').exists())
    if assets:
        wanted = set(assets)
        candidates = [path for path in candidates if path.name in wanted]
    for asset_dir in candidates:
        sample_dir = output_root / 'samples' / f'{asset_dir.name}_multiview'
        if published_sample(sample_dir) and not complete_sample(sample_dir):
            try:
                backfill_sample_labels(sample_dir)
                report.append(dict(asset=asset_dir.name, status='labels_backfilled',
                                   output=str(sample_dir)))
            except Exception as exc:
                report.append(dict(asset=asset_dir.name, status='label_backfill_failed',
                                   error=repr(exc), output=str(sample_dir)))
            continue
        if complete_sample(sample_dir):
            report.append(dict(asset=asset_dir.name, status='already_complete',
                               output=str(sample_dir)))
            continue
        if backend_error:
            report.append(dict(asset=asset_dir.name, status='blocked_render_backend', error=backend_error))
            continue
        try:
            cfg = build_config(asset_dir, revolute, slider, max_active)
            if cfg is None:
                report.append(dict(asset=asset_dir.name, status='skipped_no_active_joints'))
                continue
            if not preview:
                approved, approval_reason = _approved_preview(data_root, output_root, asset_dir.name)
                if not approved:
                    status = ('review_required' if approval_reason == 'preview_rejected'
                              else 'awaiting_preview_review')
                    report.append(dict(asset=asset_dir.name, status=status,
                                       reason_code=approval_reason))
                    if status == 'review_required':
                        review_queue.append(dict(asset=asset_dir.name, reason_code=approval_reason,
                                                 source=str((asset_dir / 'assembly.json').as_posix()),
                                                 next_action='review_preview_or_revise_motion'))
                    continue
            reference_root = output_root / ('previews/references' if preview else 'references')
            reference_dir = reference_destination(reference_root, cfg['id'])
            template_scene = _template_scene(output_root)
            if template_scene is None:
                raise FileNotFoundError('No scene.xml template found in an approved sample')
            if not complete_reference(reference_dir):
                atomic_export_reference(Assembly(cfg), reference_dir, template_scene, preview)
            if preview:
                evidence = _preview_evidence(
                    data_root, output_root, asset_dir.name,
                    {'output': str(reference_dir)},
                )
                report.append(dict(asset=asset_dir.name, status='preview_ready',
                                   output=str(reference_dir.resolve()), evidence=evidence))
                continue
            config_path = output_root / f'auto_{cfg["id"]}.json'
            dump(config_path, dict(
                approved_source=str(reference_dir.relative_to(output_root)),
                sample_output=str(sample_dir.relative_to(output_root)),
                cam02_azimuth_offset_degrees=180,
                border_fraction=.04,
                motion_policy='exact_reference_trajectory'))
            multiview(config_path)
            report.append(dict(asset=asset_dir.name, status='completed',
                               output=str(sample_dir)))
        except Exception as exc:
            text = str(exc)
            if 'unresolved joint geometry' in text:
                status, reason = 'review_required', 'unresolved_joint_geometry'
            elif 'disconnected' in text or 'component root' in text:
                status, reason = 'review_required', 'disconnected_kinematic_graph'
            elif "'bodies'" in text or 'Hidden body' in text:
                status, reason = 'review_required', 'unsupported_occurrence_body_layout'
            elif 'Reference geometry mismatch' in text:
                status, reason = 'review_required', 'reference_geometry_mismatch'
            elif 'loop closure failed' in text or 'kinematic loop solve failed' in text:
                status, reason = 'review_required', 'unresolved_loop_closure'
            else:
                status, reason = 'failed', 'unexpected_pipeline_error'
            item = dict(asset=asset_dir.name, status=status, reason_code=reason, error=repr(exc))
            report.append(item)
            if status == 'review_required':
                review_queue.append(dict(asset=asset_dir.name, reason_code=reason,
                                         source=str((asset_dir / 'assembly.json').as_posix()),
                                         error=repr(exc), next_action='manual_adapter_or_source_review'))
            print('FAILED', asset_dir.name, repr(exc), flush=True)
    if preview:
        preview_root = output_root / 'previews'
        preview_root.mkdir(parents=True, exist_ok=True)
        status_path = preview_root / 'preview_status.json'
        with file_lock(status_path.with_name(status_path.name + '.lock')):
            try:
                previous = json.loads(status_path.read_text())
            except (OSError, ValueError, TypeError):
                previous = []
            merged = {item.get('asset'): item for item in previous if item.get('asset')}
            merged.update({item.get('asset'): item for item in report if item.get('asset')})
            dump(status_path, [merged[key] for key in sorted(merged)])
        return report

    manifest = package_dataset(output_root)
    dump(output_root / 'auto_annotation_status.json', report)
    dump(output_root / 'review_queue.json', dict(version=1, items=review_queue))
    if strict and (any(item['status'] in {'failed', 'blocked_render_backend', 'review_required',
                                          'awaiting_preview_review', 'label_backfill_failed'} for item in report)
                   or manifest['rejected']):
        raise SystemExit(1)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/raw/batch_001'))
    parser.add_argument('--output-root', type=Path, default=Path('annoation'))
    parser.add_argument('--asset', action='append')
    parser.add_argument('--revolute-amplitude', type=float, default=.5)
    parser.add_argument('--slider-amplitude', type=float, default=.01)
    parser.add_argument('--max-active-joints', type=int, default=2)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--strict', action='store_true',
                        help='exit non-zero when any asset fails or rendering is blocked')
    args = parser.parse_args()
    run_dataset(args.data_root, args.output_root, args.asset,
                args.revolute_amplitude, args.slider_amplitude,
                args.max_active_joints, args.preview, args.strict)


if __name__ == '__main__':
    main()
