"""Build a deterministic dataset-level manifest from published samples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline.render.validate_sample import validate_sample


def _split(sample_id: str) -> str:
    bucket = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) % 100
    return 'test' if bucket < 10 else 'val' if bucket < 20 else 'train'


def _design_family(sample: Path, sample_json: dict) -> str:
    family = sample_json.get('design_family_id')
    if family:
        return str(family)
    sample_id = str(sample_json.get('sample_id', sample.name))
    suffix = '_multiview'
    return sample_id[:-len(suffix)] if sample_id.endswith(suffix) else sample_id


def _family_splits(families: set[str]) -> dict[str, str]:
    result = {family: _split(family) for family in families}
    if len(families) < 3:
        return result
    for missing in ('test', 'val'):
        if missing not in result.values():
            candidates = sorted(
                (family for family, split in result.items() if split == 'train'),
                key=lambda family: hashlib.sha256(family.encode()).hexdigest(),
            )
            if len(candidates) > 1:
                result[candidates[0]] = missing
    return result


def _sample_candidates(output_root: Path) -> list[Path]:
    candidates = sorted((output_root / 'samples').glob('*_multiview'))
    managed = {'samples', 'references', 'previews', 'real', 'edits', 'artllm_assets',
               'artllm_rendered', 'template_cache'}
    candidates.extend(sorted(
        path for path in output_root.iterdir()
        if path.is_dir() and path.name not in managed
        and (path / 'sample.json').is_file() and (path / 'quality.json').is_file()
    ))
    unique = []
    seen = set()
    for sample in candidates:
        try:
            sample_data = json.loads((sample / 'sample.json').read_text())
        except (OSError, ValueError, TypeError):
            sample_data = {}
        sample_id = str(sample_data.get('sample_id', sample.name))
        if sample_id in seen:
            continue
        seen.add(sample_id)
        unique.append(sample)
    return unique


def _checksums_ok(sample: Path) -> bool:
    checksum_path = sample / 'checksums.json'
    if not checksum_path.exists():
        return False
    try:
        checksums = json.loads(checksum_path.read_text())
        if not isinstance(checksums, dict) or not checksums:
            return False
        actual_files = {
            str(path.relative_to(sample))
            for path in sample.rglob('*')
            if path.is_file() and path.name != 'checksums.json'
        }
        if set(checksums) != actual_files:
            return False
        for relative, expected in checksums.items():
            path = sample / relative
            if not path.resolve().is_relative_to(sample.resolve()) or not path.is_file():
                return False
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                return False
        return True
    except (OSError, ValueError, TypeError):
        return False


def package_dataset(output_root: Path) -> dict:
    entries = []
    rejected = []
    accepted_families = set()
    for sample in _sample_candidates(output_root):
        quality_path = sample / 'quality.json'
        labels_path = sample / 'labels' / 'quality.json'
        sample_json = sample / 'sample.json'
        if not quality_path.exists() or not sample_json.exists():
            continue
        quality = json.loads(quality_path.read_text())
        labels = json.loads(labels_path.read_text()) if labels_path.exists() else None
        if quality.get('status') != 'multiview_replay_passed' or labels is None:
            rejected.append({'sample_id': sample.name, 'reason': 'quality_or_labels_incomplete'})
            continue
        if labels.get('status') != 'labels_derived':
            rejected.append({'sample_id': sample.name, 'reason': 'labels_incomplete'})
            continue
        if not _checksums_ok(sample):
            rejected.append({'sample_id': sample.name, 'reason': 'checksums_invalid'})
            continue
        validation = validate_sample(sample, check_video=True)
        if validation['status'] != 'accepted':
            rejected.append({'sample_id': sample.name, 'reason': 'validation_failed',
                             'details': validation['reason_codes']})
            continue
        videos = quality.get('videos', {})
        camera_reports = labels.get('report', {}).get('cameras', {})
        sample_data = json.loads(sample_json.read_text())
        family = _design_family(sample, sample_data)
        accepted_families.add(family)
        entry = {
            'sample_id': sample.name,
            'design_family_id': family,
            'path': str(sample.relative_to(output_root)),
            'split': None,
            'status': 'accepted',
            'quality': str(quality_path.relative_to(sample)),
            'labels': str(labels_path.relative_to(sample)),
            'videos': videos,
            'label_cameras': camera_reports,
            'validation': validation,
        }
        entries.append(entry)
    real_root = output_root / 'real'
    if real_root.is_dir():
        for sample in sorted(path for path in real_root.iterdir() if path.is_dir()):
            sample_json = sample / 'sample.json'
            quality_path = sample / 'quality.json'
            if not sample_json.is_file() or not quality_path.is_file():
                continue
            sample_data = json.loads(sample_json.read_text())
            quality = json.loads(quality_path.read_text())
            if sample_data.get('mode') != 'real_video_annotation':
                continue
            validation = validate_sample(sample, check_video=True)
            if validation['status'] != 'accepted' or not _checksums_ok(sample):
                rejected.append({'sample_id': sample.name,
                                 'reason': 'real_video_review_or_checksums_invalid',
                                 'details': validation['reason_codes']})
                continue
            family = _design_family(sample, sample_data)
            accepted_families.add(family)
            tasks = ['rgb', 'part_id', 'human_review']
            if sample_data.get('tracks'):
                tasks.append('tracks')
            if sample_data.get('decoded_motion') and quality.get('motion_decode', {}).get('status') == 'motion_decode_passed':
                tasks.extend(['motion', 'kinematics'])
            if sample_data.get('views'):
                videos = quality.get('views', {})
                labels = {name: spec.get('part_id') for name, spec in sample_data['views'].items()}
            else:
                videos = {'cam01': quality.get('video', {})}
                labels = 'labels/part_id_cam01'
            entries.append({
                'sample_id': sample.name,
                'design_family_id': family,
                'path': str(sample.relative_to(output_root)),
                'split': None,
                'status': 'accepted',
                'tasks': tasks,
                'quality': 'quality.json',
                'labels': labels,
                'videos': videos,
                'validation': validation,
            })
    family_splits = _family_splits(accepted_families)
    for entry in entries:
        entry['split'] = family_splits[entry['design_family_id']]
    manifest = {
        'schema_version': '1.0',
        'dataset_root': str(output_root),
        'sample_count': len(entries),
        'split_counts': {
            split: sum(item['split'] == split for item in entries)
            for split in ('train', 'val', 'test')
        },
        'accepted': entries,
        'rejected': rejected,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'dataset_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    )
    with (output_root / 'manifest.jsonl').open('w') as stream:
        for entry in entries:
            stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
    return manifest


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=Path('annoation'))
    args = parser.parse_args()
    print(json.dumps(package_dataset(args.output_root), ensure_ascii=False, indent=2))
