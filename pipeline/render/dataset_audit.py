"""Create a source-grounded audit report before annotation rendering."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def audit_dataset(data_root: Path, output: Path | None = None) -> dict:
    assets = []
    joint_totals: Counter[str] = Counter()
    for asset_dir in sorted(data_root.iterdir()):
        assembly_path = asset_dir / 'assembly.json'
        if not asset_dir.is_dir() or not assembly_path.exists():
            continue
        data = json.loads(assembly_path.read_text())
        occurrences = list((data.get('occurrences') or {}).values())
        joints = list((data.get('joints') or {}).values())
        types = Counter(
            str(joint.get('joint_motion', {}).get('joint_type', 'Unknown'))
            for joint in joints
        )
        joint_totals.update(types)
        mesh_files = sorted(path.name for path in asset_dir.glob('*.obj'))
        missing_meshes = ['assembly.obj'] if not (asset_dir / 'assembly.obj').exists() else []
        grounded = sum(bool(item.get('is_grounded')) for item in occurrences)
        assets.append({
            'asset_id': asset_dir.name,
            'occurrence_count': len(occurrences),
            'joint_count': len(joints),
            'grounded_count': grounded,
            'joint_types': dict(sorted(types.items())),
            'missing_mesh_candidates': missing_meshes,
            'obj_file_count': len(mesh_files),
            'status': 'review_required' if missing_meshes else 'candidate',
        })
    report = {
        'schema_version': '1.0',
        'data_root': str(data_root),
        'asset_count': len(assets),
        'joint_type_totals': dict(sorted(joint_totals.items())),
        'assets': assets,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/raw/batch_001'))
    parser.add_argument('--output', type=Path, default=Path('annoation/dataset_audit.json'))
    args = parser.parse_args()
    print(json.dumps(audit_dataset(args.data_root, args.output), ensure_ascii=False, indent=2))
