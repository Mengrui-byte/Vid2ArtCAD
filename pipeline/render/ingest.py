"""Register a source asset directory into the immutable raw area."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ingest(source: Path, root: Path, force: bool = False) -> dict:
    candidates = [source] if (source / 'assembly.json').exists() else sorted(
        path for path in source.iterdir()
        if path.is_dir() and (path / 'assembly.json').exists()
    )
    if not candidates:
        raise ValueError(f'no assembly.json assets found under {source}')
    raw_root = root / 'raw'
    raw_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for asset in candidates:
        destination = raw_root / asset.name
        if destination.exists():
            if not force:
                raise FileExistsError(f'raw asset already exists: {destination}')
            shutil.rmtree(destination)
        shutil.copytree(asset, destination)
        files = sorted(path for path in destination.rglob('*') if path.is_file())
        entries.append({
            'asset_id': asset.name,
            'source': str(asset.resolve()),
            'path': str(destination.relative_to(root)),
            'file_count': len(files),
            'sha256': {str(path.relative_to(destination)): file_sha256(path) for path in files},
        })
    report = {'schema_version': '1.0', 'source': str(source.resolve()), 'assets': entries}
    (root / 'ingest_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('data'))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    print(json.dumps(ingest(args.source, args.root, args.force), ensure_ascii=False, indent=2))
