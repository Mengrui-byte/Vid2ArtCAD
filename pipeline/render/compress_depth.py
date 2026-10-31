"""Losslessly migrate per-frame float32 depth NPY files to compressed NPZ."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


DEPTH_CONVENTION = 'camera z-depth in meters; float32 compressed NPZ (key: depth)'


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def _convert_file(path: Path) -> str:
    target = path.with_suffix('.npz')
    temporary = target.with_name(target.name + '.tmp')
    if target.exists():
        with np.load(target, allow_pickle=False) as archive:
            existing = archive['depth']
            if path.exists():
                source = np.load(path, allow_pickle=False)
                if not np.array_equal(source, existing, equal_nan=True):
                    raise ValueError(f'NPY and NPZ depth differ: {path}')
                path.unlink()
                return 'deduplicated'
            if existing.dtype != np.float32:
                raise ValueError(f'compressed depth is not float32: {target}')
        return 'already_compressed'

    source = np.load(path, allow_pickle=False)
    if source.dtype != np.float32:
        raise ValueError(f'depth is not float32: {path}')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1) as archive:
            with archive.open('depth.npy', 'w') as stream:
                np.lib.format.write_array(stream, source, allow_pickle=False)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        with np.load(temporary, allow_pickle=False) as archive:
            converted = archive['depth']
            if converted.dtype != np.float32 or not np.array_equal(source, converted, equal_nan=True):
                raise ValueError(f'compressed depth verification failed: {path}')
        os.replace(temporary, target)
        path.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return 'converted'


def _refresh_sample(sample: Path) -> None:
    label_quality_path = sample / 'labels' / 'quality.json'
    if label_quality_path.is_file():
        report = json.loads(label_quality_path.read_text())
        report['depth_convention'] = DEPTH_CONVENTION
        _write_json(label_quality_path, report)

    checksums_path = sample / 'checksums.json'
    if checksums_path.is_file():
        checksums = {
            path.relative_to(sample).as_posix(): _digest(path)
            for path in sorted(sample.rglob('*'))
            if path.is_file() and path != checksums_path
        }
        _write_json(checksums_path, checksums)


def compress_depth(root: Path, workers: int = 4) -> dict:
    root = root.resolve()
    files = sorted(root.rglob('depth_*/*.npy'))
    samples = {path.parents[2] for path in files}
    samples.update(path.parent for path in root.rglob('sample.json'))
    counts = {'converted': 0, 'deduplicated': 0, 'already_compressed': 0}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_convert_file, path) for path in files]
        for index, future in enumerate(as_completed(futures), 1):
            counts[future.result()] += 1
            if index % 1000 == 0:
                print(f'converted {index}/{len(files)} depth frames', flush=True)
    for sample in sorted(samples):
        _refresh_sample(sample)
    return {'root': str(root), 'sample_count': len(samples), 'depth_files': counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='sample root, for example output/annoation/samples')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(compress_depth(args.root, args.workers), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
