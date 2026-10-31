"""Build self-contained annotation deliverables with only runtime assets."""

from __future__ import annotations

import hashlib
import json
import shutil
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


REQUIRED_VIDEOS = {
    "rgb_cam01.mp4": "video_cam01.mp4",
    "rgb_cam02.mp4": "video_cam02.mp4",
}


def _asset_id(sample_id: str) -> str:
    return sample_id[:-len("_multiview")] if sample_id.endswith("_multiview") else sample_id


def _mesh_references(urdf: Path) -> list[str]:
    root = ET.parse(urdf).getroot()
    references = []
    for element in root.findall(".//mesh"):
        filename = element.get("filename")
        if not filename:
            raise ValueError(f"URDF mesh element has no filename: {urdf}")
        relative = Path(filename)
        if (relative.is_absolute() or ".." in relative.parts
                or not relative.parts or relative.parts[0] != "meshes"
                or ":" in filename or "\\" in filename):
            raise ValueError(f"URDF mesh path escapes sample: {filename}")
        references.append(filename)
    if not references:
        raise ValueError(f"URDF has no mesh references: {urdf}")
    return sorted(set(references))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_required(source: Path, target: Path, relative: str) -> None:
    path = source / relative
    if not path.resolve().is_relative_to(source.resolve()) or not path.is_file():
        raise ValueError(f"required deliverable is missing: {path}")
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def _validate_obj(path: Path) -> None:
    vertices = faces = 0
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("v "):
            vertices += 1
        elif line.startswith("mtllib "):
            raise ValueError(f"external material dependency is unsupported: {path}")
        elif line.startswith("f "):
            faces += 1
    if vertices == 0 or faces == 0:
        raise ValueError(f"mesh is not a usable OBJ triangle mesh: {path}")


def package_minimal_dataset(manifest_root: Path, cad_root: Path, destination: Path) -> dict:
    """Copy accepted samples into a compact, self-contained runtime dataset."""
    manifest_path = manifest_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    accepted = manifest.get("accepted", [])
    if not accepted or manifest.get("rejected"):
        raise ValueError("source manifest must contain accepted samples and no rejected samples")

    if destination.is_symlink() or (
        destination.exists()
        and (not destination.is_dir() or any(destination.iterdir()))
    ):
        raise ValueError(f"destination must not exist or must be an empty directory: {destination}")
    plans = []
    seen = set()
    for entry in accepted:
        sample_id = str(entry["sample_id"])
        asset_id = _asset_id(sample_id)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", asset_id) or asset_id in seen:
            raise ValueError(f"invalid or duplicate asset ID: {asset_id}")
        seen.add(asset_id)
        source = (manifest_root / str(entry["path"])).resolve()
        if not source.is_relative_to(manifest_root.resolve()):
            raise ValueError(f"source sample escapes manifest root: {source}")
        meshes = _mesh_references(source / "assembly.urdf")
        for relative in [*REQUIRED_VIDEOS, "assembly.urdf", *meshes]:
            file = source / relative
            if (not file.resolve().is_relative_to(source)
                    or not file.is_file() or not file.stat().st_size):
                raise ValueError(f"required deliverable is missing or unsafe: {file}")
        for mesh in meshes:
            if Path(mesh).suffix.lower() == ".obj":
                _validate_obj(source / mesh)
        step = cad_root / asset_id / "assembly.step"
        if (not step.resolve().is_relative_to(cad_root.resolve())
                or not step.is_file() or not step.stat().st_size):
            raise ValueError(f"STEP is missing or unsafe: {step}")
        plans.append((entry, asset_id, source, meshes))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".minimal-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "dataset"
        result = _build_dataset(plans, manifest_path, cad_root, staging)
        if destination.exists():
            try:
                destination.rmdir()
            except OSError as exc:
                raise ValueError(
                    f"destination is no longer empty: {destination}"
                ) from exc
        staging.rename(destination)
    return result


def _build_dataset(plans, manifest_path: Path, cad_root: Path, destination: Path) -> dict:
    destination.mkdir()
    entries = []
    for entry, asset_id, source, mesh_paths in plans:
        sample_id = str(entry["sample_id"])
        target = destination / asset_id
        if target.exists():
            raise ValueError(f"destination sample already exists: {target}")
        target.mkdir(parents=True)

        for source_video, output_video in REQUIRED_VIDEOS.items():
            _copy_required(source, target, source_video)
            (target / source_video).rename(target / output_video)
        _copy_required(source, target, "assembly.urdf")
        _copy_required(cad_root / asset_id, target, "assembly.step")

        for mesh in mesh_paths:
            _copy_required(source, target, mesh)
            if Path(mesh).suffix.lower() == ".obj":
                _validate_obj(target / mesh)

        files = sorted(
            path for path in target.rglob("*") if path.is_file()
        )
        entries.append({
            "asset_id": asset_id,
            "source_sample_id": sample_id,
            "files": [str(path.relative_to(target)) for path in files],
            "mesh_references": mesh_paths,
            "checksums": {
                str(path.relative_to(target)): _sha256(path) for path in files
            },
        })

    result = {
        "schema_version": "2.0-minimal-runtime",
        "source_manifest": "dataset_manifest.json from accepted annotation release",
        "source_cad_root": cad_root.name,
        "sample_count": len(entries),
        "required_per_sample": ["video_cam01.mp4", "video_cam02.mp4", "assembly.urdf", "assembly.step", "meshes referenced by assembly.urdf"],
        "samples": entries,
    }
    (destination / "dataset_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result
