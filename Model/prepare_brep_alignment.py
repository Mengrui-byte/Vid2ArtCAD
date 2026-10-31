#!/usr/bin/env python3
"""Build a format-alignment manifest between Vid2ArtCAD samples and Fusion 360 assets.

This deliberately does not invent B-rep latents.  It records the exact STEP/SMT
sources and leaves latent encoding to a matching pretrained B-rep encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_files(root: Path, sample: str) -> dict:
    folder = root / sample
    return {
        "sample": sample,
        "urdf": str(folder / "assembly.urdf"),
        "videos": [str(p) for p in sorted(folder.glob("rgb_cam*.mp4"))],
        "meshes": [str(p) for p in sorted(folder.glob("meshes/*.obj"))],
    }


def part_alignment(sample_root: Path, fusion_root: Path, sample: str) -> list[dict]:
    mapping_path = sample_root / sample / "part_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    fusion_dir = fusion_root / sample.removesuffix("_multiview")
    parts = []
    for item in mapping.get("parts", []):
        body_files = []
        for body_id in item.get("body_ids", []):
            step = fusion_dir / f"{body_id}.step"
            smt = fusion_dir / f"{body_id}.smt"
            obj = fusion_dir / f"{body_id}.obj"
            body_files.append({
                "body_id": body_id,
                "step": str(step) if step.exists() else None,
                "smt": str(smt) if smt.exists() else None,
                "obj": str(obj) if obj.exists() else None,
            })
        parts.append({
            "link": item["link"],
            "fusion_occurrence_id": item.get("occurrence_id"),
            "name": item.get("name"),
            "body_files": body_files,
            "target_coordinate_frame": "link-local; URDF joint transform is applied only at assembly time",
            "latent": {"status": "pending_encoder", "encoder": None, "target_file": None},
        })
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="output/annoation/samples")
    parser.add_argument("--fusion-root", default="a1.0.0_00")
    parser.add_argument("--out", default="output/brep_alignment_manifest.json")
    args = parser.parse_args()

    samples_root = Path(args.samples).resolve()
    fusion_root = Path(args.fusion_root).resolve()
    records = []
    missing = []
    for sample_dir in sorted(p for p in samples_root.iterdir() if p.is_dir()):
        fusion_name = sample_dir.name.removesuffix("_multiview")
        fusion_dir = fusion_root / fusion_name
        source = sample_files(samples_root, sample_dir.name)
        step = fusion_dir / "assembly.step"
        smt = fusion_dir / "assembly.smt"
        assembly_json = fusion_dir / "assembly.json"
        if not fusion_dir.is_dir() or not step.exists() or not smt.exists() or not assembly_json.exists():
            missing.append(sample_dir.name)
            continue
        source["fusion"] = {
            "root": str(fusion_dir),
            "assembly_json": str(assembly_json),
            "assembly_step": str(step),
            "assembly_smt": str(smt),
            "assembly_step_sha256": sha256(step),
            "assembly_smt_sha256": sha256(smt),
        }
        source["parts"] = part_alignment(samples_root, fusion_root, sample_dir.name)
        source["generation_contract"] = {
            "output": "one local STEP/B-rep per predicted link plus generated_assembly.urdf",
            "part_geometry": "local body geometry only; no assembly transform baked into the generated part",
            "assembly": "URDF parent-child joints, origins, axes and limits",
            "assembly_step": "validation reference only, never the primary generation target",
        }
        source["latent"] = {
            "status": "pending_encoder",
            "encoder": None,
            "target_file": None,
            "feature_dim": None,
        }
        records.append(source)

    payload = {
        "version": 1,
        "format_contract": {
            "source_geometry": "Fusion 360 B-rep in STEP/SMT",
            "video_geometry": "Vid2ArtCAD URDF-linked OBJ",
            "coordinate_frame": "sample-local; scale and rigid transform must be recorded by encoder adapter",
            "latent_policy": "never substitute OBJ vertices or bbox values for a B-rep latent",
        },
        "counts": {"matched": len(records), "missing": len(missing)},
        "missing_samples": missing,
        "samples": records,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}: matched={len(records)} missing={len(missing)}")


if __name__ == "__main__":
    main()
