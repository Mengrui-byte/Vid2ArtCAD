"""Pack one URDF assembly's OBJ geometry into a padded vertex/face target."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


def _rpy_matrix(rpy: list[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]], dtype=np.float32)


def _visual_meshes(link_node: ET.Element, mesh_root: Path):
    result = []
    for visual in link_node.findall("visual"):
        mesh_node = visual.find("./geometry/mesh")
        if mesh_node is None or not mesh_node.attrib.get("filename"):
            continue
        mesh = trimesh.load(mesh_root / mesh_node.attrib["filename"], force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if not len(vertices) or not len(faces):
            continue
        origin = visual.find("origin")
        xyz = np.asarray([float(x) for x in (origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()], dtype=np.float32)
        rpy = [float(x) for x in (origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
        result.append((vertices @ _rpy_matrix(rpy).T + xyz, faces))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-parts", type=int, default=16)
    parser.add_argument("--max-vertices", type=int, default=4096)
    parser.add_argument("--max-faces", type=int, default=8192)
    parser.add_argument("--allow-missing-visual", action="store_true")
    args = parser.parse_args()

    urdf_path = Path(args.urdf).resolve()
    root = ET.parse(urdf_path).getroot()
    links = [n.attrib["name"] for n in root.findall("link") if n.attrib.get("name") != "world"]
    if len(links) > args.max_parts:
        raise ValueError(f"{len(links)} links exceed --max-parts={args.max_parts}")
    vertices = np.zeros((args.max_parts, args.max_vertices, 3), dtype=np.float32)
    vertex_mask = np.zeros((args.max_parts, args.max_vertices), dtype=bool)
    faces = np.zeros((args.max_parts, args.max_faces, 3), dtype=np.int32)
    face_mask = np.zeros((args.max_parts, args.max_faces), dtype=bool)
    vertex_counts = np.zeros(args.max_parts, dtype=np.int32)
    face_counts = np.zeros(args.max_parts, dtype=np.int32)
    scale = 0.0

    for part_idx, link in enumerate(links):
        link_node = root.find(f"./link[@name='{link}']")
        meshes = _visual_meshes(link_node, urdf_path.parent) if link_node is not None else []
        if not meshes:
            if args.allow_missing_visual:
                continue
            raise ValueError(f"link {link} does not have a visual mesh")
        vertices, face_list, offset = [], [], 0
        for visual_vertices, visual_faces in meshes:
            vertices.append(visual_vertices)
            face_list.append(visual_faces + offset)
            offset += len(visual_vertices)
        v, f = np.vstack(vertices), np.vstack(face_list)
        if len(v) > args.max_vertices or len(f) > args.max_faces:
            raise ValueError(f"{link}: {len(v)} vertices/{len(f)} faces exceed configured padding")
        vertices[part_idx, :len(v)] = v
        vertex_mask[part_idx, :len(v)] = True
        vertex_counts[part_idx] = len(v)
        faces[part_idx, :len(f)] = f
        face_mask[part_idx, :len(f)] = True
        face_counts[part_idx] = len(f)
        scale = max(scale, float(np.abs(v).max()))

    # Keep mesh vertices and joint origins in the same normalized spatial frame.
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        if origin is not None:
            scale = max(scale, max(abs(float(x)) for x in origin.attrib.get("xyz", "0 0 0").split()))

    scale = max(scale, 1e-6)
    vertices /= scale
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        vertices=vertices,
        vertex_mask=vertex_mask,
        faces=faces,
        face_mask=face_mask,
        vertex_counts=vertex_counts,
        face_counts=face_counts,
        link_names=np.asarray(links, dtype="U128"),
        scale=np.asarray(scale, dtype=np.float32),
    )
    print(f"saved {args.out}: {len(links)} meshes, max V={args.max_vertices}, max F={args.max_faces}, scale={scale:.6g}")


if __name__ == "__main__":
    main()
