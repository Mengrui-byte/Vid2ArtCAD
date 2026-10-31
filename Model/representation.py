"""Kinematic supervision loader for URDF/OBJ samples."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

JOINT_TYPES = ("fixed", "revolute", "prismatic")
def _vec(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float32)
    return np.asarray([float(v) for v in text.split()], dtype=np.float32)


def _rpy_matrix(rpy: list[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]], dtype=np.float32)


def _link_bbox(link_node: ET.Element, mesh_root: Path) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for visual in link_node.findall("visual"):
        mesh_node = visual.find("./geometry/mesh")
        if mesh_node is None or not mesh_node.attrib.get("filename"):
            continue
        try:
            mesh = trimesh.load(mesh_root / mesh_node.attrib["filename"], force="mesh", process=False)
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            if not len(vertices):
                continue
            origin = visual.find("origin")
            xyz = np.asarray([float(x) for x in (origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()], dtype=np.float32)
            rpy = [float(x) for x in (origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
            points.append(vertices @ _rpy_matrix(rpy).T + xyz)
        except Exception:
            continue
    if points:
        vertices = np.vstack(points)
        return vertices.min(axis=0), vertices.max(axis=0)
    return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)


def _mesh_bbox(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        if len(vertices):
            return vertices.min(axis=0), vertices.max(axis=0)
    except Exception:
        pass
    return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)


def load_urdf_target(urdf_path: str | Path, max_parts: int = 16) -> dict[str, np.ndarray | list]:
    """Convert an URDF into fixed-size kinematics supervision tensors."""
    urdf_path = Path(urdf_path).resolve()
    root = ET.parse(urdf_path).getroot()
    links = [node.attrib["name"] for node in root.findall("link") if node.attrib.get("name") != "world"]
    links = links[:max_parts]
    index = {name: i for i, name in enumerate(links)}
    n = len(links)

    present = np.zeros(max_parts, dtype=np.float32)
    primitive = np.full(max_parts, 3, dtype=np.int64)
    bbox = np.zeros((max_parts, 6), dtype=np.float32)
    parent = np.full(max_parts, max_parts, dtype=np.int64)
    joint_type = np.zeros(max_parts, dtype=np.int64)
    axis = np.zeros((max_parts, 3), dtype=np.float32)
    origin = np.zeros((max_parts, 3), dtype=np.float32)
    rotation = np.broadcast_to(np.eye(3, dtype=np.float32), (max_parts, 3, 3)).copy()
    limits = np.zeros((max_parts, 2), dtype=np.float32)
    active = np.zeros(max_parts, dtype=np.float32)
    names = [""] * max_parts

    # Geometry and joint origins must share one sample-level spatial scale.
    global_scale = 0.0
    mesh_root = urdf_path.parent
    for i, link_name in enumerate(links):
        present[i] = 1.0
        names[i] = link_name
        link_node = root.find(f"./link[@name='{link_name}']")
        if link_node is not None and link_node.findall("./visual/geometry/mesh"):
            lo, hi = _link_bbox(link_node, mesh_root)
            bbox[i, :3] = (lo + hi) * 0.5
            bbox[i, 3:] = np.maximum(hi - lo, 1e-4)
            global_scale = max(global_scale, float(np.max(np.abs(lo))), float(np.max(np.abs(hi))))
        else:
            bbox[i, 3:] = 1.0

    for joint in root.findall("joint"):
        child_node = joint.find("child")
        parent_node = joint.find("parent")
        if child_node is None or parent_node is None or child_node.attrib.get("link") not in index:
            continue
        child = index[child_node.attrib["link"]]
        parent_name = parent_node.attrib.get("link")
        parent[child] = index.get(parent_name, max_parts)
        kind = joint.attrib.get("type", "fixed")
        if kind == "continuous":
            kind = "revolute"
        joint_type[child] = JOINT_TYPES.index(kind) if kind in JOINT_TYPES else 0
        axis_node = joint.find("axis")
        axis[child] = _vec(axis_node.attrib.get("xyz") if axis_node is not None else None, (0, 0, 1))
        origin_node = joint.find("origin")
        origin[child] = _vec(origin_node.attrib.get("xyz") if origin_node is not None else None)
        rotation[child] = _rpy_matrix(_vec(origin_node.attrib.get("rpy") if origin_node is not None else None))
        limit_node = joint.find("limit")
        if limit_node is not None:
            limits[child] = [float(limit_node.attrib.get("lower", 0.0)), float(limit_node.attrib.get("upper", 0.0))]
        active[child] = float(joint_type[child] != 0)
        global_scale = max(global_scale, float(np.max(np.abs(origin[child]))))

    global_scale = max(global_scale, 1e-6)
    bbox[:, :3] /= global_scale
    bbox[:, 3:] /= global_scale
    axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = axis / np.maximum(axis_norm, 1e-6)
    # Prismatic limits are lengths; revolute limits are angles in radians.
    limits[joint_type == JOINT_TYPES.index("prismatic")] /= global_scale
    return {
        "present": present,
        "primitive": primitive,
        "bbox": bbox,
        "parent": parent,
        "joint_type": joint_type,
        "axis": axis,
        "origin": origin / global_scale,
        "rotation": rotation,
        "limits": limits,
        "active": active,
        "names": names,
        "scale": np.asarray(global_scale, dtype=np.float32),
    }


