#!/usr/bin/env python3
"""Render ArtLLM GLB assemblies to lightweight PNG previews and MP4 turntables."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import trimesh
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
ASSET_ROOT = ROOT / "annoation" / "artllm_assets"
OUT_ROOT = ROOT / "annoation" / "artllm_rendered"
WIDTH, HEIGHT = 720, 480
FPS, FRAMES = 16, 48


def transform_origin(node):
    origin = node.find("origin")
    if origin is None:
        return np.eye(4)
    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    transform = np.eye(4)
    transform[:3, 3] = xyz
    rx, ry, rz = rpy
    for axis, angle in enumerate((rx, ry, rz)):
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.eye(4)
        if axis == 0: rotation[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
        if axis == 1: rotation[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        if axis == 2: rotation[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        transform = transform @ rotation
    return transform


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.eye(4)
    x, y, z = axis / norm
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    result = np.eye(4)
    result[:3, :3] = [[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x*y+s*z, t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]]
    return result


def load_urdf(path: Path):
    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall("link"):
        visual = link.find("visual")
        mesh_node = visual.find("geometry/mesh") if visual is not None else None
        if mesh_node is None:
            continue
        mesh_path = path.parent / mesh_node.get("filename")
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        faces = np.asarray(mesh.faces)
        limit = 6000
        if len(faces) > limit:
            faces = faces[np.linspace(0, len(faces) - 1, limit, dtype=np.int64)]
        links[link.get("name")] = (np.asarray(mesh.vertices), faces, transform_origin(visual))
    joints = {}
    children = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        axis = np.fromstring((joint.find("axis").get("xyz") if joint.find("axis") is not None else "0 0 1"), sep=" ")
        joints[child] = (parent, transform_origin(joint), axis, joint.get("type", "fixed"))
        children.add(child)
    roots = [name for name in links if name not in children]
    root_link = roots[0] if roots else next(iter(links))
    return links, joints, root_link


def urdf_parts(links, joints, root_link, phase):
    frames = {}
    def visit(link, transform):
        frames[link] = transform
        for child, (parent, origin, axis, joint_type) in joints.items():
            if parent != link:
                continue
            motion = axis_rotation(axis, 0.38 * math.sin(phase)) if joint_type == "revolute" else np.eye(4)
            visit(child, transform @ origin @ motion)
    visit(root_link, np.eye(4))
    parts = []
    for index, (name, (vertices, faces, visual)) in enumerate(links.items()):
        transform = frames.get(name, np.eye(4)) @ visual
        parts.append((trimesh.transform_points(vertices, transform), faces, index))
    return parts


def project(parts, angle):
    all_vertices = np.vstack([vertices for vertices, _, _ in parts])
    all_vertices = all_vertices[np.isfinite(all_vertices).all(axis=1)]
    center = (all_vertices.min(axis=0) + all_vertices.max(axis=0)) / 2
    extent = np.max(all_vertices.max(axis=0) - all_vertices.min(axis=0)) or 1.0
    yaw = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
    projected = []
    for vertices, faces, index in parts:
        points = (vertices - center) @ yaw.T
        if not np.isfinite(points).all():
            valid = np.isfinite(points).all(axis=1)
            points = np.where(valid[:, None], points, 0)
        points[:, 1] *= -1
        xy = points[:, [0, 1]] / extent * min(WIDTH, HEIGHT) * 0.76
        xy += np.array([WIDTH / 2, HEIGHT / 2])
        depth = points[:, 2]
        face_depth = depth[faces].mean(axis=1)
        order = np.argsort(face_depth)
        projected.extend((float(face_depth[i]), xy[faces[i]], index, faces[i]) for i in order)
    return projected


def render(parts, angle):
    image = Image.new("RGB", (WIDTH, HEIGHT), (244, 247, 249))
    draw = ImageDraw.Draw(image)
    draw.line((0, HEIGHT - 54, WIDTH, HEIGHT - 54), fill=(221, 227, 232), width=1)
    colors = [(55, 126, 184), (231, 138, 62), (66, 157, 112), (175, 100, 176), (205, 83, 77), (90, 110, 130)]
    triangles = sorted(project(parts, angle), key=lambda item: item[0])
    for depth, polygon, index, face in triangles:
        shade = max(0.63, min(1.12, 0.88 + depth * 0.002))
        base = colors[index % len(colors)]
        color = tuple(max(0, min(255, int(value * shade))) for value in base)
        draw.polygon([tuple(map(float, point)) for point in polygon], fill=color)
    return np.asarray(image)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for sample in ("bucket", "box", "partnet_40147"):
        source = ASSET_ROOT / sample / "urdf" / f"{sample}.urdf"
        if not source.exists():
            continue
        output = OUT_ROOT / sample
        output.mkdir(parents=True, exist_ok=True)
        links, joints, root_link = load_urdf(source)
        writer = imageio.get_writer(output / "turntable.mp4", fps=FPS, codec="libx264", quality=7, macro_block_size=1)
        for frame in range(FRAMES):
            phase = frame / FRAMES * math.tau
            # Keep the camera fixed; only the URDF joint state changes between frames.
            image = render(urdf_parts(links, joints, root_link, phase), 0.0)
            if frame == 0:
                Image.fromarray(image).save(output / "preview.png", optimize=True)
            writer.append_data(image)
        writer.close()
        print(sample, "links=", len(links), "joints=", len(joints), "frames=", FRAMES)


if __name__ == "__main__":
    main()
