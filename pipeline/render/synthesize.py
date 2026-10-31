"""Synthesize demo motion videos from CAD asset OBJ pairs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def _load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(str(path))
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    return m


def _transform_from_json(t):
    o = t["origin"]
    x, y, z = t["x_axis"], t["y_axis"], t["z_axis"]
    M = np.eye(4)
    M[:3, 0] = [x["x"], x["y"], x["z"]]
    M[:3, 1] = [y["x"], y["y"], y["z"]]
    M[:3, 2] = [z["x"], z["y"], z["z"]]
    M[:3, 3] = [o["x"], o["y"], o["z"]]
    return M


def _full_assembly(asset_dir: Path):
    """Load every body with its occurrence transform and one real joint."""
    meta = json.loads((asset_dir / "assembly.json").read_text())
    body_files = {p.stem: p for p in asset_dir.glob("*.obj") if p.stem != "assembly"}
    parts = []
    body_occ = {}
    for oid, occ in (meta.get("occurrences") or {}).items():
        T = _transform_from_json(occ["transform"])
        for bid in (occ.get("bodies") or {}):
            if bid in body_files:
                mesh = _load_mesh(body_files[bid])
                parts.append((oid, bid, mesh, T))
                body_occ[bid] = oid
    joints = list((meta.get("joints") or {}).values())
    if not parts or not joints:
        return None
    joint = joints[0]
    one, two = joint["occurrence_one"], joint["occurrence_two"]
    geom = joint["geometry_or_origin_two"]
    origin = geom["origin"]
    pivot = np.array([origin["x"], origin["y"], origin["z"]], dtype=float)
    axis = np.array([geom["primary_axis_vector"][k] for k in ("x", "y", "z")], dtype=float)
    axis /= max(np.linalg.norm(axis), 1e-12)
    lim = joint.get("joint_motion", {}).get("rotation_limits", {})
    lo = lim.get("minimum_value", -0.7)
    hi = lim.get("maximum_value", 0.7)
    return parts, one, two, pivot, axis, float(lo), float(hi)


def render_full_joint_video(asset_dir: Path, out_path: Path, frames: int = 60, fps: int = 20):
    loaded = _full_assembly(asset_dir)
    if loaded is None:
        return False
    parts, one, two, pivot, axis, lo, hi = loaded
    all_mesh = trimesh.util.concatenate([m.copy().apply_transform(T) for _, _, m, T in parts])
    center = all_mesh.bounds.mean(axis=0)
    radius = max(float(np.linalg.norm(all_mesh.bounds[1] - all_mesh.bounds[0]) / 2), 1e-6) * 1.15
    fig = plt.figure(figsize=(6, 6), dpi=100)
    writer = imageio.get_writer(str(out_path), fps=fps)
    for i in range(frames):
        q = lo + (hi - lo) * 0.5 * (1 - math.cos(2 * math.pi * i / (frames - 1)))
        R = trimesh.transformations.rotation_matrix(q, axis, pivot)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_axis_off()
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_zlim(center[2]-radius, center[2]+radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=35 + 25 * math.sin(i / frames * 2 * math.pi))
        for oid, _, mesh, T in parts:
            draw = mesh.copy()
            draw.apply_transform(T)
            if oid == two:
                draw.apply_transform(R)
            # Downsample only for display; source geometry remains untouched.
            if len(draw.faces) > 1200:
                draw = draw.submesh([np.linspace(0, len(draw.faces)-1, 1200, dtype=int)], append=True, repair=False)
            ax.plot_trisurf(draw.vertices[:, 0], draw.vertices[:, 1], draw.vertices[:, 2],
                            triangles=draw.faces, color=(0.55, 0.67, 0.82), linewidth=0, antialiased=False)
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        fig.clear()
    writer.close()
    plt.close(fig)
    return True


def _pick_pair(asset_dir: Path):
    """Return two body meshes, preferring ones that contact in assembly.json."""
    body_objs = sorted(p for p in asset_dir.glob("*.obj") if p.stem != "assembly")
    meshes = [(p, _load_mesh(p)) for p in body_objs]
    if len(meshes) < 2:
        return None
    pair = meshes[:2]
    meta_path = asset_dir / "assembly.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for contact in meta.get("contacts") or []:
            b1 = contact.get("entity_one", {}).get("body")
            b2 = contact.get("entity_two", {}).get("body")
            m1 = [m for p, m in meshes if p.stem == b1]
            m2 = [m for p, m in meshes if p.stem == b2]
            if m1 and m2:
                pair = [(body_objs[0], m1[0]), (body_objs[1], m2[0])] if m1[0] is not m2[0] else pair
                break
    return pair


def _setup_axes(meshes):
    all_mesh = trimesh.util.concatenate([m for _, m in meshes])
    center = all_mesh.bounds.mean(axis=0)
    radius = float(np.linalg.norm(all_mesh.bounds[1] - all_mesh.bounds[0]) / 2)
    fig = plt.figure(figsize=(4.8, 4.8), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    lim = max(radius, 1e-6) * 1.5
    ax.set_xlim(center[0] - lim, center[0] + lim)
    ax.set_ylim(center[1] - lim, center[1] + lim)
    ax.set_zlim(center[2] - lim, center[2] + lim)
    return fig, ax, center


def _bbox_radius(meshes):
    all_mesh = trimesh.util.concatenate([m for _, m in meshes])
    return float(np.linalg.norm(all_mesh.bounds[1] - all_mesh.bounds[0]) / 2)


def render_rotation_video(asset_dir: Path, out_path: Path, frames: int = 90, fps: int = 30):
    pair = _pick_pair(asset_dir)
    if pair is None:
        return False
    fig, ax, center = _setup_axes(pair)
    radius = _bbox_radius(pair)
    writer = imageio.get_writer(str(out_path), fps=fps)
    for i in range(frames):
        az = 360.0 * i / frames
        ax.view_init(elev=18, azim=az)
        ax.clear()
        ax.set_axis_off()
        for _, mesh in pair:
            ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                            triangles=mesh.faces, color=(0.55, 0.65, 0.8), edgecolor="none")
        lim = max(radius, 1e-6) * 1.5
        ax.set_xlim(center[0]-lim, center[0]+lim)
        ax.set_ylim(center[1]-lim, center[1]+lim)
        ax.set_zlim(center[2]-lim, center[2]+lim)
        ax.set_box_aspect((1, 1, 1))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        writer.append_data(buf)
    writer.close()
    plt.close(fig)
    return True


def render_motion_video(asset_dir: Path, out_path: Path, frames: int = 60, fps: int = 30):
    """Animate one body sliding along the vector connecting the two body centers."""
    pair = _pick_pair(asset_dir)
    if pair is None:
        return False
    m1, m2 = pair[0][1], pair[1][1]
    c1 = m1.bounds.mean(axis=0)
    c2 = m2.bounds.mean(axis=0)
    direction = c2 - c1
    scale = float(np.linalg.norm(direction))
    if scale < 1e-9:
        direction = np.array([0.0, 0.0, 1.0])
        scale = 1.0
    direction = direction / scale
    amplitude = min(0.3 * scale, 0.5)
    fig, ax, center = _setup_axes(pair)
    radius = _bbox_radius(pair)
    writer = imageio.get_writer(str(out_path), fps=fps)
    for i in range(frames):
        t = 0.5 * (1 - math.cos(2 * math.pi * i / frames))
        offset = direction * amplitude * t
        ax.view_init(elev=20, azim=35 + 8 * math.sin(2 * math.pi * i / frames))
        ax.clear()
        ax.set_axis_off()
        v0 = m1.vertices
        v1 = m2.vertices + offset
        ax.plot_trisurf(v0[:, 0], v0[:, 1], v0[:, 2], triangles=m1.faces,
                        color=(0.6, 0.7, 0.85), edgecolor="none")
        ax.plot_trisurf(v1[:, 0], v1[:, 1], v1[:, 2], triangles=m2.faces,
                        color=(0.85, 0.6, 0.55), edgecolor="none")
        lim = max(radius, 1e-6) * 1.5
        ax.set_xlim(center[0]-lim, center[0]+lim)
        ax.set_ylim(center[1]-lim, center[1]+lim)
        ax.set_zlim(center[2]-lim, center[2]+lim)
        ax.set_box_aspect((1, 1, 1))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        writer.append_data(buf)
    writer.close()
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", required=True)
    ap.add_argument("--out", default="/tmp/vid2artcad_videos")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for asset_path in args.assets:
        d = Path(asset_path)
        name = d.name
        full_ok = render_full_joint_video(d, out / f"{name}_full_joint.mp4")
        if full_ok:
            print(name, "full_joint:", True)
            continue
        r_ok = render_rotation_video(d, out / f"{name}_rotation.mp4")
        m_ok = render_motion_video(d, out / f"{name}_motion.mp4")
        print(name, "rotation:", r_ok, "motion:", m_ok)


if __name__ == "__main__":
    main()
