#!/usr/bin/env python3
"""Render overfit OBJ outputs to PNGs and build an image-only HTML report."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parent
BATCH = ROOT / "output" / "remote_model" / "batch_overfit_fixed_scale"
REPORT = ROOT / "output" / "remote_model" / "index.html"


def has_complete_visuals(sample_name: str) -> bool:
    urdf_path = ROOT / "output" / "annoation" / "samples" / sample_name / "assembly.urdf"
    try:
        root = ET.parse(urdf_path).getroot()
    except (OSError, ET.ParseError):
        return False
    for link in root.findall("link"):
        if link.attrib.get("name") == "world":
            continue
        meshes = link.findall("./visual/geometry/mesh")
        if not meshes or any(not mesh.attrib.get("filename") for mesh in meshes):
            return False
        if any(not (urdf_path.parent / mesh.attrib["filename"]).exists() for mesh in meshes):
            return False
    return True


def render_mesh(mesh: trimesh.Trimesh, path: Path, title: str, color: str = "#6e9fc4") -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if len(faces) > 5000:
        faces = faces[np.linspace(0, len(faces) - 1, 5000).astype(np.int64)]
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = (lo + hi) / 2.0
    radius = max(float((hi - lo).max()) / 2.0, 1e-6)
    fig = plt.figure(figsize=(6, 5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(
        vertices[faces], facecolor=color, edgecolor=(0.12, 0.15, 0.18, 0.22), linewidth=0.15, alpha=0.95
    ))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=38)
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, facecolor="#f4f6f8")
    plt.close(fig)


def render_pair(target: trimesh.Trimesh, generated: trimesh.Trimesh, path: Path, title: str) -> None:
    meshes = [(target, "target assembly", "#8a9eae"), (generated, "generated assembly", "#d27c55")]
    vertices = np.vstack([np.asarray(m.vertices, dtype=np.float32) for m, _, _ in meshes])
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = (lo + hi) / 2.0
    radius = max(float((hi - lo).max()) / 2.0, 1e-6)
    fig = plt.figure(figsize=(10, 4.7), dpi=150)
    for index, (mesh, label, color) in enumerate(meshes, 1):
        vertices_m = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if len(faces) > 5000:
            faces = faces[np.linspace(0, len(faces) - 1, 5000).astype(np.int64)]
        ax = fig.add_subplot(1, 2, index, projection="3d")
        ax.add_collection3d(Poly3DCollection(vertices_m[faces], facecolor=color, edgecolor=(.12,.15,.18,.2), linewidth=.15, alpha=.95))
        ax.set_xlim(center[0]-radius, center[0]+radius); ax.set_ylim(center[1]-radius, center[1]+radius); ax.set_zlim(center[2]-radius, center[2]+radius)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=24, azim=38); ax.set_axis_off(); ax.set_title(label)
    fig.suptitle(title); fig.tight_layout(pad=.6)
    fig.savefig(path, facecolor="#f4f6f8"); plt.close(fig)


def _rpy_matrix(rpy: list[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def _transform(origin: list[float], rpy: list[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _rpy_matrix(rpy)
    matrix[:3, 3] = origin
    return matrix


def load_urdf_assembly(urdf_path: Path) -> tuple[trimesh.Trimesh, dict[str, trimesh.Trimesh]]:
    """Place each link mesh in world coordinates using the URDF tree."""
    root = ET.parse(urdf_path).getroot()
    links = {node.attrib["name"]: node for node in root.findall("link") if node.attrib.get("name") != "world"}
    joints = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        origin = joint.find("origin")
        xyz = (origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()
        rpy = (origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()
        joints[child.attrib["link"]] = (parent.attrib["link"], _transform([float(x) for x in xyz], [float(x) for x in rpy]))
    world_cache = {"world": np.eye(4, dtype=np.float64)}

    def world_transform(link: str) -> np.ndarray:
        if link not in world_cache:
            parent, local = joints.get(link, ("world", np.eye(4, dtype=np.float64)))
            world_cache[link] = world_transform(parent) @ local
        return world_cache[link]

    placed = {}
    for name, node in links.items():
        visual_meshes = []
        for visual in node.findall("visual"):
            mesh_node = visual.find("./geometry/mesh")
            if mesh_node is None or not mesh_node.attrib.get("filename"):
                continue
            mesh_path = (urdf_path.parent / mesh_node.attrib["filename"]).resolve()
            if not mesh_path.exists():
                continue
            mesh = trimesh.load(mesh_path, force="mesh", process=False).copy()
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                continue
            origin = visual.find("origin")
            xyz = (origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()
            rpy = (origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()
            mesh.apply_transform(_transform([float(x) for x in xyz], [float(x) for x in rpy]))
            visual_meshes.append(mesh)
        if visual_meshes:
            merged = trimesh.util.concatenate(visual_meshes)
            merged.apply_transform(world_transform(name))
            placed[name] = merged
    if not placed:
        raise ValueError(f"no visual meshes found in {urdf_path}")
    return trimesh.util.concatenate(list(placed.values())), placed


def render_sample(sample: Path) -> list[Path]:
    out = sample / "screenshots"
    out.mkdir(exist_ok=True)
    target_root = ROOT / "output" / "annoation" / "samples" / sample.name
    target_combined, target_parts = load_urdf_assembly(target_root / "assembly.urdf")
    generated_combined, generated_parts = load_urdf_assembly(sample / "generated_assembly.urdf")
    paths = [out / "assembly_comparison.png"]
    render_pair(target_combined, generated_combined, paths[0], f"{sample.name} - FK assembly comparison")
    for name, mesh in sorted(generated_parts.items()):
        path = out / f"{name}.png"
        if name in target_parts:
            render_pair(target_parts[name], mesh, path, f"{name} FK comparison")
        else:
            render_mesh(mesh, path, name, "#d27c55")
        paths.append(path)
    return paths


def image_data(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    samples = []
    for sample in sorted(BATCH.iterdir()):
        if (
            not sample.is_dir()
            or not (sample / "generated_assembly.obj").exists()
            or not has_complete_visuals(sample.name)
        ):
            continue
        paths = render_sample(sample)
        samples.append({"name": sample.name, "images": [(p.stem, image_data(p)) for p in paths]})
    cards = []
    for sample in samples:
        images = "".join(
            f'<figure><img src="{src}" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>'
            for name, src in sample["images"]
        )
        cards.append(f'<article><h2>{html.escape(sample["name"])}</h2><div class="grid">{images}</div></article>')
    REPORT.write_text(TEMPLATE.replace("__CARDS__", "\n".join(cards)), encoding="utf-8")
    print(f"Wrote {REPORT} with {len(samples)} samples")


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Overfit 渲染截图</title>
<style>
:root{--bg:#f4f6f8;--panel:#fff;--ink:#17202a;--muted:#687581;--line:#d9e0e7}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,"PingFang SC","Microsoft YaHei",sans-serif}header{background:#17202a;color:#fff;padding:28px max(20px,calc((100vw - 1440px)/2))}header h1{margin:0;font-size:28px}header p{margin:5px 0 0;color:#bdc8d2}main{max-width:1440px;margin:auto;padding:22px 20px 48px}article{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:16px;margin-bottom:18px}h2{margin:0 0 12px;font-size:18px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}figure{margin:0;border:1px solid var(--line);border-radius:5px;overflow:hidden;background:#f8fafb}img{display:block;width:100%;height:auto;cursor:zoom-in;transition:opacity .15s}img:hover{opacity:.82}figcaption{padding:8px 10px;font-weight:600}.note{color:var(--muted);margin:0 0 18px}.lightbox{position:fixed;inset:0;z-index:20;display:none;align-items:center;justify-content:center;padding:24px;background:rgba(10,16,22,.88);cursor:zoom-out}.lightbox.open{display:flex}.lightbox img{max-width:96vw;max-height:92vh;width:auto;height:auto;object-fit:contain;cursor:zoom-out;box-shadow:0 8px 40px #0008}.lightbox button{position:fixed;right:18px;top:14px;border:0;background:#fff;color:#17202a;border-radius:4px;font-size:25px;width:38px;height:38px;cursor:pointer}@media(max-width:600px){.grid{grid-template-columns:1fr 1fr}}
</style></head><body><header><h1>模型生成结果渲染截图</h1><p>只展示离线渲染后的 PNG，不加载 OBJ、不依赖 CDN；点击任意图片查看大图</p></header><main><p class="note">以下为旧版模型批量 overfit 成功样本及其生成部件截图。运动学恢复对照见 <a href="kinematics_report.html" style="color:#2d6f9f">kinematics_report.html</a>。</p>__CARDS__</main><div class="lightbox" id="lightbox"><button aria-label="关闭">×</button><img alt="放大预览"></div><script>const box=document.querySelector('#lightbox'),big=box.querySelector('img');document.querySelectorAll('main img').forEach(img=>img.addEventListener('click',()=>{big.src=img.src;big.alt=img.alt;box.classList.add('open')}));function closeBox(){box.classList.remove('open');big.removeAttribute('src')}box.addEventListener('click',e=>{if(e.target===box||e.target.tagName==='BUTTON')closeBox()});addEventListener('keydown',e=>{if(e.key==='Escape')closeBox()});</script></body></html>'''


if __name__ == "__main__":
    main()
