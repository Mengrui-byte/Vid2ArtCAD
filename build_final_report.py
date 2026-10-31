#!/usr/bin/env python3
"""Build the single presentation report without OBJ content."""

from __future__ import annotations

import html
import base64
import mimetypes
import xml.etree.ElementTree as ET
from pathlib import Path

from build_kinematic_report import BATCH, TARGETS, compare, parse_urdf

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "output" / "remote_model" / "model_details.html"


def fmt(value, unit=""):
    return "—" if value is None else f"{value:.4f}{unit}"


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def has_complete_visuals(sample_name: str) -> bool:
    urdf_path = TARGETS / sample_name / "assembly.urdf"
    try:
        root = ET.parse(urdf_path).getroot()
    except (OSError, ET.ParseError):
        return False
    for link in root.findall("link"):
        if link.attrib.get("name") == "world":
            continue
        mesh = link.find("./visual/geometry/mesh")
        filename = mesh.attrib.get("filename") if mesh is not None else None
        if not filename or not (urdf_path.parent / filename).exists():
            return False
    return True


def sample_card(sample: Path) -> str:
    target = parse_urdf(TARGETS / sample.name / "assembly.urdf")
    predicted = parse_urdf(sample / "generated_assembly.urdf")
    result = compare(target, predicted)
    rows = []
    for item in result["rows"]:
        target_joint, predicted_joint = item["target"], item["pred"]
        rows.append(
            f'<tr><td>{html.escape(item["link"])}</td>'
            f'<td class="{"good" if item["parent"] else "bad"}">{html.escape(target_joint["parent"])} → {html.escape(predicted_joint["parent"])}</td>'
            f'<td class="{"good" if item["type"] else "bad"}">{html.escape(target_joint["type"])} → {html.escape(predicted_joint["type"])}</td>'
            f'<td>{fmt(item["axis"], "°")}</td><td>{fmt(item["origin"], " m")}</td><td>{fmt(item["limit"], " m/rad")}</td></tr>'
        )
    image_dir = f"batch_overfit/{sample.name}/screenshots"
    gallery = "".join(
        f'<figure class="image-card"><img class="zoomable" src="{data_url(path)}" alt="{html.escape(path.stem)}"><figcaption>{html.escape(path.stem)}，点击查看大图</figcaption></figure>'
        for path in sorted((sample / "screenshots").glob("*.png"))
    )
    video_a = data_url(TARGETS / sample.name / "rgb_cam01.mp4")
    video_b = data_url(TARGETS / sample.name / "rgb_cam02.mp4")
    return f'''
<section class="panel sample">
  <div class="sample-head"><div><h2>{html.escape(sample.name)}</h2><p class="muted">标注 {result["target_count"]} 个部件，预测 {result["pred_count"]} 个部件，共对齐 {result["common_count"]} 个</p></div>
  <div class="metrics"><b>召回 {result["presence"]*100:.1f}%</b><b>父节点 {result["parent"]*100:.1f}%</b><b>类型 {result["joint_type"]*100:.1f}%</b><b>轴线 {fmt(result["axis_mean"], "°")}</b><b>原点 {fmt(result["origin_mean"], " m")}</b></div></div>
  <h3>输入视频</h3><div class="videos"><figure><video controls preload="metadata" src="{video_a}"></video><figcaption class="muted">视角 01 · cam01</figcaption></figure><figure><video controls preload="metadata" src="{video_b}"></video><figcaption class="muted">视角 02 · cam02</figcaption></figure></div>
  <h3>运动学逐部件对照</h3><div class="table-scroll"><table><thead><tr><th>部件</th><th>父节点：标注 → 生成</th><th>关节：标注 → 生成</th><th>轴线误差</th><th>原点误差</th><th>限位误差</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
  <h3>离线渲染截图</h3><div class="gallery">{gallery}</div>
</section>'''


TEMPLATE = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vid2ArtCAD 模型结构与训练结果汇报</title>
<style>
:root{--bg:#f3f5f7;--panel:#fff;--ink:#17202a;--muted:#66727d;--line:#d8e0e7;--blue:#4e8db7;--green:#287a52;--red:#b04444}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,"PingFang SC","Microsoft YaHei",sans-serif}
header{background:#17202a;color:white;padding:28px max(20px,calc((100vw - 1440px)/2))}header h1{margin:0;font-size:27px}header p{margin:5px 0 0;color:#c5d0d9}
main{max-width:1440px;margin:auto;padding:22px 20px 50px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:16px;margin-bottom:18px}.panel h2{margin:0 0 12px;font-size:19px}.panel h3{font-size:15px;margin:18px 0 8px}.muted{color:var(--muted)}
.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;align-items:stretch}.node{border:1px solid var(--line);border-radius:5px;padding:12px;background:#f8fafb}.node strong{display:block;margin-bottom:5px}.arrow{display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:20px}.code{background:#101820;color:#d9e5ee;border-radius:5px;padding:13px;white-space:pre-wrap;font:13px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace}
.labels{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.label{border-left:3px solid var(--blue);background:#f8fafb;padding:9px}.label b{display:block}.sample-head{display:flex;justify-content:space-between;gap:14px;align-items:flex-start}.metrics{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}.metrics b{border:1px solid var(--line);background:#f8fafb;border-radius:4px;padding:5px 8px;font-size:12px;white-space:nowrap}
.videos{display:grid;grid-template-columns:1fr 1fr;gap:12px}.videos video{display:block;width:100%;aspect-ratio:16/9;background:#101820;border-radius:5px}.videos figure,.image-card{margin:0}.videos figcaption{padding-top:5px}.table-scroll{overflow:auto}table{width:100%;min-width:780px;border-collapse:collapse}th,td{text-align:left;padding:7px 6px;border-bottom:1px solid var(--line);vertical-align:top;white-space:nowrap}th{color:var(--muted);font-weight:600}.good{color:var(--green)}.bad{color:var(--red)}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.image-card{border:1px solid var(--line);border-radius:5px;overflow:hidden;background:#f8fafb}.image-card img{display:block;width:100%;cursor:zoom-in}.image-card figcaption{padding:8px 10px;font-weight:600}.lightbox{position:fixed;inset:0;z-index:20;display:none;align-items:center;justify-content:center;padding:24px;background:#0a1016e8;cursor:zoom-out}.lightbox.open{display:flex}.lightbox img{max-width:96vw;max-height:92vh;width:auto;height:auto;box-shadow:0 8px 40px #0008}.lightbox button{position:fixed;right:18px;top:14px;border:0;background:#fff;color:#17202a;border-radius:4px;font-size:25px;width:38px;height:38px;cursor:pointer}
@media(max-width:900px){.flow{grid-template-columns:1fr}.arrow{transform:rotate(90deg);height:14px}.labels{grid-template-columns:1fr 1fr}.sample-head{display:block}.metrics{justify-content:flex-start;margin-top:10px}}@media(max-width:560px){.labels,.videos{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Vid2ArtCAD 模型结构与训练结果汇报</h1><p>双视角视频输入、运动学恢复、离线渲染结果与逐样本误差对照</p></header><main>
<section class="panel"><h2>模型结构</h2><div class="flow"><div class="node"><strong>双视角视频</strong><span class="muted">cam01 / cam02，16 帧采样</span></div><div class="arrow">→</div><div class="node"><strong>V-JEPA ViT-B</strong><span class="muted">两个冻结的视频编码器</span></div><div class="arrow">→</div><div class="node"><strong>DualViewFusion</strong><span class="muted">视角特征融合与 Transformer</span></div><div class="arrow">→</div><div class="node"><strong>Part slots</strong><span class="muted">最多 16 个部件槽位</span></div><div class="arrow">→</div><div class="node"><strong>输出头</strong><span class="muted">运动学参数与几何顶点</span></div></div><h3>输出含义</h3><p class="muted">运动学头预测部件存在概率、父节点、关节类型、轴线、原点和限位；几何头预测各部件的三维顶点位置。</p></section>
<section class="panel"><h2>优化目标</h2><div class="code">L = L_kinematics + L_geometry + 0.1 * L_jepa

L_kinematics = L_present + L_parent + L_joint_type + L_axis + L_origin + L_limits
L_geometry   = masked mean squared error(predicted_geometry, target_geometry)
L_jepa       = 0  (当前训练入口中未启用 JEPA 预训练目标)</div><p class="muted">当前旧版训练未加入边长、法线、Laplacian、Chamfer 或可微渲染损失。</p></section>
<section class="panel"><h2>监督标签</h2><div class="labels"><div class="label"><b>geometry</b><span>部件三维几何监督</span></div><div class="label"><b>present / parent</b><span>部件存在和父子树关系</span></div><div class="label"><b>joint_type / active</b><span>关节类型和可动性</span></div><div class="label"><b>axis / origin</b><span>关节轴线和原点</span></div><div class="label"><b>limits</b><span>运动范围</span></div><div class="label"><b>恢复评价</b><span>分类准确率和连续参数误差</span></div></div></section>
__SAMPLES__
</main><div class="lightbox" id="lightbox"><button aria-label="关闭">×</button><img alt="放大预览"></div><script>const box=document.querySelector('#lightbox'),big=box.querySelector('img');document.querySelectorAll('.zoomable').forEach(img=>img.addEventListener('click',()=>{big.src=img.src;big.alt=img.alt;box.classList.add('open')}));function closeBox(){box.classList.remove('open');big.removeAttribute('src')}box.addEventListener('click',e=>{if(e.target===box||e.target.tagName==='BUTTON')closeBox()});addEventListener('keydown',e=>{if(e.key==='Escape')closeBox()});</script></body></html>'''


def main() -> None:
    samples = [
        path for path in sorted(BATCH.iterdir())
        if path.is_dir()
        and (path / "generated_assembly.urdf").exists()
        and has_complete_visuals(path.name)
    ]
    REPORT.write_text(TEMPLATE.replace("__SAMPLES__", "\n".join(sample_card(path) for path in samples)), encoding="utf-8")
    print(f"Wrote {REPORT} with {len(samples)} samples")


if __name__ == "__main__":
    main()
