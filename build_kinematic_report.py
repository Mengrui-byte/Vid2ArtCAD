#!/usr/bin/env python3
"""Build an offline HTML report comparing target and generated URDF kinematics."""

from __future__ import annotations

import html
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BATCH = ROOT / "output" / "remote_model" / "batch_overfit_fixed_scale"
TARGETS = ROOT / "output" / "annoation" / "samples"
REPORT = ROOT / "output" / "remote_model" / "kinematics_report.html"


def vector(text: str | None, default: tuple[float, ...]) -> list[float]:
    if not text:
        return list(default)
    return [float(x) for x in text.split()]


def parse_urdf(path: Path) -> dict:
    root = ET.parse(path).getroot()
    links = {node.attrib["name"] for node in root.findall("link") if node.attrib["name"] != "world"}
    joints = {}
    for node in root.findall("joint"):
        child = node.find("child")
        parent = node.find("parent")
        if child is None or parent is None:
            continue
        child_name = child.attrib["link"]
        origin = node.find("origin")
        axis = node.find("axis")
        limit = node.find("limit")
        joints[child_name] = {
            "name": node.attrib.get("name", ""),
            "type": node.attrib.get("type", "unknown"),
            "parent": parent.attrib["link"],
            "origin": vector(origin.attrib.get("xyz") if origin is not None else None, (0, 0, 0)),
            "rpy": vector(origin.attrib.get("rpy") if origin is not None else None, (0, 0, 0)),
            "axis": vector(axis.attrib.get("xyz") if axis is not None else None, (0, 0, 0)),
            "lower": float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
            "upper": float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
        }
    return {"links": sorted(links), "joints": joints}


def canonical_type(kind: str) -> str:
    # The current model has no continuous class; compare continuous against revolute.
    return "revolute" if kind == "continuous" else kind


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def axis_error(a: list[float], b: list[float]) -> float | None:
    na, nb = norm(a), norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    dot = abs(sum(x * y for x, y in zip(a, b)) / (na * nb))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compare(target: dict, pred: dict) -> dict:
    target_links, pred_links = set(target["links"]), set(pred["links"])
    common = sorted(target_links & pred_links)
    rows = []
    parent_ok = type_ok = 0
    axis_values, origin_values, limit_values = [], [], []
    for link in common:
        t, p = target["joints"].get(link), pred["joints"].get(link)
        if not t or not p:
            continue
        parent_match = t["parent"] == p["parent"]
        type_match = canonical_type(t["type"]) == canonical_type(p["type"])
        if parent_match:
            parent_ok += 1
        if type_match:
            type_ok += 1
        ae = axis_error(t["axis"], p["axis"]) if canonical_type(t["type"]) != "fixed" else None
        oe = distance(t["origin"], p["origin"])
        le = None
        if t["lower"] is not None and p["lower"] is not None and t["upper"] is not None and p["upper"] is not None:
            le = (abs(t["lower"] - p["lower"]) + abs(t["upper"] - p["upper"])) / 2
        if ae is not None:
            axis_values.append(ae)
        origin_values.append(oe)
        if le is not None:
            limit_values.append(le)
        rows.append({"link": link, "target": t, "pred": p, "parent": parent_match, "type": type_match, "axis": ae, "origin": oe, "limit": le})
    return {
        "target_count": len(target_links),
        "pred_count": len(pred_links),
        "common_count": len(common),
        "presence": len(common) / max(len(target_links), 1),
        "parent": parent_ok / max(len(rows), 1),
        "joint_type": type_ok / max(len(rows), 1),
        "axis_mean": sum(axis_values) / len(axis_values) if axis_values else None,
        "origin_mean": sum(origin_values) / len(origin_values) if origin_values else None,
        "limit_mean": sum(limit_values) / len(limit_values) if limit_values else None,
        "rows": rows,
    }


def fmt(value: float | None, unit: str = "") -> str:
    return "—" if value is None else f"{value:.4f}{unit}"


def esc(value: object) -> str:
    return html.escape(str(value))


def render_tree(kin: dict, label: str) -> str:
    children: dict[str, list[str]] = {}
    for link in kin["links"]:
        joint = kin["joints"].get(link, {"parent": "world"})
        children.setdefault(joint["parent"], []).append(link)

    def branch(parent: str, depth: int = 0) -> str:
        out = []
        for child in sorted(children.get(parent, [])):
            joint = kin["joints"].get(child, {})
            out.append(f'<div class="tree-node" style="--depth:{depth}"><b>{esc(child)}</b><span>{esc(joint.get("type", "?"))}</span></div>')
            out.append(branch(child, depth + 1))
        return "".join(out)

    return f'<div class="tree"><strong>{esc(label)}</strong>{branch("world")}</div>'


def sample_card(sample: Path) -> str:
    target_path = TARGETS / sample.name / "assembly.urdf"
    pred_path = sample / "generated_assembly.urdf"
    target, pred = parse_urdf(target_path), parse_urdf(pred_path)
    result = compare(target, pred)
    metrics = [
        ("部件召回", f'{result["presence"] * 100:.1f}%'),
        ("父节点准确率", f'{result["parent"] * 100:.1f}%'),
        ("关节类型准确率", f'{result["joint_type"] * 100:.1f}%'),
        ("轴线平均误差", fmt(result["axis_mean"], "°")),
        ("原点平均误差", fmt(result["origin_mean"], " m")),
        ("限位平均误差", fmt(result["limit_mean"], " m/rad")),
    ]
    metric_html = "".join(f'<div class="metric"><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in metrics)
    rows = []
    for row in result["rows"]:
        t, p = row["target"], row["pred"]
        rows.append(
            "<tr>"
            f'<td><b>{esc(row["link"])}</b></td>'
            f'<td class="{"good" if row["parent"] else "bad"}">{esc(t["parent"])} → {esc(p["parent"])}</td>'
            f'<td class="{"good" if row["type"] else "bad"}">{esc(t["type"])} → {esc(p["type"])}</td>'
            f'<td>{fmt(row["axis"], "°")}</td><td>{fmt(row["origin"], " m")}</td><td>{fmt(row["limit"], " m/rad")}</td>'
            "</tr>"
        )
    image = f"batch_overfit/{sample.name}/screenshots/assembly_comparison.png"
    return f'''<article class="sample"><div class="sample-head"><div><h2>{esc(sample.name)}</h2><p>标注 {result["target_count"]} 个部件，预测 {result["pred_count"]} 个部件，共对齐 {result["common_count"]} 个</p></div><a href="index.html">返回渲染截图</a></div><div class="metrics">{metric_html}</div><div class="trees">{render_tree(target, "标注运动学树")}{render_tree(pred, "生成运动学树")}</div><details open><summary>逐部件运动学误差</summary><div class="table-wrap"><table><thead><tr><th>部件</th><th>父节点</th><th>关节类型</th><th>轴线误差</th><th>原点误差</th><th>限位误差</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></details><img class="comparison" src="{image}" alt="{esc(sample.name)} 目标与生成装配对照" loading="lazy"></article>'''


MODEL_SECTION = '<section class="sample"><h2>模型结构</h2><p><b>双视角视频</b> → <b>V-JEPA ViT-B</b> → <b>DualViewFusion</b> → <b>Part slots</b> → <b>运动学头 + OBJ 顶点头</b>。运动学头预测部件存在概率、父节点、关节类型、轴线、原点和限位；几何头在固定三角面拓扑下预测每个顶点的 XYZ。</p></section><section class="sample"><h2>优化目标</h2><pre style="background:#101820;color:#d9e5ee;border-radius:5px;padding:13px;white-space:pre-wrap;font:13px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace">L = L_kinematics + L_vertex + 0.1 * L_jepa\n\nL_kinematics = L_present + L_parent + L_joint_type + L_axis + L_origin + L_limits\nL_vertex      = masked mean squared error(predicted_vertices, target_vertices)\nL_jepa        = 0  (当前训练入口中未启用 JEPA 预训练目标)</pre><p>当前旧版训练没有边长、法线、Laplacian、Chamfer 或可微渲染损失；运动学指标和几何截图需要分别查看。</p></section><section class="sample"><h2>监督标签</h2><p><b>vertices / faces</b>：OBJ 几何；<b>present / parent</b>：URDF 部件存在和父子树；<b>joint_type / active</b>：关节类别和可动性；<b>axis / origin / limits</b>：关节空间参数。</p></section>'

TEMPLATE = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>运动学恢复结果</title><style>
:root{{--bg:#f4f6f8;--panel:#fff;--ink:#17202a;--muted:#687581;--line:#d9e0e7;--good:#287a52;--bad:#b04444}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,"PingFang SC","Microsoft YaHei",sans-serif}}header{{background:#17202a;color:white;padding:28px max(20px,calc((100vw - 1440px)/2))}}header h1{{margin:0;font-size:28px}}header p{{margin:6px 0 0;color:#bdc8d2}}main{{max-width:1440px;margin:auto;padding:22px 20px 48px}}.sample{{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:16px;margin-bottom:18px}}.sample-head{{display:flex;justify-content:space-between;gap:15px;align-items:flex-start}}h2{{margin:0 0 4px;font-size:19px}}p{{color:var(--muted);margin:0 0 14px}}a{{color:#2d6f9f}}.metrics{{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:12px 0 18px}}.metric{{border:1px solid var(--line);background:#f8fafb;border-radius:5px;padding:10px}}.metric span,.metric b{{display:block}}.metric span{{color:var(--muted);font-size:12px}}.metric b{{font-size:18px;margin-top:3px}}.trees{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}}.tree{{border:1px solid var(--line);border-radius:5px;padding:12px;background:#fbfcfd}.tree-node{{margin-left:calc(var(--depth) * 20px);border-left:2px solid #b9c7d1;padding:4px 8px;margin-top:5px}}.tree-node span{{color:var(--muted);margin-left:10px;font-size:12px}}details{{border-top:1px solid var(--line);padding-top:12px}}summary{{cursor:pointer;font-weight:600;margin-bottom:10px}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:720px}}th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);white-space:nowrap}}th{{color:var(--muted);font-weight:600}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}.comparison{{display:block;width:min(100%,900px);margin:18px auto 0;border:1px solid var(--line);border-radius:5px;cursor:zoom-in}}.lightbox{{position:fixed;inset:0;z-index:20;display:none;align-items:center;justify-content:center;padding:24px;background:rgba(10,16,22,.88);cursor:zoom-out}}.lightbox.open{{display:flex}}.lightbox img{{max-width:96vw;max-height:92vh;width:auto;height:auto;cursor:zoom-out;box-shadow:0 8px 40px #0008}}.lightbox button{{position:fixed;right:18px;top:14px;border:0;background:#fff;color:#17202a;border-radius:4px;font-size:25px;width:38px;height:38px;cursor:pointer}}@media(max-width:900px){{.metrics{{grid-template-columns:repeat(3,1fr)}}.trees{{grid-template-columns:1fr}}}}@media(max-width:560px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.sample-head{{display:block}}}}
</style></head><body><header><h1>运动学恢复结果</h1><p>逐样本比较标注 URDF 与生成 URDF。父节点、关节类型为分类准确率；轴线、原点、限位为参数误差。</p></header><main>__CARDS__</main><div class="lightbox" id="lightbox"><button aria-label="关闭">×</button><img alt="放大预览"></div><script>const box=document.querySelector('#lightbox'),big=box.querySelector('img');document.querySelectorAll('.comparison').forEach(img=>img.addEventListener('click',()=>{{big.src=img.src;big.alt=img.alt;box.classList.add('open')}}));function closeBox(){{box.classList.remove('open');big.removeAttribute('src')}}box.addEventListener('click',e=>{{if(e.target===box||e.target.tagName==='BUTTON')closeBox()}});addEventListener('keydown',e=>{{if(e.key==='Escape')closeBox()}});</script></body></html>'''


def main() -> None:
    cards = []
    for sample in sorted(BATCH.iterdir()):
        if sample.is_dir() and (sample / "generated_assembly.urdf").exists() and (TARGETS / sample.name / "assembly.urdf").exists():
            cards.append(sample_card(sample))
    REPORT.write_text(TEMPLATE.replace("__CARDS__", MODEL_SECTION + "\n" + "\n".join(cards)), encoding="utf-8")
    print(f"Wrote {REPORT} with {len(cards)} samples")


if __name__ == "__main__":
    main()
