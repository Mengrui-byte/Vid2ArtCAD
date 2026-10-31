#!/usr/bin/env python3
"""Build a single-file visual-only report from rendered annotation media."""

from __future__ import annotations

import base64
import html
import json
import mimetypes
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANNOTATION_ROOT = ROOT / "annoation"
OUTPUT = ANNOTATION_ROOT / "index.html"


def b64(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def collect() -> list[dict]:
    result = []
    for folder in sorted(ANNOTATION_ROOT.iterdir()):
        if not folder.is_dir() or not (folder / "sample.json").exists():
            continue
        description = read_json(folder / "motion_description.json")
        result.append({
            "id": folder.name,
            "title": description.get("title_cn", folder.name),
            "description": description.get("description_cn", ""),
            "videos": [b64(folder / "rgb_cam01.mp4"), b64(folder / "rgb_cam02.mp4")],
        })
    return result


def collect_artllm_rendered() -> list[dict]:
    result = []
    root = ANNOTATION_ROOT / "artllm_rendered"
    for folder in sorted(root.iterdir()) if root.exists() else []:
        preview, video = folder / "preview.png", folder / "turntable.mp4"
        if preview.exists() and video.exists():
            result.append({"id": folder.name, "preview": b64(preview), "video": b64(video)})
    return result


def main() -> None:
    data = json.dumps({"samples": collect(), "artllm": collect_artllm_rendered()}, ensure_ascii=False, separators=(",", ":"))
    OUTPUT.write_text(TEMPLATE.replace("__DATA__", data), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024 / 1024:.1f} MB)")


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>1020组会</title>
<style>
:root{--ink:#17202a;--muted:#66717d;--line:#d9e0e7;--paper:#f5f7f9;--panel:#fff;--accent:#1769aa}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,"PingFang SC","Microsoft YaHei",sans-serif}
header{background:#17202a;color:#fff;padding:30px max(22px,calc((100vw - 1380px)/2))}header h1{margin:0;font-size:28px}header p{margin:4px 0 0;color:#bdc8d2}
main{max-width:1380px;margin:auto;padding:24px 20px 50px}.idea{padding:18px 20px;background:#fff;border-left:4px solid var(--accent);border-radius:5px;margin-bottom:24px}.idea h2{margin:0 0 5px;font-size:19px}.idea p{margin:0;color:#40505f}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:18px}input{flex:1;min-width:240px;border:1px solid var(--line);border-radius:5px;padding:10px 12px;font:inherit}.count{color:var(--muted);margin-left:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(600px,1fr));gap:18px}.sample{background:var(--panel);border:1px solid var(--line);border-radius:7px;overflow:hidden}.sample-head{padding:17px 18px 11px;border-bottom:1px solid var(--line)}h2{margin:0;font-size:19px}.id{font-size:12px;color:var(--muted)}.description{padding:12px 18px 0;margin:0;color:#40505f}
.videos{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:15px 18px 17px}figure{margin:0}video{display:block;width:100%;aspect-ratio:16/9;background:#101820;border-radius:4px}figcaption{font-size:12px;color:var(--muted);padding-top:5px}
.artllm-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}.artllm-card{background:var(--panel);border:1px solid var(--line);border-radius:7px;overflow:hidden}.artllm-card img{display:block;width:calc(100% - 36px);margin:15px 18px 10px;border:1px solid var(--line);border-radius:4px}.artllm-card video{margin:0 18px;width:calc(100% - 36px)}.artllm-card figcaption{padding:5px 18px 17px}
.empty{padding:35px;color:var(--muted);text-align:center}@media(max-width:700px){.grid{grid-template-columns:1fr}.videos{grid-template-columns:1fr}.frames{grid-template-columns:repeat(3,1fr)}.count{width:100%;margin-left:0}}
</style>
</head>
<body>
<header><h1>1020组会</h1><p>标注数据集与三维运动学结构重建</p></header>
<main>
<section class="idea"><h2>我的 Idea</h2><p>数据集输入包含视频、相机参数和运动的自然语言描述；以运动学特征作为辅助监督标签，使用 V-JEPA 作为视频编码器识别部件特征与运动特征，最终重建并输出 BRep 和 URDF。模型采用 V-JEPA 视频编码器加文本编码器和相机编码器的多模态骨干，先经融合 Transformer 预测部件中心、部件框、关节类型、轴线和关节范围等运动学特征，再由条件 BRep 生成头与运动学预测头分别输出 BRep 和 URDF。</p></section>
<div class="toolbar"><input id="search" type="search" placeholder="搜索样本名称或编号"><span id="count" class="count"></span></div>
<section id="grid" class="grid"></section>
<h2 class="section-title">ArtLLM 三维结构渲染</h2><p class="section-note">三维预测结构已离线渲染为旋转视频和预览图。</p><section id="artllm" class="artllm-grid"></section>
</main>
<script>
const report=__DATA__,samples=report.samples,grid=document.querySelector('#grid'),search=document.querySelector('#search'),count=document.querySelector('#count');
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){const q=search.value.trim().toLowerCase();const items=samples.filter(x=>[x.id,x.title,x.description].join(' ').toLowerCase().includes(q));count.textContent=`${items.length} / ${samples.length} 个样本`;grid.innerHTML=items.map(x=>`<article class="sample"><div class="sample-head"><h2>${esc(x.title)}</h2><div class="id">${esc(x.id)}</div></div><p class="description">${esc(x.description)}</p><div class="videos"><figure><video controls preload="none" src="${x.videos[0]}"></video><figcaption>标注渲染视频 · 视角 01</figcaption></figure><figure><video controls preload="none" src="${x.videos[1]}"></video><figcaption>标注渲染视频 · 视角 02</figcaption></figure></div></article>`).join('')||'<div class="empty">没有匹配的样本</div>'}search.addEventListener('input',render);render();document.querySelector('#artllm').innerHTML=report.artllm.map(x=>`<article class="artllm-card"><h2 class="sample-head">${esc(x.id)}</h2><img src="${x.preview}" alt="${esc(x.id)} 三维结构渲染"><figure><video controls preload="none" src="${x.video}"></video><figcaption>ArtLLM 三维结构旋转渲染</figcaption></figure></article>`).join('');
</script>
</body></html>'''


if __name__ == "__main__":
    main()
