#!/usr/bin/env python3
"""Build a standalone HTML report for one remote model run."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "output" / "remote_model" / "dual_vjepa_obj_urdf_22218"
OUT = ROOT / "output" / "remote_model" / "index.html"
VIDEO_DIR = ROOT / "annoation" / "22218_2f85f6ae"


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> None:
    assets = {
        "obj": data_url(RUN / "generated_assembly.obj"),
        "preview": data_url(RUN / "preview.png"),
        "cam01": data_url(VIDEO_DIR / "rgb_cam01.mp4"),
        "cam02": data_url(VIDEO_DIR / "rgb_cam02.mp4"),
    }
    parts = [{"name": p.stem, "image": data_url(p)} for p in sorted((RUN / "parts_preview").glob("*.png"))]
    tree = json.loads(text(RUN / "kinematic_tree.json"))
    log = text(RUN / "train.log")
    payload = json.dumps({"assets": assets, "parts": parts, "tree": tree, "log": log}, ensure_ascii=False)
    OUT.write_text(TEMPLATE.replace("__PAYLOAD__", payload), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024 / 1024:.1f} MB)")


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V-JEPA 双视角模型结果</title>
<style>
:root{--bg:#f4f6f8;--panel:#fff;--ink:#17202a;--muted:#687581;--line:#d9e0e7;--accent:#1769aa;--good:#18794e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,"PingFang SC","Microsoft YaHei",sans-serif}
header{padding:28px max(20px,calc((100vw - 1440px)/2));background:#17202a;color:#fff}header h1{margin:0;font-size:28px}header p{margin:5px 0 0;color:#bdc8d2}
main{max-width:1440px;margin:auto;padding:22px 20px 48px}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:18px}.stat,.panel{background:var(--panel);border:1px solid var(--line);border-radius:7px}.stat{padding:14px 16px}.stat b{display:block;font-size:22px;margin-top:2px}.muted{color:var(--muted);font-size:12px}
.layout{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(320px,.65fr);gap:18px}.panel{padding:16px}.panel h2{font-size:18px;margin:0 0 12px}.viewer{height:620px;background:#101820;border-radius:5px;overflow:hidden;position:relative}.viewer canvas{display:block;width:100%;height:100%}.hint{position:absolute;left:12px;bottom:10px;color:#d9e4ed;background:#17202acc;padding:5px 8px;border-radius:4px;font-size:12px}
.videos{display:grid;grid-template-columns:1fr 1fr;gap:12px}.videos video{width:100%;background:#101820;border-radius:5px;aspect-ratio:16/9}.table{width:100%;border-collapse:collapse;font-size:12px}.table th,.table td{text-align:left;padding:7px 5px;border-bottom:1px solid var(--line);vertical-align:top}.table th{color:var(--muted);font-weight:600}.prob{color:var(--good);font-variant-numeric:tabular-nums}.log{white-space:pre-wrap;background:#101820;color:#d8e4ef;padding:12px;border-radius:5px;overflow:auto;font:12px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace}
.parts{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.part{border:1px solid var(--line);border-radius:5px;overflow:hidden;background:#f8fafb}.part img{display:block;width:100%;background:#101820}.part figcaption{padding:8px 10px;font-weight:600}
@media(max-width:900px){.stats{grid-template-columns:repeat(2,1fr)}.layout{grid-template-columns:1fr}.viewer{height:480px}}@media(max-width:560px){.videos{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<header><h1>V-JEPA 双视角 OBJ/URDF 重建</h1><p>样本 22218_2f85f6ae · A100 40GB · 800 steps</p></header>
<main>
<section class="stats">
  <div class="stat"><span class="muted">训练步数</span><b>800 / 800</b></div>
  <div class="stat"><span class="muted">最终总损失</span><b>0.00139</b></div>
  <div class="stat"><span class="muted">预测部件</span><b id="partCount">-</b></div>
  <div class="stat"><span class="muted">模型权重</span><b>19.4 MB</b></div>
</section>
<section class="panel" style="margin-bottom:18px"><h2>静态预览</h2><img id="preview" alt="生成的装配静态预览" style="display:block;width:100%;max-height:620px;object-fit:contain;background:#101820;border-radius:5px"></section>
<section class="layout">
  <article class="panel"><h2>生成的三维装配</h2><div class="viewer" id="viewer"><div class="hint">拖拽旋转 · 滚轮缩放 · 右键平移</div></div></article>
  <aside class="panel"><h2>运动学树预测</h2><table class="table"><thead><tr><th>部件</th><th>父节点</th><th>关节</th><th>存在概率</th></tr></thead><tbody id="tree"></tbody></table></aside>
</section>
<section class="panel" style="margin-top:18px"><h2>输入视频</h2><div class="videos"><figure><video controls muted loop></video><figcaption class="muted">cam01</figcaption></figure><figure><video controls muted loop></video><figcaption class="muted">cam02</figcaption></figure></div></section>
<section class="panel" style="margin-top:18px"><h2>生成部件</h2><div class="parts" id="parts"></div></section>
<section class="panel" style="margin-top:18px"><h2>训练日志</h2><pre class="log" id="log"></pre></section>
</main>
<script>window.REPORT=__PAYLOAD__;</script>
<script type="module">
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import {OrbitControls} from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import {OBJLoader} from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/OBJLoader.js';
const report=window.REPORT;
const blob=async(data,type)=>URL.createObjectURL(new Blob([Uint8Array.from(atob(data.split(',')[1]),c=>c.charCodeAt(0))],{type}));
document.querySelector('#preview').src=report.assets.preview;
document.querySelector('#parts').innerHTML=report.parts.map(p=>`<figure class="part"><img src="${p.image}" alt="${p.name}"><figcaption>${p.name}</figcaption></figure>`).join('');
document.querySelectorAll('video')[0].src=report.assets.cam01;
document.querySelectorAll('video')[1].src=report.assets.cam02;
document.querySelector('#log').textContent=report.log;
const active=report.tree.slots.filter(x=>x.link_name);
document.querySelector('#partCount').textContent=active.length;
document.querySelector('#tree').innerHTML=active.map(x=>`<tr><td>${x.link_name}</td><td>${x.parent_slot < active.length ? active[x.parent_slot].link_name : 'world'}</td><td>${['fixed','revolute','prismatic'][x.joint_type_id]||'unknown'}</td><td class="prob">${(x.present_probability*100).toFixed(2)}%</td></tr>`).join('');
const scene=new THREE.Scene();scene.background=new THREE.Color(0x101820);const camera=new THREE.PerspectiveCamera(45,1,.001,100);camera.position.set(1.4,1.1,1.6);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));document.querySelector('#viewer').prepend(renderer.domElement);const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xffffff,0x304050,2.2));const key=new THREE.DirectionalLight(0xffffff,2.4);key.position.set(2,3,4);scene.add(key);scene.add(new THREE.GridHelper(2,20,0x526170,0x26313b));
const objUrl=await blob(report.assets.obj,'text/plain');const object=new OBJLoader().parse(await (await fetch(objUrl)).text());let box=new THREE.Box3().setFromObject(object);let center=box.getCenter(new THREE.Vector3()), size=box.getSize(new THREE.Vector3());object.position.sub(center);const scale=1/Math.max(size.x,size.y,size.z);object.scale.setScalar(scale);object.traverse(n=>{if(n.isMesh){n.material=new THREE.MeshStandardMaterial({color:0x6e9fc4,roughness:.68,metalness:.12});}});scene.add(object);controls.target.set(0,0,0);
function resize(){const el=document.querySelector('#viewer');const w=el.clientWidth,h=el.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()}addEventListener('resize',resize);resize();(function loop(){requestAnimationFrame(loop);controls.update();renderer.render(scene,camera)})();
</script>
</body></html>'''


if __name__ == "__main__":
    main()
