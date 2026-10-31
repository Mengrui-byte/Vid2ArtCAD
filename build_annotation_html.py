#!/usr/bin/env python3
"""Build a self-contained browser index for annoation samples."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
ANNOTATION_ROOT = ROOT / "annoation"
OUTPUT = ANNOTATION_ROOT / "index.html"
ARTLLM_ROOT = Path("/Users/mengrui/Desktop/batch_full_20261020_131442")
ARTLLM_ZIP = Path("/Users/mengrui/Desktop/full_pipeline_bucket.zip")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def rel(path: Path) -> str:
    return path.relative_to(ANNOTATION_ROOT).as_posix()


def collect_sample(folder: Path) -> dict:
    description = read_json(folder / "motion_description.json")
    quality = read_json(folder / "quality.json")
    sample = read_json(folder / "sample.json")
    views = read_json(folder / "views.json")
    source_quality = read_json(folder / "source_quality.json")
    files = sorted(
        path.name
        for path in folder.iterdir()
        if path.is_file() and path.name not in {".DS_Store", "index.html"}
    )
    return {
        "id": folder.name,
        "title": description.get("title_cn", folder.name),
        "description": description.get("description_cn", ""),
        "duration": description.get("duration_s", quality.get("videos", {}).get("cam01", {}).get("frame_count", 0) / max(quality.get("videos", {}).get("cam01", {}).get("fps", 1), 1)),
        "status": quality.get("status", "unknown"),
        "drivers": description.get("drivers") or list(description.get("commands", {}).keys()),
        "commands": description.get("commands", {}),
        "units": description.get("joint_units", {}),
        "description_json": description,
        "quality_json": quality,
        "sample_json": sample,
        "views_json": views,
        "source_quality_json": source_quality,
        "preview": f"{folder.name}/preview/views_contact_sheet.png",
        "video01": f"{folder.name}/rgb_cam01.mp4",
        "video02": f"{folder.name}/rgb_cam02.mp4",
        "files": files,
    }


def collect_artllm() -> dict:
    if not ARTLLM_ROOT.exists() and not ARTLLM_ZIP.exists():
        return {"available": False, "items": []}
    success = set(read_text(ARTLLM_ROOT / "success.txt").split())
    failed_lines = [line.strip() for line in read_text(ARTLLM_ROOT / "failed.txt").splitlines() if line.strip()]
    failed = {line.split(":", 1)[0]: line.split(":", 1)[1].strip() if ":" in line else "失败" for line in failed_lines}
    items = []
    output_root = ARTLLM_ROOT / "output"
    for folder in sorted(path for path in output_root.iterdir() if path.is_dir()):
        if folder.name in failed:
            continue
        artllm = folder / "artllm"
        articulation = read_json(artllm / "articulation.json")
        bbox = read_json(artllm / "bbox.json")
        if not articulation and not bbox and not (artllm / f"{folder.name}.txt").exists():
            continue
        output_files = sorted(str(path.relative_to(ARTLLM_ROOT)) for path in folder.rglob("*") if path.is_file())
        items.append({
            "id": folder.name,
            "source": str(ARTLLM_ROOT / "output" / folder.name),
            "assets": artllm_assets(folder.name),
            "status": "success" if folder.name in success else "failed" if folder.name in failed else "completed",
            "failure": failed.get(folder.name, ""),
            "joint_count": len(articulation) if isinstance(articulation, list) else 0,
            "joint_types": sorted({joint.get("type", "unknown") for joint in articulation if isinstance(joint, dict)}),
            "link_count": len(bbox) if isinstance(bbox, list) else 0,
            "articulation": articulation,
            "bbox": bbox,
            "text": read_text(artllm / f"{folder.name}.txt"),
            "files": output_files,
        })
    if ARTLLM_ZIP.exists():
        with ZipFile(ARTLLM_ZIP) as archive:
            prefix = "full_pipeline_bucket/output/bucket/"
            names = [name for name in archive.namelist() if name.startswith(prefix) and not name.endswith("/")]
            articulation_name = prefix + "artllm/articulation.json"
            bbox_name = prefix + "artllm/bbox.json"
            articulation = json.loads(archive.read(articulation_name).decode("utf-8")) if articulation_name in names else []
            bbox = json.loads(archive.read(bbox_name).decode("utf-8")) if bbox_name in names else []
            text_name = prefix + "artllm/bucket.txt"
            text = archive.read(text_name).decode("utf-8") if text_name in names else ""
            items.insert(0, {
                "id": "bucket",
                "source": str(ARTLLM_ZIP),
                "assets": artllm_assets("bucket"),
                "status": "completed",
                "failure": "",
                "joint_count": len(articulation),
                "joint_types": sorted({joint.get("type", "unknown") for joint in articulation if isinstance(joint, dict)}),
                "link_count": len(bbox),
                "articulation": articulation,
                "bbox": bbox,
                "text": text,
                "files": [name.removeprefix("full_pipeline_bucket/") for name in names],
            })
    return {
        "available": True,
        "root": str(ARTLLM_ROOT),
        "zip": str(ARTLLM_ZIP) if ARTLLM_ZIP.exists() else "",
        "batch_readme": read_text(ARTLLM_ROOT / "README.txt"),
        "items": items,
    }


def artllm_assets(sample_id: str) -> dict:
    root = f"artllm_assets/{sample_id}"
    stem = f"train_cfg_-1.000000_"
    return {
        "input": f"{root}/xpart/{stem}input_boxgpt_{sample_id}.glb",
        "parts": f"{root}/xpart/{stem}dit_boxgpt_{sample_id}.glb",
        "bbox": f"{root}/xpart/{stem}dit_boxgpt_{sample_id}_bbox.glb",
        "explode": f"{root}/xpart/{stem}explode_boxgpt_{sample_id}.glb",
        "ply": f"{root}/artllm/{sample_id}.ply",
        "urdf": f"{root}/urdf/{sample_id}.urdf",
    }


def main() -> None:
    samples = [
        collect_sample(path)
        for path in sorted(ANNOTATION_ROOT.iterdir())
        if path.is_dir() and (path / "sample.json").is_file()
    ]
    report = {"samples": samples, "artllm": collect_artllm()}
    data = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    output = TEMPLATE.replace("__SAMPLE_DATA__", data)
    OUTPUT.write_text(output, encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(samples)} samples)")


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>1020组会</title>
  <style>
    :root { color-scheme: light; --ink:#17202a; --muted:#66717d; --line:#d9e0e7; --paper:#f5f7f9; --panel:#fff; --accent:#1769aa; --ok:#19724a; --warn:#9a5b00; --bad:#a53a35; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:var(--paper); font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,"PingFang SC","Microsoft YaHei",sans-serif; }
    header { background:#17202a; color:#fff; padding:28px max(20px, calc((100vw - 1380px) / 2)); }
    header h1 { margin:0 0 5px; font-size:25px; font-weight:700; letter-spacing:0; }
    header p { margin:0; color:#bdc8d2; }
    main { max-width:1380px; margin:0 auto; padding:22px 20px 44px; }
    .toolbar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:18px; }
    input, select { border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:5px; padding:9px 11px; font:inherit; }
    input { flex:1; min-width:220px; }
    .count { color:var(--muted); margin-left:auto; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(560px,1fr)); gap:18px; }
    .sample { background:var(--panel); border:1px solid var(--line); border-radius:7px; overflow:hidden; box-shadow:0 2px 8px rgba(23,32,42,.04); }
    .sample-head { display:flex; justify-content:space-between; gap:15px; padding:17px 18px 12px; border-bottom:1px solid var(--line); }
    h2 { margin:0; font-size:19px; line-height:1.3; }
    .id { color:var(--muted); font-size:12px; margin-top:3px; }
    .badge { height:max-content; border:1px solid #a8d9bf; color:var(--ok); background:#edf9f2; border-radius:999px; padding:3px 8px; white-space:nowrap; font-size:12px; }
    .description { padding:13px 18px 0; margin:0; }
    .videos { display:grid; grid-template-columns:1fr 1fr; gap:10px; padding:15px 18px 10px; }
    figure { margin:0; min-width:0; }
    video, .preview { display:block; width:100%; aspect-ratio:16/9; object-fit:cover; background:#edf0f3; border:1px solid var(--line); border-radius:4px; }
    figcaption { color:var(--muted); font-size:12px; padding-top:5px; }
    .meta { display:flex; flex-wrap:wrap; gap:7px; padding:3px 18px 14px; }
    .chip { background:#f0f4f7; color:#40505f; border-radius:4px; padding:4px 7px; font-size:12px; }
    details { border-top:1px solid var(--line); padding:0 18px; }
    summary { cursor:pointer; padding:11px 0; color:var(--accent); font-weight:600; }
    .json { background:#17202a; color:#e4edf4; overflow:auto; padding:12px; margin:0 0 14px; border-radius:4px; max-height:280px; font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre-wrap; word-break:break-word; }
    .links { display:flex; flex-wrap:wrap; gap:8px; padding:0 18px 17px; }
    .links a { color:var(--accent); text-decoration:none; border:1px solid var(--line); border-radius:4px; padding:5px 8px; font-size:12px; }
    .links a:hover { border-color:var(--accent); }
    .section-title { margin:30px 0 12px; font-size:21px; }
    .section-note { color:var(--muted); margin:-5px 0 15px; }
    .artllm-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(640px,1fr)); gap:18px; }
    .artllm-card { background:var(--panel); border:1px solid var(--line); border-radius:7px; overflow:hidden; }
    .artllm-head { display:flex; justify-content:space-between; gap:12px; padding:16px 18px 12px; border-bottom:1px solid var(--line); }
    .artllm-head h3 { margin:0; font-size:18px; }
    .artllm-status { border-radius:999px; padding:3px 8px; font-size:12px; white-space:nowrap; }
    .artllm-status.success { color:var(--ok); background:#edf9f2; border:1px solid #a8d9bf; }
    .artllm-status.failed { color:var(--bad); background:#fff0ef; border:1px solid #e6b7b4; }
    .artllm-body { padding:13px 18px 17px; }
    .viewer { height:430px; position:relative; overflow:hidden; background:#101820; border-radius:5px; }
    .viewer canvas { display:block; width:100%; height:100%; }
    .viewer-loading { position:absolute; inset:0; display:grid; place-items:center; color:#c8d3dc; pointer-events:none; }
    .view-controls { display:flex; flex-wrap:wrap; gap:6px; padding:10px 0 0; }
    .view-controls button { border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:4px; padding:6px 9px; cursor:pointer; font:12px inherit; }
    .view-controls button.active { color:#fff; background:var(--accent); border-color:var(--accent); }
    .view-hint { color:var(--muted); font-size:12px; margin-top:7px; }
    @media (max-width:700px) { .grid { grid-template-columns:1fr; } .videos { grid-template-columns:1fr; } .count { margin-left:0; width:100%; } }
  </style>
</head>
<body>
  <header>
    <h1>1020组会</h1>
    <p>标注数据集与三维运动学结构重建</p>
  </header>
  <main>
    <section class="idea">
      <h2 class="section-title">我的 Idea</h2>
      <p class="section-note">先标注数据集；再使用 V-JEPA 作为视频编码器，从视频中识别部件特征与运动学特征；随后重建三维结构和运动学结构，最终输出 BRep 与 URDF。</p>
    </section>
    <div class="toolbar">
      <input id="search" type="search" placeholder="搜索样本编号、名称或描述">
      <select id="status"><option value="all">全部状态</option><option value="multiview_replay_passed">校验通过</option></select>
      <span id="count" class="count"></span>
    </div>
    <section id="grid" class="grid"></section>
    <h2 class="section-title">ArtLLM 结构推理结果</h2>
    <p class="section-note">本页仅展示已完成的 ArtLLM 推理结果。使用下方按钮切换不同 3D 推理产物，拖动旋转、滚轮缩放。</p>
    <section id="artllm" class="artllm-grid"></section>
  </main>
  <script>
    const report = __SAMPLE_DATA__;
    const samples = report.samples;
    const artllm = report.artllm;
    const grid = document.querySelector('#grid');
    const search = document.querySelector('#search');
    const status = document.querySelector('#status');
    const count = document.querySelector('#count');
    const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    const json = value => esc(JSON.stringify(value, null, 2));
    function render() {
      const query = search.value.trim().toLowerCase();
      const filtered = samples.filter(item => {
        const text = [item.id, item.title, item.description].join(' ').toLowerCase();
        return (!query || text.includes(query)) && (status.value === 'all' || item.status === status.value);
      });
      count.textContent = `${filtered.length} / ${samples.length} 个样本`;
      grid.innerHTML = filtered.map(item => {
        const chips = [`时长 ${Number(item.duration).toFixed(1)} s`, `驱动 ${item.drivers.length} 个`].concat(item.drivers.map(driver => `${esc(driver)} ${item.commands[driver] ?? ''}${item.units[driver] ? ' ' + item.units[driver] : ''}`));
        return `<article class="sample">
          <div class="sample-head"><div><h2>${esc(item.title)}</h2><div class="id">${esc(item.id)}</div></div><span class="badge">${esc(item.status)}</span></div>
          <p class="description">${esc(item.description || '暂无描述')}</p>
          <div class="videos">
            <figure><video controls preload="metadata" poster="${esc(item.preview)}" src="${esc(item.video01)}"></video><figcaption>视角 01 · cam01</figcaption></figure>
            <figure><video controls preload="metadata" poster="${esc(item.preview)}" src="${esc(item.video02)}"></video><figcaption>视角 02 · cam02</figcaption></figure>
          </div>
          <div class="meta">${chips.map(chip => `<span class="chip">${chip}</span>`).join('')}</div>
          <details><summary>查看结构化标注 JSON</summary><pre class="json">${json({sample:item.sample_json, description:item.description_json, quality:item.quality_json, views:item.views_json, source_quality:item.source_quality_json})}</pre></details>
          <details><summary>查看文件清单</summary><pre class="json">${json(item.files)}</pre></details>
          <div class="links"><a href="${esc(item.preview)}" target="_blank">打开预览图</a><a href="${esc(item.video01)}" download>下载 cam01</a><a href="${esc(item.video02)}" download>下载 cam02</a></div>
        </article>`;
      }).join('') || '<p>没有匹配的样本。</p>';
    }
    search.addEventListener('input', render); status.addEventListener('change', render); render();
    document.querySelector('#artllm').innerHTML = (artllm.items || []).map(item => {
      return `<article class="artllm-card">
        <div class="artllm-head"><h3>${esc(item.id)}</h3><span class="artllm-status success">3D 推理结果</span></div>
        <div class="artllm-body">
          <div class="viewer" data-viewer="${esc(item.id)}"><div class="viewer-loading">点击下方按钮加载 3D 模型</div></div>
          <div class="view-controls" data-controls="${esc(item.id)}">
            <button data-kind="input">输入分件</button><button data-kind="parts">预测分件</button><button data-kind="bbox">bbox 结果</button><button data-kind="explode">爆炸图</button><button data-kind="ply">ArtLLM 点云</button><button data-kind="urdf">URDF 网格</button>
          </div>
          <div class="view-hint">${item.link_count} 个部件 · ${item.joint_count} 个关节 · ${esc(item.joint_types.join(' / ') || '无关节')}</div>
        </div>
      </article>`;
    }).join('');
  </script>
  <script type="module">
    import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
    import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';
    import { GLTFLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';
    import { PLYLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/PLYLoader.js';
    import { OBJLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/OBJLoader.js';
    const report3d = __SAMPLE_DATA__;
    const gltfLoader = new GLTFLoader();
    const plyLoader = new PLYLoader();
    const objLoader = new OBJLoader();
    const viewers = new Map();
    function frameObject(camera, object, controls) {
      const box = new THREE.Box3().setFromObject(object);
      if (box.isEmpty()) return;
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z) * 0.65 || 1;
      camera.position.copy(center).add(new THREE.Vector3(radius * 1.8, radius * 1.2, radius * 1.8));
      camera.near = Math.max(radius / 1000, 0.001); camera.far = radius * 100; camera.updateProjectionMatrix();
      controls.target.copy(center); controls.update();
    }
    function makeScene(element) {
      const renderer = new THREE.WebGLRenderer({antialias:true});
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setClearColor(0x101820, 1);
      element.querySelector('canvas')?.remove(); element.appendChild(renderer.domElement);
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1000);
      const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
      scene.add(new THREE.HemisphereLight(0xddeeff, 0x334455, 2.2));
      const key = new THREE.DirectionalLight(0xffffff, 2.5); key.position.set(3,5,4); scene.add(key);
      const grid = new THREE.GridHelper(10, 20, 0x405160, 0x26343e); grid.position.y = -1.5; scene.add(grid);
      const state = {renderer,scene,camera,controls,current:null};
      viewers.set(element.dataset.viewer, state);
      const resize = () => { const rect=element.getBoundingClientRect(); renderer.setSize(rect.width,rect.height,false); camera.aspect=rect.width/rect.height; camera.updateProjectionMatrix(); };
      new ResizeObserver(resize).observe(element); resize();
      const tick = () => { controls.update(); renderer.render(scene,camera); requestAnimationFrame(tick); }; tick();
      return state;
    }
    function clearModel(state) { if (!state.current) return; state.scene.remove(state.current); state.current.traverse(obj => { obj.geometry?.dispose(); if (obj.material) { const mats=Array.isArray(obj.material)?obj.material:[obj.material]; mats.forEach(mat=>mat.dispose()); }}); state.current=null; }
    async function loadModel(item, kind, state) {
      clearModel(state); const loading = document.querySelector(`[data-viewer="${item.id}"] .viewer-loading`); loading.style.display='grid';
      const path = item.assets[kind]; let object;
      try {
        if (kind === 'ply') { const geometry = await plyLoader.loadAsync(path); geometry.computeVertexNormals(); const material = new THREE.PointsMaterial({size:0.018, vertexColors:!!geometry.getAttribute('color'), color:0x86c8ff}); object = new THREE.Points(geometry, material); }
        else if (kind === 'urdf') { const group = new THREE.Group(); const names = Array.from({length:item.link_count}, (_,i)=>`${item.assets.urdf.replace(/[^/]+$/, 'meshes/link_' + i + '.obj')}`); for (const name of names) { try { const mesh=await objLoader.loadAsync(name); mesh.traverse(o=>{if(o.isMesh)o.material=new THREE.MeshStandardMaterial({color:0x66a9d8,metalness:.1,roughness:.7});}); group.add(mesh); } catch(e) {} } object=group; }
        else { const gltf = await gltfLoader.loadAsync(path); object = gltf.scene; }
        state.current=object; state.scene.add(object); frameObject(state.camera,object,state.controls); loading.style.display='none';
      } catch (error) { loading.textContent='模型加载失败，请通过本地 HTTP 服务打开页面'; }
    }
    for (const item of report3d.artllm.items || []) {
      const element=document.querySelector(`[data-viewer="${item.id}"]`); if (!element) continue; const state=makeScene(element);
      document.querySelectorAll(`[data-controls="${item.id}"] button`).forEach(button => button.addEventListener('click', () => { document.querySelectorAll(`[data-controls="${item.id}"] button`).forEach(b=>b.classList.remove('active')); button.classList.add('active'); loadModel(item,button.dataset.kind,state); }));
    }
  </script>
</body>
</html>
'''


if __name__ == "__main__":
    main()
