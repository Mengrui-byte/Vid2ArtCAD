#!/usr/bin/env python3
"""Build a single-file, offline-shareable version of annoation/index.html."""

from __future__ import annotations

import base64
import gzip
import json
import mimetypes
import os
from pathlib import Path

import build_annotation_html


ROOT = Path(__file__).resolve().parent
ANNOTATION_ROOT = ROOT / "annoation"
OUTPUT = ANNOTATION_ROOT / "index.html"
LIB_ROOT = Path("/tmp/v2ac_lib")
ASSET_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".mp4", ".glb", ".gltf", ".bin", ".ply", ".obj", ".mtl", ".urdf",
}


def data_url(path: Path) -> str:
    raw = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:text/javascript;base64,{raw}"


def collect_assets() -> dict[str, Path]:
    assets: dict[str, Path] = {}
    report = {
        "samples": [
            build_annotation_html.collect_sample(path)
            for path in sorted(ANNOTATION_ROOT.iterdir())
            if path.is_dir() and (path / "sample.json").is_file()
        ],
        "artllm": build_annotation_html.collect_artllm(),
    }
    for sample in report["samples"]:
        for field in ("preview", "video01", "video02"):
            logical = sample[field]
            assets[logical] = ANNOTATION_ROOT / logical
    for item in report["artllm"]["items"]:
        for logical in item["assets"].values():
            path = ANNOTATION_ROOT / logical
            if path.is_file() and path.suffix.lower() in ASSET_EXTENSIONS:
                assets[logical] = path
        urdf_dir = (ANNOTATION_ROOT / item["assets"]["urdf"]).parent / "meshes"
        if urdf_dir.is_dir():
            for path in urdf_dir.iterdir():
                if path.is_file() and path.suffix.lower() in ASSET_EXTENSIONS:
                    logical = path.relative_to(ANNOTATION_ROOT).as_posix()
                    assets[logical] = path
    return assets


def embed_assets(html: str) -> str:
    assets = collect_assets()
    ids = {name: f"embedded_asset_{index}" for index, name in enumerate(sorted(assets))}
    payloads = []
    for name in sorted(assets):
        path = assets[name]
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        content = path.read_bytes()
        compressed = path.suffix.lower() in {".glb", ".obj", ".ply"}
        if compressed:
            content = gzip.compress(content, compresslevel=6, mtime=0)
        encoded = base64.b64encode(content).decode("ascii")
        flag = "gzip" if compressed else "none"
        payloads.append(f'<script type="application/octet-stream" id="{ids[name]}" data-mime="{mime}" data-compression="{flag}">{encoded}</script>')
    mapping = json.dumps(ids, ensure_ascii=False, separators=(",", ":"))
    helper = f'''<script>
window.__embeddedAssetIds = {mapping};
window.__embeddedAssetUrls = new Map();
window.assetUrl = async function(path) {{
  const id = window.__embeddedAssetIds[path];
  if (!id) return path;
  if (window.__embeddedAssetUrls.has(path)) return window.__embeddedAssetUrls.get(path);
  const task = (async () => {{
    const node = document.getElementById(id);
    const raw = atob(node.textContent.trim());
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    let content = bytes;
    if (node.dataset.compression === 'gzip') {{
      content = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    }}
    return URL.createObjectURL(new Blob([content], {{type: node.dataset.mime || 'application/octet-stream'}}));
  }})();
  window.__embeddedAssetUrls.set(path, task);
  return task;
}};
</script>'''
    html = html.replace("<body>", "<body>\n" + helper, 1)
    html = html.replace('src="${esc(item.video01)}"', 'data-asset="${esc(item.video01)}"')
    html = html.replace('src="${esc(item.video02)}"', 'data-asset="${esc(item.video02)}"')
    html = html.replace('poster="${esc(item.preview)}"', 'data-poster="${esc(item.preview)}"')
    video_helper = '''<script>
document.querySelectorAll('video[data-poster]').forEach(async video => {
  video.poster = await assetUrl(video.dataset.poster);
});
document.addEventListener('pointerdown', async event => {
  const video = event.target.closest('video[data-asset]');
  if (!video || video.src) return;
  video.src = await assetUrl(video.dataset.asset);
  if (video.dataset.poster) video.poster = await assetUrl(video.dataset.poster);
}, true);
</script>'''
    html = html.replace("<script type=\"module\">", video_helper + "\n  <script type=\"module\">", 1)
    html = html.replace("</body>", "\n".join(payloads) + "\n</body>", 1)
    return html


def inline_three_modules(html: str) -> str:
    modules = {
        "three": LIB_ROOT / "three.module.js",
        "three/addons/controls/OrbitControls.js": LIB_ROOT / "OrbitControls.js",
        "three/addons/loaders/GLTFLoader.js": LIB_ROOT / "GLTFLoader.js",
        "three/addons/loaders/PLYLoader.js": LIB_ROOT / "PLYLoader.js",
        "three/addons/loaders/OBJLoader.js": LIB_ROOT / "OBJLoader.js",
    }
    imports = {
        "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js": "three",
        "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js": "three/addons/controls/OrbitControls.js",
        "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js": "three/addons/loaders/GLTFLoader.js",
        "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/PLYLoader.js": "three/addons/loaders/PLYLoader.js",
        "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/OBJLoader.js": "three/addons/loaders/OBJLoader.js",
    }
    import_map = {"imports": {key: data_url(path) for key, path in modules.items()}}
    html = html.replace(
        '<script type="module">',
        '<script type="importmap">' + json.dumps(import_map, separators=(",", ":")) + '</script>\n  <script type="module">',
        1,
    )
    for old, new in imports.items():
        html = html.replace(old, new)
    html = html.replace("const path = assetUrl(item.assets[kind]);", "const path = await assetUrl(item.assets[kind]);")
    html = html.replace("objLoader.loadAsync(assetUrl(name))", "objLoader.loadAsync(await assetUrl(name))")
    return html


def main() -> None:
    build_annotation_html.main()
    html = OUTPUT.read_text(encoding="utf-8")
    html = inline_three_modules(embed_assets(html))
    html = html.replace(str(build_annotation_html.ARTLLM_ROOT), "已内嵌到此 HTML")
    html = html.replace(str(build_annotation_html.ARTLLM_ZIP), "已内嵌到此 HTML")
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Wrote single-file report: {OUTPUT} ({OUTPUT.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
