"""Local dashboard for persistent annotation tasks."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from pipeline.render.task_queue import build_queue, claim_task, complete_task, reopen_task


HTML = """<!doctype html>
<meta charset='utf-8'><title>Vid2ArtCAD Queue</title>
<style>
body{font:14px system-ui,sans-serif;margin:24px;color:#20252b;background:#f5f6f7}
header{display:flex;justify-content:space-between;align-items:end;gap:16px;flex-wrap:wrap}
h1{font-size:22px;margin:0}.summary{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}
.stat{background:white;border:1px solid #d8dce0;border-radius:6px;padding:10px 14px;min-width:115px}.stat b{display:block;font-size:20px}
table{width:100%;border-collapse:collapse;background:white;border:1px solid #d8dce0}
th,td{padding:9px 10px;border-bottom:1px solid #e5e7e9;text-align:left;vertical-align:top}th{background:#eef0f2;font-size:12px}
button,input{font:inherit;padding:6px 8px;border:1px solid #aeb5bc;border-radius:4px;background:white}button{cursor:pointer}
.status{font-weight:600}.pending{color:#8a5a00}.claimed{color:#005a9c}.done{color:#28733b}.failed{color:#a12525}.not_applicable{color:#6a7076}
.small{color:#68717a;font-size:12px}.actions{white-space:nowrap}
</style>
<header><div><h1>标注任务队列</h1><div class='small' id='root'></div></div>
<div><input id='worker' placeholder='标注员 / worker' value='local'><button id='refresh'>刷新</button></div></header>
<div class='summary' id='summary'></div><table><thead><tr><th>任务</th><th>样本 / 视角</th><th>状态</th><th>原因</th><th>耗时</th><th>操作</th></tr></thead><tbody id='rows'></tbody></table>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(url,options){const r=await fetch(url,options);const x=await r.json();if(!r.ok)throw Error(x.error||r.statusText);return x}
const elapsed=t=>(t.attempt_history||[]).reduce((sum,a)=>sum+Number(a.duration_s||0),0)+Number(t.duration_s||0);
async function refresh(){const q=await api('/api/queue');document.querySelector('#root').textContent=q.dataset_root;const c=q.summary.status_counts||{};
document.querySelector('#summary').innerHTML=[['总任务',q.summary.task_count],['待处理',c.pending||0],['领取中',c.claimed||0],['已完成',c.done||0],['不适用',c.not_applicable||0],['耗时',`${(q.summary.total_duration_h||0).toFixed(2)} h`],['估算成本',`$${(q.summary.estimated_cost_usd||0).toFixed(2)}`]].map(x=>`<div class='stat'><span>${x[0]}</span><b>${x[1]}</b></div>`).join('');
document.querySelector('#rows').innerHTML=q.tasks.map(t=>`<tr><td><b>${esc(t.stage)}</b><div class='small'>${esc(t.task_id)}</div><div class='small'>${(t.attempt_history||[]).length} 次历史尝试</div></td><td>${esc(t.sample_id)}<div class='small'>${esc(t.view)}</div></td><td class='status ${esc(t.status)}'>${esc(t.status)}<div class='small'>${esc(t.worker||'')}</div></td><td class='small'>${esc((t.reason_codes||[]).join(', '))}</td><td>${elapsed(t).toFixed(1)} s</td><td class='actions'>${t.status==='pending'?`<button data-action="claim" data-task="${esc(t.task_id)}">领取</button>`:''}${t.status==='claimed'?`<button data-action="done" data-task="${esc(t.task_id)}">完成</button> <button data-action="review" data-task="${esc(t.task_id)}">需复核</button> <button data-action="failed" data-task="${esc(t.task_id)}">失败</button>`:''}${['review','failed'].includes(t.status)?`<button data-action="reopen" data-task="${esc(t.task_id)}">重新打开</button>`:''}</td></tr>`).join('')}
async function claim(id){try{await api('/api/claim',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({task_id:id,worker:document.querySelector('#worker').value})});refresh()}catch(e){alert(e.message)}}
async function finish(id,status){let notes=prompt('备注（可选）','')??'';try{await api('/api/complete',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({task_id:id,status,notes})});refresh()}catch(e){alert(e.message)}}
async function reopen(id){let notes=prompt('重开原因（可选）','')??'';try{await api('/api/reopen',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({task_id:id,notes})});refresh()}catch(e){alert(e.message)}}
document.querySelector('#rows').addEventListener('click',e=>{const button=e.target.closest('button[data-action]');if(!button)return;const action=button.dataset.action,id=button.dataset.task;if(action==='claim')claim(id);else if(action==='reopen')reopen(id);else finish(id,action)});
document.querySelector('#refresh').addEventListener('click',refresh);
refresh();setInterval(refresh,10000);
</script>"""


def _json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get('Content-Length', '0'))
    return json.loads(handler.rfile.read(length) or b'{}')


def make_handler(output_root: Path, queue_path: Path, hourly_rate_usd: float,
                 data_root: Path | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'Vid2ArtCADQueue/1.0'

        def log_message(self, _format, *_args):
            return

        def send_json(self, status: int, value: dict):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/':
                body = HTML.encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == '/api/queue':
                return self.send_json(200, build_queue(output_root, queue_path, hourly_rate_usd, data_root))
            self.send_json(404, {'error': 'not found'})

        def do_POST(self):
            try:
                body = _json_body(self)
                path = urlparse(self.path).path
                if path == '/api/claim':
                    result = claim_task(queue_path, str(body.get('task_id', '')), str(body.get('worker', '')))
                elif path == '/api/complete':
                    result = complete_task(queue_path, str(body.get('task_id', '')),
                                           str(body.get('status', 'done')), body.get('duration_s'),
                                           body.get('notes'))
                elif path == '/api/reopen':
                    result = reopen_task(queue_path, str(body.get('task_id', '')), body.get('notes'))
                else:
                    return self.send_json(404, {'error': 'not found'})
                self.send_json(200, result)
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_json(400, {'error': str(exc)})

    return Handler


def serve(output_root: Path, queue_path: Path | None = None, host: str = '127.0.0.1',
          port: int = 8766, hourly_rate_usd: float = 25.0,
          data_root: Path | None = None) -> None:
    output_root = output_root.resolve()
    queue_path = (queue_path or output_root / 'annotation_queue.json').resolve()
    data_root = data_root.resolve() if data_root else None
    build_queue(output_root, queue_path, hourly_rate_usd, data_root)
    server = ThreadingHTTPServer((host, port), make_handler(output_root, queue_path, hourly_rate_usd, data_root))
    print(f'queue review server: http://{host}:{port}/')
    server.serve_forever()
