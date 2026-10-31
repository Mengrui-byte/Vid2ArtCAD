import fs from 'node:fs';
import assert from 'node:assert/strict';
import { apply } from '../dsh-plugin/src/index.mjs';

const log = fs.readFileSync(new URL('../.dsh/server.log', import.meta.url), 'utf8');
const url = log.match(/http:\/\/127\.0\.0\.1:\d+\/\?token=[\w-]+/g)?.at(-1);
assert.ok(url, 'Start the local server before running the smoke test');
const origin = new URL(url).origin;
const auth = await fetch(url, { redirect: 'manual', signal: AbortSignal.timeout(10000) });
const cookie = auth.headers.getSetCookie().map((entry) => entry.split(';')[0]).join('; ');
const page = await fetch(origin, { headers: { Cookie: cookie }, signal: AbortSignal.timeout(10000) });
assert.equal(page.status, 200);
const response = await fetch(`${origin}/api/agentPresets/list`, {
  method: 'POST',
  headers: { Cookie: cookie, Origin: origin, 'Content-Type': 'application/json' },
  body: JSON.stringify({ type: 'client-request', rpcId: 'cad-smoke', method: 'agentPresets/list', payload: { args: {} } }),
  signal: AbortSignal.timeout(10000),
});
const result = await response.json();
assert.equal(result.result.ok, true);
assert.ok(result.result.value.presets.some((preset) => preset.id === 'cad-motion' && preset.isDefault));
const tools = [];
apply({ tools: { register(tool) { tools.push(tool); } } });
const inspector = tools.find((tool) => tool.name === 'cad_annotation_inspect_asset');
const asset = JSON.parse(await inspector.execute({ asset_id: '21634_e5e20375' }));
assert.equal(asset.inventory.body_count, 3);
assert.equal(asset.collision_scan.status, 'not_run');
console.log(JSON.stringify({ web: 'passed', default_preset: 'cad-motion', registered_tools: tools.length, local_official_tool_execution: 'passed', asset: asset.asset_id, model_inference: 'not_tested_requires_credentials' }, null, 2));
