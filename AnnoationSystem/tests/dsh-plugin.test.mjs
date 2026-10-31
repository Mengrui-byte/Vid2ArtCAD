import test from 'node:test';
import assert from 'node:assert/strict';
import { apply } from '../dsh-plugin/src/index.mjs';

test('official DSH defineTool accepts all six CAD tool definitions', () => {
  const registered = [];
  apply({ tools: { register(tool) { registered.push(tool); } } });
  assert.equal(registered.length, 6);
  assert.equal(new Set(registered.map((tool) => tool.name)).size, 6);
  assert.ok(registered.some((tool) => tool.name === 'cad_annotation_render_motion_video'));
  assert.ok(registered.some((tool) => tool.name === 'cad_annotation_save_motion_design'));
  for (const tool of registered) {
    assert.equal(tool.parameters.type, 'object');
    assert.equal(tool.output.schema.type, 'string');
  }
});
