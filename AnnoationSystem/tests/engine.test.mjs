import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createReviewTask, inspectAsset, motionFeatureBatch, rankAssets, saveMotionDesign, saveReview } from '../dsh-plugin/src/engine.mjs';

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'annoation-system-'));
  const asset = path.join(root, 'sample_01');
  fs.mkdirSync(asset);
  const assembly = {
    bodies: { base: { name: 'Base', obj: 'base.obj' }, lid: { name: 'Hinge Lid', obj: 'lid.obj' } },
    joints: {},
    occurrences: {},
    root: { bodies: { base: { is_visible: true }, lid: { is_visible: true } } },
    holes: [
      { body: 'base', type: 'RoundHole', diameter: 0.2, origin: { x: 0, y: 1, z: 2 }, direction: { x: 1, y: 0, z: 0 } },
      { body: 'lid', type: 'RoundHole', diameter: 0.2, origin: { x: 4, y: 1, z: 2 }, direction: { x: -1, y: 0, z: 0 } },
    ],
    contacts: [{ entity_one: { body: 'base', surface_type: 'Cylinder' }, entity_two: { body: 'lid', surface_type: 'Cylinder' } }],
  };
  fs.writeFileSync(path.join(asset, 'assembly.json'), JSON.stringify(assembly));
  fs.writeFileSync(path.join(asset, 'base.obj'), 'v 0 0 0\nf 1 2 3\n');
  return { root, asset };
}

test('inspects aligned cross-body hole axes as unvalidated hypotheses', () => {
  const { root } = fixture();
  const report = inspectAsset(root, 'sample_01');
  assert.equal(report.inventory.occurrence_count, 0);
  assert.equal(report.geometric_motion_hypotheses.length, 1);
  assert.equal(report.geometric_motion_hypotheses[0].status, 'hypothesis_only');
  assert.deepEqual(report.geometric_motion_hypotheses[0].axis_direction, [1, 0, 0]);
  assert.equal(report.collision_scan.status, 'not_run');
  assert.equal(report.video.status, 'not_generated');
  fs.rmSync(root, { recursive: true, force: true });
});

test('rejects paths that escape configured dataset root', () => {
  const { root } = fixture();
  assert.throws(() => inspectAsset(root, '../elsewhere'), /unsupported characters/);
  fs.rmSync(root, { recursive: true, force: true });
});

test('ranks candidates and records a non-ground-truth human review', () => {
  const { root } = fixture();
  const report = inspectAsset(root, 'sample_01');
  const ranking = rankAssets(root, 5);
  assert.equal(ranking.assets[0].asset_id, 'sample_01');
  const output = path.join(root, 'out');
  const created = createReviewTask(output, report, 'check hinge evidence');
  assert.equal(created.status, 'pending_human_review');
  assert.throws(() => createReviewTask(output, report), /already exists/);
  const reviewed = saveReview(output, 'sample_01', 'approved_hypothesis', 'reviewer-1', 'axis plausible');
  assert.equal(reviewed.status, 'hypothesis_approved_not_ground_truth');
  fs.rmSync(root, { recursive: true, force: true });
});

test('paginates parsed motion features and persists an explicitly unvalidated LLM design', () => {
  const { root } = fixture();
  const batch = motionFeatureBatch(root, 0, 1, true);
  assert.equal(batch.pagination.eligible_asset_count, 1);
  assert.equal(batch.assets[0].inferred_axis_hypotheses.length, 1);
  assert.equal(batch.assets[0].bodies.length, 2);
  assert.equal(batch.pagination.complete, true);
  assert.equal(batch.dataset_summary.joint_free_asset_count, 1);

  const proposal = {
    motion_type: 'revolute',
    motion_intent: 'Open the lid for access',
    rationale: 'The cross-body mounting holes share an axis.',
    fixed_body_ids: ['base'],
    moving_body_ids: ['lid'],
    joint_or_axis_reference: 'aligned hole axis from feature hypothesis 0',
    coordinate_range: { minimum: -0.4, maximum: 0.4, unit: 'rad' },
    trajectory_profile: 'smooth_open_hold_close',
    duration_seconds: 3.0,
    trajectory_keyframes: [
      { time_seconds: 0, coordinate: 0 },
      { time_seconds: 1.2, coordinate: 0.4 },
      { time_seconds: 1.8, coordinate: 0.4 },
      { time_seconds: 3, coordinate: 0 },
    ],
    evidence: ['two cross-body holes have parallel directions and coaxial origins'],
    assumptions: ['the bodies correspond to the fixed base and moving lid'],
    confidence: 0.7,
  };
  const saved = saveMotionDesign(path.join(root, 'out'), root, 'sample_01', JSON.stringify(proposal));
  assert.equal(saved.status, 'proposal_requires_unit_and_geometry_review');
  assert.equal(saved.design.ground_truth, false);
  assert.equal(saved.design.collision_validation, 'not_run');
  assert.equal(saved.design.trajectory_keyframes.length, 4);
  assert.equal(saved.design.source_sha256, batch.assets[0].source_sha256);
  assert.throws(() => saveMotionDesign(path.join(root, 'out'), root, 'sample_01', JSON.stringify({ ...proposal, moving_body_ids: ['unknown'] })), /unknown body id/);
  fs.rmSync(root, { recursive: true, force: true });
});

test('returns source joint axes and source limits without inventing units', () => {
  const { root, asset } = fixture();
  const assemblyPath = path.join(asset, 'assembly.json');
  const assembly = JSON.parse(fs.readFileSync(assemblyPath, 'utf8'));
  assembly.joints = {
    joint_1: {
      name: 'Hinge',
      occurrence_one: 'occ_a',
      occurrence_two: 'occ_b',
      joint_motion: {
        joint_type: 'RevoluteJointType',
        rotation_limits: { minimum_value: -0.5, maximum_value: 0.5, is_minimum_value_enabled: true, is_maximum_value_enabled: true },
      },
      geometry_or_origin_one: { origin: { x: 0, y: 1, z: 2 }, primary_axis_vector: { x: 1, y: 0, z: 0 } },
    },
  };
  fs.writeFileSync(assemblyPath, JSON.stringify(assembly));
  const features = motionFeatureBatch(root, 0, 5, false).assets[0];
  assert.equal(features.joints[0].source_joint_type, 'RevoluteJointType');
  assert.equal(features.joints[0].rotation_limits.minimum_enabled, true);
  assert.equal(features.joints[0].rotation_limits.source_unit, 'not_declared_in_assembly_json');
  assert.deepEqual(features.joints[0].frame_one.primary_axis, [1, 0, 0]);
  fs.rmSync(root, { recursive: true, force: true });
});
