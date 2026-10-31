import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { randomUUID } from 'node:crypto';

const EPS = 1e-10;

function loadJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function vec3(value) {
  if (!value || !['x', 'y', 'z'].every((key) => Number.isFinite(value[key]))) return null;
  return [value.x, value.y, value.z];
}

function normalize(a) {
  const n = Math.hypot(...a);
  return n > EPS ? a.map((v) => v / n) : null;
}

function dot(a, b) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function distanceToLine(point, origin, direction) {
  const delta = point.map((value, index) => value - origin[index]);
  const projection = dot(delta, direction);
  return Math.hypot(...delta.map((value, index) => value - projection * direction[index]));
}

function bodyNames(assembly) {
  return Object.fromEntries(Object.entries(assembly.bodies ?? {}).map(([id, body]) => [id, body.name ?? id]));
}

function compactPoint(value) {
  return vec3(value);
}

function jointFeatures(assembly) {
  return Object.entries(assembly.joints ?? {}).map(([id, joint]) => {
    const motion = joint.joint_motion ?? {};
    const geometry = (value) => value ? {
      origin: compactPoint(value.origin),
      primary_axis: compactPoint(value.primary_axis_vector),
      secondary_axis: compactPoint(value.secondary_axis_vector),
      tertiary_axis: compactPoint(value.tertiary_axis_vector),
      entity: value.entity_one ? {
        occurrence: value.entity_one.occurrence ?? null,
        body: value.entity_one.body ?? null,
        surface_type: value.entity_one.surface_type ?? null,
      } : null,
    } : null;
    const limits = (raw) => raw ? {
      minimum: raw.minimum_value ?? null,
      maximum: raw.maximum_value ?? null,
      rest: raw.rest_value ?? null,
      minimum_enabled: Boolean(raw.is_minimum_value_enabled),
      maximum_enabled: Boolean(raw.is_maximum_value_enabled),
      rest_enabled: Boolean(raw.is_rest_value_enabled),
      source_unit: 'not_declared_in_assembly_json',
    } : null;
    return {
      id,
      name: joint.name ?? id,
      source_joint_type: motion.joint_type ?? joint.type ?? 'unknown',
      parent_component: joint.parent_component ?? null,
      occurrence_one: joint.occurrence_one ?? null,
      occurrence_two: joint.occurrence_two ?? null,
      rotation_axis_label: motion.rotation_axis ?? null,
      rotation_limits: limits(motion.rotation_limits),
      slide_limits: limits(motion.slide_limits),
      frame_one: geometry(joint.geometry_or_origin_one),
      frame_two: geometry(joint.geometry_or_origin_two),
      is_flipped: Boolean(joint.is_flipped),
      source_values_preserved: true,
    };
  });
}

function occurrenceFeatures(assembly) {
  return Object.entries(assembly.occurrences ?? {}).map(([id, occurrence]) => ({
    id,
    name: occurrence.name ?? id,
    component: occurrence.component ?? occurrence.component_id ?? null,
    parent: occurrence.parent ?? occurrence.parent_occurrence ?? null,
    bodies: Object.keys(occurrence.bodies ?? {}),
    transform: occurrence.transform ?? occurrence.transform_matrix ?? null,
  }));
}

function componentFeatures(assembly) {
  return Object.entries(assembly.components ?? {}).map(([id, component]) => ({
    id,
    name: component.name ?? component.part_number ?? id,
    type: component.type ?? null,
    body_ids: Array.isArray(component.bodies) ? component.bodies : Object.keys(component.bodies ?? {}),
  }));
}

function holeFeatures(assembly) {
  return (assembly.holes ?? []).map((hole, index) => ({
    index,
    body: hole.body ?? null,
    type: hole.type ?? 'unknown',
    origin: compactPoint(hole.origin),
    direction: normalize(vec3(hole.direction) ?? []),
    diameter: Number.isFinite(hole.diameter) ? hole.diameter : null,
    length: Number.isFinite(hole.length) ? hole.length : null,
  }));
}

function featureRecord(assetId, assembly, sourceHash = null, assetPath = null) {
  const bodies = bodyNames(assembly);
  const jointList = jointFeatures(assembly);
  const occurrences = occurrenceFeatures(assembly);
  const holes = holeFeatures(assembly);
  const digest = sourceHash ?? crypto.createHash('sha256').update(JSON.stringify(assembly)).digest('hex');
  return {
    asset_id: assetId,
    source_sha256: digest,
    body_count: Object.keys(bodies).length,
    occurrence_count: occurrences.length,
    joint_count: jointList.length,
    hole_count: holes.length,
    bodies: Object.entries(assembly.bodies ?? {}).map(([id, body]) => ({
      id,
      name: body.name ?? id,
      type: body.type ?? null,
      visible: assembly.root?.bodies?.[id]?.is_visible ?? null,
      material: body.material?.name ?? null,
      appearance: body.appearance?.name ?? body.material?.appearance?.name ?? null,
      physical_properties: body.physical_properties ?? null,
    })),
    components: componentFeatures(assembly),
    occurrences,
    joints: jointList,
    holes,
    contact_pairs: crossBodyContacts(assembly, bodies).map((pair) => ({
      body_ids: pair.body_ids,
      body_names: pair.body_names,
      feature_count: pair.count,
      surface_pairs: [...new Set(pair.surface_pairs.map((surfaces) => surfaces.join(' <-> ')))],
    })),
    inferred_axis_hypotheses: inferAxisEvidence(assembly),
    visual_assets: assetPath ? visualAssets(assetPath, assembly) : null,
    hierarchy_status: occurrences.length ? 'available' : 'missing',
  };
}

function collectHoleAxes(assembly) {
  return (assembly.holes ?? []).flatMap((hole, index) => {
    const origin = vec3(hole.origin);
    const direction = vec3(hole.direction);
    const axis = direction && normalize(direction);
    if (!origin || !axis || !hole.body) return [];
    return [{ index, body: hole.body, origin, axis, diameter: hole.diameter ?? null, kind: hole.type ?? 'unknown' }];
  });
}

function inferAxisEvidence(assembly) {
  const holes = collectHoleAxes(assembly);
  const clusters = [];
  for (let i = 0; i < holes.length; i += 1) {
    for (let j = i + 1; j < holes.length; j += 1) {
      const first = holes[i];
      const second = holes[j];
      if (first.body === second.body) continue;
      const alignment = Math.abs(dot(first.axis, second.axis));
      if (alignment < 0.995) continue;
      const lineDistance = distanceToLine(second.origin, first.origin, first.axis);
      if (lineDistance > 0.08) continue;
      const matched = [first, second];
      const groupBodies = new Set(matched.map((hole) => hole.body));
      for (const candidate of holes) {
        if (groupBodies.has(candidate.body)) continue;
        const candidateAlignment = Math.abs(dot(candidate.axis, first.axis));
        if (candidateAlignment >= 0.995 && distanceToLine(candidate.origin, first.origin, first.axis) <= 0.08) {
          matched.push(candidate);
          groupBodies.add(candidate.body);
        }
      }
      const direction = first.axis.find((value) => Math.abs(value) > 1e-8) < 0
        ? first.axis.map((value) => -value)
        : first.axis;
      const point = first.origin.map((value, index) => value - dot(first.origin, direction) * direction[index]);
      const key = `${direction.map((value) => value.toFixed(3)).join(',')}:${point.map((value) => value.toFixed(2)).join(',')}:${[...groupBodies].sort().join(',')}`;
      if (!clusters.some((cluster) => cluster.key === key)) {
        clusters.push({
          key,
          type: 'revolute_candidate',
          axis_direction: direction,
          axis_point_closest_to_origin: point,
          supporting_holes: matched.map((hole) => ({
            index: hole.index,
            body: hole.body,
            diameter: hole.diameter,
            origin: hole.origin,
            direction: hole.axis,
          })),
          supporting_body_ids: [...groupBodies],
          alignment_threshold: 0.995,
          line_distance_threshold_source_units: 0.08,
          status: 'hypothesis_only',
        });
      }
    }
  }
  return clusters.sort((a, b) => b.supporting_holes.length - a.supporting_holes.length);
}

function crossBodyContacts(assembly, names) {
  const pairs = new Map();
  for (const contact of assembly.contacts ?? []) {
    const one = contact.entity_one ?? {};
    const two = contact.entity_two ?? {};
    if (!one.body || !two.body || one.body === two.body) continue;
    const ids = [one.body, two.body].sort();
    const key = ids.join('|');
    const item = pairs.get(key) ?? { body_ids: ids, body_names: ids.map((id) => names[id] ?? id), count: 0, surface_pairs: [] };
    item.count += 1;
    item.surface_pairs.push([one.surface_type ?? 'unknown', two.surface_type ?? 'unknown']);
    pairs.set(key, item);
  }
  return [...pairs.values()].sort((a, b) => b.count - a.count);
}

function objQuickStats(file) {
  if (!fs.existsSync(file)) return { exists: false };
  let vertices = 0;
  let faces = 0;
  let invalidLines = 0;
  for (const line of fs.readFileSync(file, 'utf8').split(/\r?\n/)) {
    if (line.startsWith('v ')) vertices += 1;
    if (line.startsWith('f ')) {
      faces += 1;
      if (line.trim().split(/\s+/).length < 4) invalidLines += 1;
    }
  }
  return { exists: true, vertices, faces, malformed_face_lines: invalidLines, watertight: 'not_checked' };
}

function visualAssets(assetPath, assembly) {
  const candidates = ['assembly.png', 'assembly.jpg', 'assembly.webp'];
  const assemblyImage = candidates.map((name) => path.join(assetPath, name)).find((file) => fs.existsSync(file)) ?? null;
  return {
    assembly_image: assemblyImage,
    body_images: Object.entries(assembly.bodies ?? {}).map(([id, body]) => {
      const image = body.png ? path.join(assetPath, body.png) : null;
      return { body_id: id, name: body.name ?? id, image: image && fs.existsSync(image) ? image : null };
    }).filter((item) => item.image),
    instruction: 'Use read_image on assembly_image and body_images when visual product semantics are needed. Images describe appearance, not kinematic truth.',
  };
}

export function inspectAsset(dataRoot, assetId) {
  if (!/^[A-Za-z0-9_-]+$/.test(assetId)) throw new Error('asset_id contains unsupported characters');
  const assetPath = path.resolve(dataRoot, assetId);
  if (path.dirname(assetPath) !== path.resolve(dataRoot)) throw new Error('asset path escapes configured data root');
  const assemblyPath = path.join(assetPath, 'assembly.json');
  if (!fs.existsSync(assemblyPath)) throw new Error(`assembly.json not found for ${assetId}`);
  const assembly = loadJson(assemblyPath);
  const digest = crypto.createHash('sha256').update(fs.readFileSync(assemblyPath)).digest('hex');
  const features = featureRecord(assetId, assembly, digest, assetPath);
  const names = bodyNames(assembly);
  const rootBodies = assembly.root?.bodies ?? {};
  const holeEvidence = inferAxisEvidence(assembly);
  const contacts = crossBodyContacts(assembly, names);
  const bodyInventory = Object.entries(assembly.bodies ?? {}).map(([id, body]) => ({
    id,
    name: body.name ?? id,
    visible: rootBodies[id]?.is_visible ?? null,
    type: body.type ?? null,
    material: body.material?.name ?? null,
    appearance: body.appearance?.name ?? body.material?.appearance?.name ?? null,
    physical_properties: body.physical_properties ?? null,
    mesh: objQuickStats(path.join(assetPath, body.obj ?? `${id}.obj`)),
  }));
  const declaredJoints = assembly.joints ?? {};
  const occurrenceCount = Object.keys(assembly.occurrences ?? {}).length;
  const jointCount = Object.keys(declaredJoints).length;
  const cues = bodyInventory.flatMap((body) => {
    const matches = body.name.match(/hinge|lid|cover|cap|door|flap|slider|drawer|lever|handle|wheel|shaft|knob|switch|button/ig) ?? [];
    return matches.map((cue) => ({ body_id: body.id, cue: cue.toLowerCase() }));
  });
  return {
    schema_version: '1.0',
    asset_id: assetId,
    source: { assembly_json: assemblyPath, sha256: digest },
    motion_features: features,
    visual_assets: visualAssets(assetPath, assembly),
    inventory: { body_count: bodyInventory.length, occurrence_count: occurrenceCount, declared_joint_count: jointCount, hole_feature_count: (assembly.holes ?? []).length },
    bodies: bodyInventory,
    semantic_name_cues: cues,
    cross_body_contact_pairs: contacts,
    geometric_motion_hypotheses: holeEvidence,
    collision_scan: { status: 'not_run', reason: 'No validated collision backend is configured; limits must not be inferred from visual motion.' },
    video: { status: 'available_after_motion_design', reason: 'A revolute or prismatic proposal can be rendered as a synthetic preview; collision validation remains not_run.' },
    annotation_status: jointCount > 0 ? 'source_joint_data_available_review_required' : 'hypothesis_review_required',
    warnings: [
      ...(occurrenceCount === 0 ? ['no_occurrence_hierarchy; body membership cannot be assumed to equal rigid-part grouping'] : []),
      ...(jointCount === 0 ? ['no_declared_joints; inferred motion is a hypothesis, not recovered ground truth'] : []),
      ...(!bodyInventory.some((body) => body.mesh.exists) ? ['no_OBJ_meshes_found'] : []),
    ],
  };
}

export function motionFeatureBatch(dataRoot, offset = 0, limit = 5, jointFreeOnly = false, outputRoot = null) {
  const safeOffset = Math.max(0, Math.floor(offset));
  const safeLimit = Math.max(1, Math.min(20, Math.floor(limit)));
  const assetIds = fs.readdirSync(dataRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && fs.existsSync(path.join(dataRoot, entry.name, 'assembly.json')))
    .map((entry) => entry.name)
    .sort();
  const eligible = [];
  const errors = [];
  let jointFreeCount = 0;
  let jointBearingCount = 0;
  for (const assetId of assetIds) {
    try {
      const assembly = loadJson(path.join(dataRoot, assetId, 'assembly.json'));
      const hasJoints = Object.keys(assembly.joints ?? {}).length > 0;
      if (hasJoints) jointBearingCount += 1;
      else jointFreeCount += 1;
      if (jointFreeOnly && hasJoints) continue;
      eligible.push(assetId);
    } catch (error) {
      errors.push({ asset_id: assetId, error: String(error.message ?? error) });
    }
  }
  const page = eligible.slice(safeOffset, safeOffset + safeLimit).map((assetId) => {
    const assemblyPath = path.join(dataRoot, assetId, 'assembly.json');
    const assembly = loadJson(assemblyPath);
    const sourceHash = crypto.createHash('sha256').update(fs.readFileSync(assemblyPath)).digest('hex');
    const record = featureRecord(assetId, assembly, sourceHash, path.join(dataRoot, assetId));
    const designDir = outputRoot ? path.join(outputRoot, 'designs', assetId) : null;
    record.existing_design_count = designDir && fs.existsSync(designDir)
      ? fs.readdirSync(designDir).filter((name) => name.endsWith('.json')).length
      : 0;
    record.design_status = record.existing_design_count ? 'proposal_exists_check_before_revising' : 'no_proposal_yet';
    return record;
  });
  return {
    schema_version: '1.0',
    data_root: path.resolve(dataRoot),
    filters: { joint_free_only: jointFreeOnly },
    dataset_summary: { discovered_asset_count: assetIds.length, joint_free_asset_count: jointFreeCount, joint_bearing_asset_count: jointBearingCount },
    pagination: {
      offset: safeOffset,
      limit: safeLimit,
      returned: page.length,
      eligible_asset_count: eligible.length,
      next_offset: safeOffset + page.length < eligible.length ? safeOffset + page.length : null,
      complete: safeOffset + page.length >= eligible.length,
    },
    assets: page,
    parse_errors: errors,
  };
}

export function saveMotionDesign(outputRoot, dataRoot, assetId, proposalJson) {
  if (!/^[A-Za-z0-9_-]+$/.test(assetId)) throw new Error('asset_id contains unsupported characters');
  const proposal = typeof proposalJson === 'string' ? JSON.parse(proposalJson) : proposalJson;
  if (!proposal || typeof proposal !== 'object' || Array.isArray(proposal)) throw new Error('proposal must be a JSON object');
  const assetPath = path.join(path.resolve(dataRoot), assetId);
  const assemblyPath = path.join(assetPath, 'assembly.json');
  if (!fs.existsSync(assemblyPath)) throw new Error(`assembly.json not found for ${assetId}`);
  const assembly = loadJson(assemblyPath);
  const sourceHash = crypto.createHash('sha256').update(fs.readFileSync(assemblyPath)).digest('hex');
  const features = featureRecord(assetId, assembly, sourceHash);
  const bodyIds = new Set(Object.keys(assembly.bodies ?? {}));
  const requiredStrings = ['motion_type', 'motion_intent', 'rationale', 'trajectory_profile'];
  for (const field of requiredStrings) {
    if (typeof proposal[field] !== 'string' || !proposal[field].trim()) throw new Error(`proposal.${field} is required`);
  }
  if (!['revolute', 'prismatic', 'cylindrical', 'fixed_sequence', 'other'].includes(proposal.motion_type)) {
    throw new Error('unsupported motion_type');
  }
  if (!Array.isArray(proposal.moving_body_ids)) throw new Error('moving_body_ids must be an array');
  if (!Array.isArray(proposal.fixed_body_ids)) throw new Error('fixed_body_ids must be an array');
  if (proposal.motion_type !== 'other' && (!proposal.moving_body_ids.length || !proposal.fixed_body_ids.length)) {
    throw new Error('non-other motion requires fixed and moving bodies');
  }
  for (const id of [...proposal.moving_body_ids, ...proposal.fixed_body_ids]) {
    if (!bodyIds.has(id)) throw new Error(`unknown body id in proposal: ${id}`);
  }
  if (proposal.moving_body_ids.some((id) => proposal.fixed_body_ids.includes(id))) throw new Error('a body cannot be both fixed and moving');
  const range = proposal.coordinate_range;
  if (!range || !Number.isFinite(range.minimum) || !Number.isFinite(range.maximum) || range.minimum >= range.maximum) {
    throw new Error('coordinate_range requires finite minimum < maximum');
  }
  if (!['rad', 'm', 'source_unit', 'mixed'].includes(range.unit)) throw new Error('coordinate_range.unit must be rad, m, source_unit, or mixed');
  if (proposal.motion_type !== 'other' && (typeof proposal.joint_or_axis_reference !== 'string' || !proposal.joint_or_axis_reference.trim())) {
    throw new Error('joint_or_axis_reference is required; cite a source joint id or geometric evidence');
  }
  const sourceJoints = assembly.joints ?? {};
  if (Object.hasOwn(sourceJoints, proposal.joint_or_axis_reference)) {
    const sourceJoint = sourceJoints[proposal.joint_or_axis_reference];
    const kind = sourceJoint.joint_motion?.joint_type ?? '';
    const limits = /revolute/i.test(kind) ? sourceJoint.joint_motion?.rotation_limits
      : /prismatic/i.test(kind) ? sourceJoint.joint_motion?.slide_limits : null;
    if (/revolute/i.test(kind) && proposal.motion_type !== 'revolute') throw new Error('proposal motion_type conflicts with referenced source revolute joint');
    if (/prismatic/i.test(kind) && proposal.motion_type !== 'prismatic') throw new Error('proposal motion_type conflicts with referenced source prismatic joint');
    if (limits?.is_minimum_value_enabled && range.minimum < limits.minimum_value) throw new Error('coordinate_range.minimum exceeds enabled source-joint lower limit');
    if (limits?.is_maximum_value_enabled && range.maximum > limits.maximum_value) throw new Error('coordinate_range.maximum exceeds enabled source-joint upper limit');
  }
  const duration = Number.isFinite(proposal.duration_seconds) ? proposal.duration_seconds : null;
  const keyframes = proposal.trajectory_keyframes ?? [];
  if (!Array.isArray(keyframes)) throw new Error('trajectory_keyframes must be an array');
  let previousTime = -Infinity;
  for (const [index, keyframe] of keyframes.entries()) {
    if (!Number.isFinite(keyframe.time_seconds) || !Number.isFinite(keyframe.coordinate)) throw new Error(`trajectory_keyframes[${index}] requires finite time_seconds and coordinate`);
    if (keyframe.time_seconds <= previousTime || keyframe.time_seconds < 0) throw new Error('trajectory keyframe times must be nonnegative and strictly increasing');
    if (keyframe.coordinate < range.minimum || keyframe.coordinate > range.maximum) throw new Error(`trajectory_keyframes[${index}].coordinate is outside coordinate_range`);
    previousTime = keyframe.time_seconds;
  }
  if (proposal.motion_type !== 'other' && keyframes.length < 2) throw new Error('moving designs require at least two trajectory_keyframes');
  if (duration !== null && (duration <= 0 || previousTime > duration)) throw new Error('duration_seconds must be positive and cover all trajectory keyframes');
  const unresolved = features.joint_count === 0 || range.unit === 'source_unit' || range.unit === 'mixed';
  const design = {
    schema_version: '1.0',
    design_id: `${assetId}:${randomUUID()}`,
    asset_id: assetId,
    source_sha256: features.source_sha256,
    created_at: new Date().toISOString(),
    designed_by: 'dsh_llm_proposal',
    motion_type: proposal.motion_type,
    motion_intent: proposal.motion_intent,
    rationale: proposal.rationale,
    fixed_body_ids: [...new Set(proposal.fixed_body_ids)],
    moving_body_ids: [...new Set(proposal.moving_body_ids)],
    unassigned_body_ids: [...bodyIds].filter((id) => !proposal.moving_body_ids.includes(id) && !proposal.fixed_body_ids.includes(id)),
    joint_or_axis_reference: proposal.joint_or_axis_reference ?? (proposal.motion_type === 'other' ? 'insufficient evidence' : null),
    axis_origin: proposal.axis_origin ?? null,
    axis_direction: proposal.axis_direction ?? null,
    coordinate_range: range,
    trajectory_profile: proposal.trajectory_profile,
    duration_seconds: duration,
    trajectory_keyframes: keyframes,
    repetitions: Number.isInteger(proposal.repetitions) && proposal.repetitions > 0 ? proposal.repetitions : 1,
    evidence: Array.isArray(proposal.evidence) ? proposal.evidence : [],
    assumptions: Array.isArray(proposal.assumptions) ? proposal.assumptions : [],
    confidence: Number.isFinite(proposal.confidence) ? Math.max(0, Math.min(1, proposal.confidence)) : null,
    source_joint_count: features.joint_count,
    source_units: { assembly_coordinate_unit: 'not declared in assembly.json', proposed_coordinate_unit: range.unit },
    status: unresolved ? 'proposal_requires_unit_and_geometry_review' : 'proposal_requires_collision_and_human_review',
    collision_validation: 'not_run',
    video_generation: 'not_enabled',
    ground_truth: false,
  };
  const designDir = path.join(outputRoot, 'designs', assetId);
  fs.mkdirSync(designDir, { recursive: true });
  const target = path.join(designDir, `${design.design_id.replaceAll(':', '_')}.json`);
  fs.writeFileSync(target, `${JSON.stringify(design, null, 2)}\n`, { flag: 'wx' });
  return { path: target, status: design.status, design };
}

export function rankAssets(dataRoot, limit = 50) {
  const ids = fs.readdirSync(dataRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && fs.existsSync(path.join(dataRoot, entry.name, 'assembly.json')))
    .map((entry) => entry.name)
    .sort();
  const assets = ids.map((id) => {
    try {
      const assembly = loadJson(path.join(dataRoot, id, 'assembly.json'));
      const bodies = bodyNames(assembly);
      const cueCount = Object.values(bodies).reduce((count, name) => count + (/hinge|lid|cover|cap|door|flap|slider|drawer|lever|handle|wheel|shaft|knob|switch|button/i.test(name) ? 1 : 0), 0);
      const jointCount = Object.keys(assembly.joints ?? {}).length;
      const occurrenceCount = Object.keys(assembly.occurrences ?? {}).length;
      const contactPairs = crossBodyContacts(assembly, bodies).length;
      const holeCount = (assembly.holes ?? []).length;
      const score = (jointCount === 0 ? 20 : 0) + Math.min(cueCount, 3) * 3 + Math.min(contactPairs, 3) + (holeCount > 1 ? 2 : 0);
      return { asset_id: id, score, joint_count: jointCount, occurrence_count: occurrenceCount, cross_body_contact_pair_count: contactPairs, hole_feature_count: holeCount, semantic_cue_count: cueCount, status: jointCount > 0 ? 'joint_reference_available_calibration_asset' : 'hypothesis_review_candidate' };
    } catch (error) {
      return { asset_id: id, score: -1, status: 'inspection_failed', error: String(error.message ?? error) };
    }
  });
  return { schema_version: '1.0', data_root: path.resolve(dataRoot), summary: { asset_count: ids.length, ranked_count: Math.min(Math.max(1, limit), assets.length) }, assets: assets.sort((a, b) => b.score - a.score || a.asset_id.localeCompare(b.asset_id)).slice(0, Math.max(1, limit)) };
}

export function createReviewTask(outputRoot, report, note = '') {
  fs.mkdirSync(outputRoot, { recursive: true });
  const reviewDir = path.join(outputRoot, 'reviews', report.asset_id);
  fs.mkdirSync(reviewDir, { recursive: true });
  const target = path.join(reviewDir, 'task.json');
  const task = {
    schema_version: '1.0',
    task_id: `${report.asset_id}:motion-review`,
    asset_id: report.asset_id,
    input_sha256: report.source.sha256,
    status: 'pending_human_review',
    created_at: new Date().toISOString(),
    proposed_motion_hypotheses: report.geometric_motion_hypotheses,
    collision_scan: report.collision_scan,
    decision: null,
    decision_history: [],
    notes: note,
  };
  fs.writeFileSync(target, `${JSON.stringify(task, null, 2)}\n`, { flag: 'wx' });
  return { task_path: target, task_id: task.task_id, status: task.status };
}

export function saveReview(outputRoot, assetId, decision, reviewer, notes = '') {
  if (!/^[A-Za-z0-9_-]+$/.test(assetId)) throw new Error('asset_id contains unsupported characters');
  if (!['approved_hypothesis', 'rejected', 'needs_more_evidence'].includes(decision)) throw new Error('unsupported review decision');
  if (!reviewer.trim()) throw new Error('reviewer is required');
  const taskPath = path.join(outputRoot, 'reviews', assetId, 'task.json');
  const task = loadJson(taskPath);
  if (task.asset_id !== assetId) throw new Error('task asset_id mismatch');
  task.decision = { value: decision, reviewer: reviewer.trim(), notes, reviewed_at: new Date().toISOString() };
  task.decision_history ??= [];
  task.decision_history.push(task.decision);
  task.status = decision === 'approved_hypothesis' ? 'hypothesis_approved_not_ground_truth' : decision;
  fs.writeFileSync(taskPath, `${JSON.stringify(task, null, 2)}\n`);
  return { task_id: task.task_id, status: task.status, decision: task.decision };
}
