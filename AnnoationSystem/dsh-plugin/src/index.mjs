import { defineTool } from '@deepseek-ai/dsh-tools'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { createReviewTask, inspectAsset, motionFeatureBatch, rankAssets, saveMotionDesign } from './engine.mjs'

const execFileAsync = promisify(execFile)

export const name = 'vid2artcad-annotation-tools'
export const inject = ['tools']

const dataRoot = process.env.ANNOATION_DATA_ROOT
  ?? fileURLToPath(new URL('../../../a1.0.0_00', import.meta.url))
const outputRoot = process.env.ANNOATION_OUTPUT_ROOT
  ?? fileURLToPath(new URL('../../output', import.meta.url))

function asText(value) {
  return JSON.stringify(value, null, 2)
}

export function apply(ctx) {
  ctx.tools.register(defineTool({
    name: 'cad_annotation_render_motion_video',
    description: 'Render a saved motion design as an MP4 preview from the asset OBJ meshes. This is a synthetic design preview only: collision_validation remains not_run and the result is not a mechanical limit or ground truth.',
    parameters: {
      asset_id: { type: 'string', required: true, description: 'Asset directory name.' },
      design_path: { type: 'string', required: true, description: 'Path returned by cad_annotation_save_motion_design.' },
      frames: { type: 'number', description: 'Frame count, default 60.' },
      fps: { type: 'number', description: 'Frames per second, default 20.' },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args) {
      if (!/^[A-Za-z0-9_-]+$/.test(args.asset_id)) throw new Error('asset_id contains unsupported characters')
      const designPath = path.resolve(String(args.design_path))
      const designRoot = path.resolve(outputRoot, 'designs')
      if (designPath !== designRoot && !designPath.startsWith(`${designRoot}${path.sep}`)) {
        throw new Error('design_path must point to a saved file under the configured output/designs directory')
      }
      const output = path.join(outputRoot, 'videos', args.asset_id, `${Date.now()}_design_preview.mp4`)
      const assetPath = path.join(dataRoot, args.asset_id)
      const python = process.env.ANNOATION_PYTHON ?? fileURLToPath(new URL('../../../.venv/bin/python', import.meta.url))
      const { stdout } = await execFileAsync(python, [
        fileURLToPath(new URL('../../scripts/render_motion_video.py', import.meta.url)),
        '--asset', assetPath, '--design', designPath, '--output', output,
        '--frames', String(Math.max(10, Math.min(300, Math.floor(args.frames ?? 60)))),
        '--fps', String(Math.max(1, Math.min(60, Math.floor(args.fps ?? 20)))),
      ], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 4 })
      return stdout.trim()
    },
  }))

  ctx.tools.register(defineTool({
    name: 'cad_annotation_rank_assets',
    description: 'Rank CAD assets for motion-annotation review using declared joints, semantic body names, cross-body contacts, and aligned hole axes. This ranks review candidates only; it does not establish a joint or motion limit.',
    parameters: {
      limit: { type: 'number', description: 'Maximum results, from 1 to 200.' },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args) {
      const limit = Math.max(1, Math.min(200, Math.floor(args.limit ?? 50)))
      return asText(rankAssets(dataRoot, limit))
    },
  }))

  ctx.tools.register(defineTool({
    name: 'cad_annotation_get_motion_feature_batch',
    description: 'Read parsed kinematic, geometric, material, physical, and visual evidence for a page of source CAD assets. Use offset pagination until complete to cover the dataset. Returns source joints/limits, component and occurrence structure, body names, holes, contacts, image paths, and explicitly labeled inferred axis hypotheses.',
    parameters: {
      offset: { type: 'number', description: 'Zero-based offset from the sorted asset list.' },
      limit: { type: 'number', description: 'Page size from 1 to 20; default 5.' },
      joint_free_only: { type: 'boolean', description: 'If true, include only assets with no declared source joints.' },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args) {
      return asText(motionFeatureBatch(dataRoot, args.offset ?? 0, args.limit ?? 5, args.joint_free_only ?? false, outputRoot))
    },
  }))

  ctx.tools.register(defineTool({
    name: 'cad_annotation_save_motion_design',
    description: 'Persist the LLM-designed motion proposal for an asset. Supply proposal_json with motion_type, motion_intent, rationale, fixed_body_ids, moving_body_ids, coordinate_range {minimum, maximum, unit}, trajectory_profile, and trajectory_keyframes [{time_seconds, coordinate}]. Saves a proposal only; it is not collision-validated, executable, or ground truth.',
    parameters: {
      asset_id: { type: 'string', required: true, description: 'Asset directory name under the configured dataset root.' },
      proposal_json: { type: 'string', required: true, description: 'JSON object describing one product motion/degree of freedom.' },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args) {
      return asText(saveMotionDesign(outputRoot, dataRoot, args.asset_id, args.proposal_json))
    },
  }))

  ctx.tools.register(defineTool({
    name: 'cad_annotation_inspect_asset',
    description: 'Inspect one CAD asset and return auditable part inventory, mesh file statistics, contact pairs, hole-axis motion hypotheses, and explicit collision/video readiness. Hypotheses are not ground truth.',
    parameters: {
      asset_id: { type: 'string', required: true, description: 'Asset directory name under the configured dataset root.' },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args) {
      return asText(inspectAsset(dataRoot, args.asset_id))
    },
  }))

  ctx.tools.register(defineTool({
    name: 'cad_annotation_create_review_task',
    description: 'Create an immutable-input, pending-human-review task for an asset inspection. Does not approve or publish an annotation.',
    parameters: {
      asset_id: { type: 'string', required: true, description: 'Asset directory name under the configured dataset root.' },
      note: { type: 'string', description: 'Optional reason or review focus.' },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args) {
      const report = inspectAsset(dataRoot, args.asset_id)
      return asText(createReviewTask(outputRoot, report, args.note ?? ''))
    },
  }))

}
