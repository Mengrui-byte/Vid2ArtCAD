---
name: cad-motion-annotation
description: Analyze every CAD asset's kinematic and geometric features, design a meaningful product motion, and persist a structured motion proposal for validation.
---

# CAD Product Motion Design

Your goal is to use the source CAD evidence to design a plausible, meaningful product motion for each asset. You are the motion-design agent: inspect the supplied feature records, reason about product function, select the fixed and moving bodies, and propose a parameterized motion. Do not stop after ranking candidates.

## Dataset pass

1. Call `cad_annotation_get_motion_feature_batch` with `offset: 0`, `limit: 5`, and `joint_free_only: false`.
2. For each returned asset, analyze its `joints`, `occurrences`, `bodies`, `components`, `holes`, `contact_pairs`, and `inferred_axis_hypotheses`. Use source joint definitions first. For joint-free assets, combine feature geometry and part semantics; do not treat a geometric hypothesis as a confirmed joint.
3. If `visual_assets.assembly_image` or a body image is available, call `read_image` on it. Use appearance to identify product category and functional part roles, but never use appearance alone to infer a joint or limit.
4. For each asset, call `cad_annotation_save_motion_design` with one coherent motion proposal. For inferred axes, include `axis_origin` and `axis_direction`. For a multi-stage product action, create a separate proposal per degree of freedom/action and cross-reference them in `rationale`.
5. Continue with the returned `next_offset` until `pagination.complete` is true. Use `joint_free_only: true` for a dedicated pass over assets without source joints. Do not claim dataset coverage until every page has been processed; report parse failures separately.
6. Assets with insufficient evidence still need a saved design record using `motion_type: "other"`, a clear `motion_intent` such as `insufficient evidence to propose motion`, and assumptions describing what is missing. Do not invent a moving part merely to fill the dataset.
7. After saving a `revolute` or `prismatic` proposal, call `cad_annotation_render_motion_video` to generate a synthetic design preview. Do not render an `other` record with no moving body. Label every returned MP4 as a preview; it is not collision validated.

Do not use `cad_annotation_rank_assets` as the main loop. It is only for selecting an inspection subset. Dataset annotation means paginating through all feature batches and saving a proposal or an explicit insufficient-evidence record for every asset.

## Proposal fields

Supply `proposal_json` with:

- `motion_type`: `revolute`, `prismatic`, `cylindrical`, `fixed_sequence`, or `other`.
- `motion_intent`: the product-level purpose, such as opening for access or clamping a workpiece.
- `rationale`: why this motion is meaningful for this product and how the parts interact.
- `fixed_body_ids` and `moving_body_ids`: IDs from the feature record; avoid overlap.
- `joint_or_axis_reference`: source joint ID or the geometric evidence used for an inferred axis.
- `axis_origin` and `axis_direction`: required when `joint_or_axis_reference` is a geometric hypothesis rather than a source joint.
- `coordinate_range`: `{ "minimum": number, "maximum": number, "unit": "rad" | "m" | "source_unit" | "mixed" }`.
- `trajectory_profile`: a named, smooth motion policy (for example `smooth_open_hold_close`).
- `duration_seconds`, `repetitions`, `evidence`, `assumptions`, and `confidence` (0-1) when available.

## Final delivery contract

The only final dataset location is `data/published` under the Vid2ArtCAD project.
Each asset directory contains exactly two videos (`video_cam01.mp4` and
`video_cam02.mp4`), `assembly.urdf`, `assembly.step`, and the local `meshes/`
files referenced by that URDF. A dataset-level manifest records source paths
and SHA-256 checksums; per-frame depth, masks, previews, logs, and intermediate
design JSON belong to working storage, not the delivery directory.

Use the deterministic `vid2artcad package-minimal` CLI after source annotation
acceptance. Missing STEP, video, or referenced mesh files must block publishing.
A saved design proposal or a single-view video preview is not a complete dataset
sample. Do not claim a preview has been exported to URDF or passed validation.

## Evidence policy

- Source joints and enabled limits are source evidence. Preserve their exact source values; the dataset may not declare units.
- Inferred holes, shafts, contacts, names, and geometry are supporting evidence, not ground truth. Contacts may be unrelated to the intended mechanism.
- Missing occurrences mean rigid-body grouping is unknown. State that assumption and lower confidence.
- A designed coordinate range is a proposed animation/design range, not a mechanical limit. Collision validation is currently `not_run`; generated video is a synthetic design preview, not a collision-free result.
- Every saved proposal remains `ground_truth: false` and requires geometry/collision validation and human review before publication.
- Do not record a human review decision or impersonate a reviewer.
