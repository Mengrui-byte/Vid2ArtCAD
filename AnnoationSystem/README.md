# Vid2ArtCAD DSH Annotation Agent

An independent DSH Harness plugin bundle for LLM-assisted product-motion design from CAD assets. DSH provides the agent runtime; deterministic tools parse each asset's source kinematics and geometry, while the model reasons about product intent and proposes a parameterized motion design.

## Current scope

Final deliverables use the project-level `vid2artcad package-minimal` stage:
two videos, `assembly.urdf`, `assembly.step`, and the URDF-referenced `meshes/`
per asset, plus one dataset manifest. See `../pipeline/render/minimal_dataset.py`.
The agent's saved designs and single-view previews are intermediate evidence,
not complete published samples. This exporter does not generate missing URDFs,
STEP files, or second views, and does not certify collision-free motion.

- Paginates through every asset, optionally restricting the pass to assets without declared joints.
- Exposes normalized source joint types, limits (including enabled flags and unknown units), coordinate-frame axes, occurrences, components, holes, contacts, and inferred axis hypotheses to the model.
- Exposes assembly/body images plus names, material, appearance, physical properties, holes, contacts, and source joints so the model can form a product-identity hypothesis before designing motion.
- Lets the model propose fixed/moving bodies, motion intent, degree of freedom, coordinate range, trajectory profile, and time-coordinate keyframes.
- Persists proposals with source hashes and explicit validation status under `AnnoationSystem/output/designs/`.
- Renders saved revolute/prismatic proposals as synthetic MP4 previews under `AnnoationSystem/output/videos/`; previews are not collision validation or mechanical ground truth.

## Installed DSH Runtime

DSH `0.2.1-rc.2` is installed locally through the root `package.json` and pinned by `package-lock.json`. No global installation is required. Node.js 24.15.0 was used for verification.

```sh
cd /Users/mengrui/Desktop/Vid2ArtCAD/AnnoationSystem
npm ci                    # reinstall the pinned runtime if needed
npm run server:start      # background local server, defaults to port 3080
npm run server:status     # prints the current token-bearing browser URL
npm run smoke             # requires the running server and local dataset
npm run server:stop
```

The renderer uses the root Python environment. Install its dependencies once with
`uv pip install --python ../.venv/bin/python -r requirements-render.txt`.

Use `npm start` for foreground operation, or `ANNOATION_PORT=3081 npm run server:start` when port 3080 is occupied. The server binds only to `127.0.0.1`. The access token is a credential: do not share it or expose the service to a network.

`scripts/dsh.mjs` generates a project-local overlay and makes **CAD Motion Designer** the default agent preset. Its persona embeds the motion-design skill, and its only model-callable tools are the six CAD tools plus read-only image access. The preset does not expose general shell or file-edit tools. Other shipped DSH presets remain selectable; switching presets can change the available permissions/tools.

Sessions, credentials and logs use the ignored `.dsh/` directory. Session telemetry is disabled. The launcher uses `/Users/mengrui/Desktop/Vid2ArtCAD` as the DSH workspace root so the agent can read both `AnnoationSystem` and the sibling dataset images. DSH is developer-preview software, not a security boundary; plugin filesystem access is still ordinary host-process access.

Optional paths must be exported before starting the service:

```sh
export ANNOATION_DATA_ROOT=/absolute/path/to/a1.0.0_00
export ANNOATION_OUTPUT_ROOT=/absolute/path/to/AnnoationSystem/output
```

## Model Authentication

Configure a model connection through the DSH Web settings using your own DeepSeek credentials/account, or export `DEEPSEEK_API_KEY` in your local shell before starting the server. Never commit credentials or put keys into prompts. The install-time environment had no API key, so model inference and model-driven tool dispatch have **not** been tested.

After authentication, begin with this single-asset request before a dataset batch:

```text
Use the CAD tools to inspect asset 21634_e5e20375. Use read_image on the assembly
image, identify the product and parts, save one evidence-based motion proposal, then
render a synthetic preview. State uncertainty, keep collision_validation as not_run,
and do not claim a mechanical limit.
```

The smoke test checks authenticated HTTP access, default-preset activation, and a local call to the plugin built with the official DSH tool API. It does not exercise an LLM turn or tool execution through the DSH agent loop.

## Agent contract

The model should execute this loop, not call only the ranking tool:

```text
offset = 0
repeat:
  page = cad_annotation_get_motion_feature_batch(offset=offset, limit=5)
  for asset in page.assets:
    inspect asset.joints, occurrences, bodies, components, holes, contacts, inferred_axis_hypotheses
    design one meaningful product motion or explicitly record insufficient evidence
    cad_annotation_save_motion_design(asset_id, proposal_json)
  offset = page.pagination.next_offset
until page.pagination.complete == true
```

For joint-free assets, use a second pass with `joint_free_only=true`. Each page contains `dataset_summary` and the returned assets contain `source_sha256`, so the agent can report coverage and reproduce which source was analyzed. Existing designs are surfaced as `design_status`; the agent must inspect them before creating a revision.

## Tools

- `cad_annotation_get_motion_feature_batch`: return parsed motion features from a page of assets; continue with the returned offset until complete.
- `cad_annotation_inspect_asset`: inspect one asset and return detailed source evidence and readiness gates.
- `cad_annotation_save_motion_design`: validate required fields/body IDs/keyframes and persist the model's proposal. Does not simulate it.
- `cad_annotation_render_motion_video`: render a saved revolute/prismatic proposal to MP4; output is a synthetic preview with collision validation marked `not_run`.
- `cad_annotation_rank_assets`: rank assets for triage; not a substitute for processing every page.
- `cad_annotation_create_review_task`: persist a pending human-review task.

Human decisions are recorded with the local CLI, not an LLM-callable tool, so the agent cannot impersonate a reviewer:

```sh
node dsh-plugin/src/cli.mjs review --output-root output --asset ASSET_ID \\
  --decision approved_hypothesis --reviewer REVIEWER_ID --notes "Evidence reviewed"
```

## Local tests

Requires Node.js 20 or later:

```sh
node --test tests/*.test.mjs
node dsh-plugin/src/cli.mjs inspect --data-root ../a1.0.0_00 --asset 21634_e5e20375
```

No model API key is needed for deterministic tests. For a full dataset pass, configure DSH with a model provider and instruct the agent to use the `cad-motion-annotation` skill. The agent must page until `pagination.complete`, save one motion design per asset, and report parse failures and low-evidence assets. Agent behavior depends on the DSH runtime and configured model provider.
