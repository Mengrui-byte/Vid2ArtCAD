---
name: vid2artcad-annotation
description: Run the Vid2ArtCAD CAD-motion annotation workflow, review kinematic previews, and package accepted minimal two-view datasets.
---

# Vid2ArtCAD Annotation

Use this skill when auditing or annotating the Fusion CAD assets in `a1.0.0_00`, or when packaging accepted Vid2ArtCAD samples.

## Workflow

1. Audit the source before generating annotations:

   ```bash
   python3 -m pipeline.render.dataset_audit --data-root a1.0.0_00 --output data/work/dataset_audit.json
   ```

   Report asset, occurrence, joint-type, and missing-geometry counts. Assets without occurrences do not have a known part-instance graph. Assets without source joints need a separate evidence-based motion proposal; do not invent ground-truth joints.

2. Run the deterministic kinematic preview only for explicit candidate IDs:

   ```bash
   OPENBLAS_NUM_THREADS=1 python3 -B -m pipeline.render.auto_annotation \
     --data-root a1.0.0_00 --output-root data/work/annotation \
     --asset <asset-id> --preview
   ```

   Inspect `previews/references/<asset-id>/contact_sheet.png`, `quality.json`, and the motion evidence. The first run needs an approved sample `scene.xml` available under the chosen output root; use a validated annotation sample as the scene template when bootstrapping a new work root.

3. Record an explicit preview decision using the installed `vid2artcad preview-review` command. Approve only motion that is visible, coherent with the source kinematics, and useful for the product. Reject or defer ambiguous motion. Approval is bound to hashes of the source and preview evidence; regenerate and review if those change.

4. Run formal rendering only for assets with fresh approved previews. The renderer creates synchronized 360-frame, 30-FPS views and a validated sample. Keep intermediate references, previews, logs, and reports under `data/work/annotation`; do not publish them as final samples.

5. Package only samples accepted by the dataset manifest:

   ```bash
   vid2artcad package-minimal --manifest-root <accepted-release> \
     --cad-root a1.0.0_00 \
     --destination /Users/mengrui/Desktop/ArtCAD-dataset
   ```

   Final output contains `dataset_manifest.json` and one directory per accepted asset. Each asset contains exactly two videos (`video_cam01.mp4`, `video_cam02.mp4`), `assembly.urdf`, `assembly.step`, and only the mesh files referenced by the URDF. The destination may already exist only when it is empty; never overwrite a non-empty release.

## Evidence Rules

- Source joint definitions and enabled limits are evidence. Proposed animation amplitudes are not mechanical limits.
- DSH or LLM product/function descriptions are advisory. They cannot establish a joint, limit, collision-free path, or ground truth without source or validation evidence.
- A preview is not a validated final sample. Keep rejected, blocked, or uncertain assets out of the published dataset and state why.
- The compact delivery dataset is external to this Git repository. Do not commit source CAD data, generated videos, meshes, or release archives.
