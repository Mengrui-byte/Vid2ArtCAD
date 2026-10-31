# Dual-view Video to OBJ and URDF

The overfit model has four components:

1. Two independent `VideoJEPAEncoder` instances load Meta's official V-JEPA
   2.1 ViT-B/16 checkpoint. Cam01 and cam02 are encoded separately.
2. `DualViewFusion` adds view embeddings and fuses both token sequences with a
   Transformer encoder.
3. `OBJVertexDecoder` predicts fixed-topology vertices for every assembly link.
4. `KinematicsBranch` predicts link presence, parent, joint type, axis, origin,
   and limits; inference writes these predictions as a URDF.
The file still contains `BrepGenStructuredDiffusion`: a video-conditioned Transformer denoiser over
   hierarchical CAD nodes. Node features contain a primitive bounding box and
   a local shape latent; node classes are solid, face, edge, or vertex, with
   parent/incidence references. This branch follows the BrepGen structured
   latent diffusion approach, not the previous bbox-only prediction head.
It is not optimized in the OBJ/URDF overfit command because this sample has no
B-rep supervision.

## B-rep Training Data

The repository currently has OBJ meshes and URDF kinematics for the sample,
but no STEP/B-rep files, BrepGen-processed face/edge features, or BrepGen VAE
checkpoint. OBJ bounds are not valid B-rep supervision. Therefore the model
does not claim to export a STEP/B-rep from the current sample yet.

The old 800-step bbox experiment did not use V-JEPA weights and is not a
V-JEPA/BrepGen result. The current OBJ run uses the official pretrained
V-JEPA 2 video encoder, but uses fixed input mesh topology and does not yet
generate B-rep topology.

Prepare one fixed-size target archive with the official BrepGen preprocessing
and geometry VAEs. The archive format expected by the overfit script is:

```text
node_features: float32 [max_nodes, 6 + 48]
  first 6 values: normalized primitive bbox
  remaining 48 values: local geometry latent, padded for node types
node_type: int64 [max_nodes]       # solid=0, face=1, edge=2, vertex=3, pad=4
parent: int64 [max_nodes]          # parent node index; root/pad class=max_nodes
node_mask: bool [max_nodes]
```

The `48` geometry channels are an adapter target size, not a promise that an
arbitrary archive is directly compatible with BrepGen's separately trained
face/edge VAEs. A final CAD export still requires loading the matching official
face/edge VAE decoders and BrepGen topology reconstruction/postprocessing.

## Remote Setup and Overfit Command

Download the official small checkpoint and repository on the server:

```bash
cd /root/Vid2ArtCAD
bash Model/download_vjepa_remote.sh
```

For an OBJ-only overfit run, first pack the existing URDF-linked mesh targets:

```bash
python -m Model.prepare_obj_target \
  --urdf annoation/22218_2f85f6ae/assembly.urdf \
  --out /tmp/assembly_mesh_target.npz

/root/v2env/bin/python -m Model.train_overfit \
  --video-a annoation/22218_2f85f6ae/rgb_cam01.mp4 \
  --video-b annoation/22218_2f85f6ae/rgb_cam02.mp4 \
  --urdf annoation/22218_2f85f6ae/assembly.urdf \
  --mesh-target /tmp/assembly_mesh_target.npz \
  --out output/dual_vjepa_obj_urdf --steps 800 --frames 16 \
  --vjepa-repo /root/vjepa2 \
  --vjepa-checkpoint /root/.cache/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
```

Install the encoder runtime first with `pip install -e '.[vjepa]'`. On first
run, Transformers downloads the official checkpoint and processor from the
Hugging Face model repository and caches them in the standard HF cache (or the
directory passed to `--vjepa-cache`). Use a CUDA-enabled PyTorch environment
for the video feature extraction.

Outputs are `generated_assembly.obj`, `parts/*.obj`, `generated_assembly.urdf`,
`kinematic_tree.json`, and `checkpoint.pt`. This OBJ experiment learns vertex
positions conditioned on both videos while retaining the training sample's
triangle connectivity. It is
an overfit reconstruction test, not topology generation or generalization to
unseen assemblies. The separate BrepGen diffusion branch remains available
for B-rep latent training when BrepGen-encoded targets and its matching VAE
decoders are supplied. Do not use the old bbox checkpoint with this model.
