#!/usr/bin/env bash
set -u

cd /root/Vid2ArtCAD
source /root/v2env/bin/activate

DATA_ROOT=/root/Vid2ArtCAD/training_data
TARGET_ROOT=/root/Vid2ArtCAD/training_targets_fixed_scale
OUTPUT_ROOT=/root/Vid2ArtCAD/output/batch_overfit_fixed_scale
mkdir -p "$TARGET_ROOT" "$OUTPUT_ROOT"

for sample_dir in "$DATA_ROOT"/*_multiview; do
  [ -d "$sample_dir" ] || continue
  sample_id=$(basename "$sample_dir")
  urdf="$sample_dir/assembly.urdf"
  target="$TARGET_ROOT/$sample_id.npz"
  output="$OUTPUT_ROOT/$sample_id"
  log="$output/train.log"
  mkdir -p "$output"

  if [ ! -f "$urdf" ] || [ ! -f "$sample_dir/rgb_cam01.mp4" ] || [ ! -f "$sample_dir/rgb_cam02.mp4" ]; then
    echo "[$sample_id] missing required input" | tee "$log"
    continue
  fi

  max_parts=$(python - "$urdf" <<'PY'
import sys
import xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
count = sum(1 for node in root.findall("link") if node.attrib.get("name") != "world")
print(max(16, count))
PY
  )

  read -r max_vertices max_faces <<< "$(python - "$urdf" <<'PY'
import sys
import xml.etree.ElementTree as ET
import trimesh
root = ET.parse(sys.argv[1]).getroot()
max_v, max_f = 1, 1
base = __import__('pathlib').Path(sys.argv[1]).parent
for link in root.findall("link"):
    mesh_node = link.find("./visual/geometry/mesh")
    if mesh_node is None or not mesh_node.attrib.get("filename"):
        continue
    path = (base / mesh_node.attrib["filename"]).resolve()
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
        max_v = max(max_v, len(mesh.vertices))
        max_f = max(max_f, len(mesh.faces))
    except Exception:
        pass
print(max(4096, max_v), max(8192, max_f))
PY
)"

  if [ ! -f "$target" ]; then
    echo "[$sample_id] preparing target max_parts=$max_parts" | tee "$log"
    if ! python -m Model.prepare_obj_target --urdf "$urdf" --out "$target" --max-parts "$max_parts" --max-vertices "$max_vertices" --max-faces "$max_faces" --allow-missing-visual >>"$log" 2>&1; then
      echo "[$sample_id] target preparation failed" | tee -a "$log"
      continue
    fi
  fi

  if [ -f "$output/generated_assembly.urdf" ] && [ -f "$output/checkpoint.pt" ]; then
    echo "[$sample_id] already complete" | tee -a "$log"
    continue
  fi

  echo "[$sample_id] training max_parts=$max_parts" | tee -a "$log"
  python -m Model.train_overfit \
    --video-a "$sample_dir/rgb_cam01.mp4" \
    --video-b "$sample_dir/rgb_cam02.mp4" \
    --urdf "$urdf" \
    --mesh-target "$target" \
    --out "$output" \
    --steps 800 \
    --frames 16 \
    --max-parts "$max_parts" \
    --vjepa-repo /root/vjepa2 \
    --vjepa-checkpoint /root/.cache/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt \
    >>"$log" 2>&1
  status=$?
  echo "[$sample_id] exit=$status" | tee -a "$log"
done

echo "batch complete"
