#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${VJEPA_ENV:-/root/v2env}"
REPO_DIR="${VJEPA_REPO:-/root/vjepa2}"
CACHE_DIR="${VJEPA_CACHE:-/root/.cache/vjepa2}"
CHECKPOINT="$CACHE_DIR/vjepa2_1_vitb_dist_vitG_384.pt"
CHECKPOINT_URL="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"

test -x "$ENV_DIR/bin/python" || { echo "Missing environment: $ENV_DIR" >&2; exit 2; }
mkdir -p "$CACHE_DIR"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/facebookresearch/vjepa2.git "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --ff-only
fi

"$ENV_DIR/bin/python" -m pip install pyyaml einops beartype iopath timm decord
wget -c "$CHECKPOINT_URL" -O "$CHECKPOINT"

"$ENV_DIR/bin/python" - "$REPO_DIR" "$CHECKPOINT" <<'PY'
import sys
import torch

repo, checkpoint = sys.argv[1:]
model, _ = torch.hub.load(repo, "vjepa2_1_vit_base_384", source="local", pretrained=False, num_frames=16)
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
state = payload.get("ema_encoder", payload.get("target_encoder", payload.get("encoder", payload.get("model", payload))))
for prefix in ("module.", "backbone.", "encoder."):
    state = {(key[len(prefix):] if key.startswith(prefix) else key): value for key, value in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
print(type(model).__name__, "parameters=", sum(p.numel() for p in model.parameters()))
print("missing=", len(missing), "unexpected=", len(unexpected))
print("cuda=", torch.cuda.is_available())
PY

echo "Ready: $CHECKPOINT"
