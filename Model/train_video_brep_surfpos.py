"""Overfit official BrepGen surface-position diffusion to one STEP part/video.

This is a staged training check, not a STEP exporter: surface/edge VAE and
the other three pretrained diffusion branches are required for valid B-reps.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from diffusers import DDPMScheduler
from network import SurfPosNet

from .model import DualViewFusion, VideoJEPAEncoder
from .video import load_video
from .video_brep_condition import VideoConditionedBrepGen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-a", required=True)
    parser.add_argument("--video-b", required=True)
    parser.add_argument("--target", required=True, help="BrepGen parsed single-part pickle")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--surfpos-weight", help="Official pretrained DeepCAD surface-position checkpoint")
    args = parser.parse_args()
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with Path(args.target).open("rb") as stream:
        target = pickle.load(stream)
    bbox = np.asarray(target["surf_bbox_wcs"], dtype=np.float32)
    if not 0 < len(bbox) <= 30:
        raise ValueError(f"target has {len(bbox)} faces; DeepCAD supports at most 30")
    bbox = np.concatenate((bbox, np.repeat(bbox[-1:], 30 - len(bbox), axis=0))) * 3.0
    clean = torch.from_numpy(bbox).unsqueeze(0).to(device)

    preprocessor = torch.hub.load("/root/vjepa2", "vjepa2_preprocessor", source="local", pretrained=False, crop_size=384)
    encoder = VideoJEPAEncoder(num_frames=args.frames).to(device).eval()
    with torch.no_grad():
        video_a = preprocessor(load_video(args.video_a, args.frames))[0].unsqueeze(0).to(device)
        video_b = preprocessor(load_video(args.video_b, args.frames))[0].unsqueeze(0).to(device)
        tokens_a, _ = encoder(video_a)
        tokens_b, _ = encoder(video_b)
    del encoder, video_a, video_b
    fusion = DualViewFusion(192).to(device)
    # Only surfpos is instantiated here to leave GPU memory for its optimizer.
    denoiser = SurfPosNet(False).to(device)
    if args.surfpos_weight:
        denoiser.load_state_dict(torch.load(args.surfpos_weight, map_location="cpu", weights_only=False), strict=True)
    model = VideoConditionedBrepGen({"surfpos": denoiser},
                                    required_branches=("surfpos",)).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                              prediction_type="epsilon", beta_start=1e-4, beta_end=0.02,
                              clip_sample=False)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(fusion.parameters()), lr=args.lr)
    history = []
    for step in range(1, args.steps + 1):
        video_tokens = fusion(tokens_a, tokens_b)
        timestep = torch.randint(0, 1000, (1,), device=device)
        noise = torch.randn_like(clean)
        predicted = model("surfpos", video_tokens, scheduler.add_noise(clean, noise, timestep), timestep, None)
        loss = (predicted - noise).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(model.parameters()) + list(fusion.parameters()), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
        if step == 1 or step % max(1, args.steps // 10) == 0:
            print(f"step={step} noise_loss={history[-1]:.6f}", flush=True)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"condition": model.state_dict(),
                "fusion": fusion.state_dict(), "target_uid": target["uid"],
                "pretrained": bool(args.surfpos_weight), "args": vars(args)}, output / "surfpos_conditioned.pt")
    (output / "history.json").write_text(json.dumps(history) + "\n")
    print(f"saved {output / 'surfpos_conditioned.pt'}", flush=True)


if __name__ == "__main__":
    main()
