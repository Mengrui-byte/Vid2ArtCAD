"""Small video loader for the single-video overfit test."""

from __future__ import annotations

import imageio.v3 as iio
import numpy as np
import math
import torch
import torch.nn.functional as F


def load_video(path: str, frames: int = 64) -> np.ndarray:
    props = iio.improps(path, plugin="FFMPEG")
    count_value = props.n_images
    if not np.isfinite(count_value) or count_value <= 0:
        meta = iio.immeta(path, plugin="FFMPEG")
        count_value = math.ceil(float(meta["duration"]) * float(meta["fps"]))
    count = int(count_value)
    if count <= 0:
        raise ValueError(f"video contains no frames: {path}")
    ids = np.linspace(0, count - 1, frames).round().astype(np.int64)
    wanted = set(ids.tolist())
    sampled = {}
    for index, frame in enumerate(iio.imiter(path, plugin="FFMPEG")):
        if index in wanted:
            sampled[index] = np.asarray(frame)[..., :3]
        if index >= ids[-1]:
            break
    return np.stack([sampled[index] for index in ids], axis=0).astype(np.uint8)


def preprocess_vjepa21(frames: np.ndarray, size: int = 384) -> torch.Tensor:
    """Convert T,H,W,C uint8 frames to normalized B,C,T,H,W V-JEPA input."""
    video = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
    video = F.interpolate(video, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    video = (video - mean) / std
    return video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
