"""Renderer contract. Backend-specific camera conventions stay in the adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class RenderSpec:
    resolution_wh: tuple[int, int] = (640, 480)
    fps: int = 30
    camera_axes: str = "x_right_y_down_z_forward"
    depth_type: str = "z_depth_m"


class Renderer(Protocol):
    def render_clip(
        self,
        asset_dir: Path,
        clip_dir: Path,
        q: dict[str, np.ndarray],
        T_world_link: dict[str, np.ndarray],
        camera_K: np.ndarray,
        camera_world: np.ndarray,
        spec: RenderSpec,
    ) -> None:
        """Write rgb.mp4, timestamps.npy, camera.npz, part_id/*.png, depth and valid masks."""


class LabelExporter:
    @staticmethod
    def surface_projection(
        uv: np.ndarray,
        z: np.ndarray,
        depth: np.ndarray,
        part_id_image: np.ndarray,
        expected_part_id: int,
        tolerance_px: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = part_id_image.shape
        projection_valid = np.isfinite(uv).all(axis=-1) & (z > 0)
        x = uv[..., 0]
        y = uv[..., 1]
        in_frame = projection_valid & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        xi = np.clip(x, 0, w - 1)
        yi = np.clip(y, 0, h - 1)
        depth_ok = np.abs(z - depth[yi.astype(int), xi.astype(int)]) <= 0.01
        part_ok = part_id_image[yi.astype(int), xi.astype(int)] == expected_part_id
        visible = in_frame & depth_ok & part_ok
        return projection_valid, in_frame, visible
