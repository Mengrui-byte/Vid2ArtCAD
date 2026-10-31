"""Differentiable SO(3) utilities for URDF joint supervision."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    first = F.normalize(x[..., :3], dim=-1, eps=1e-8)
    second = x[..., 3:] - (first * x[..., 3:]).sum(dim=-1, keepdim=True) * first
    second = F.normalize(second, dim=-1, eps=1e-8)
    return torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-1)


def geodesic_angle(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = predicted.transpose(-1, -2) @ target
    skew = torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
                        relative[..., 0, 2] - relative[..., 2, 0],
                        relative[..., 1, 0] - relative[..., 0, 1]), dim=-1)
    sin = torch.linalg.vector_norm(skew, dim=-1) / 2
    cos = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.atan2(sin, cos.clamp(-1, 1))


def matrix_to_rpy(matrix: np.ndarray) -> tuple[float, float, float]:
    pitch = math.atan2(-float(matrix[2, 0]), math.hypot(float(matrix[0, 0]), float(matrix[1, 0])))
    if abs(abs(pitch) - math.pi / 2) < 1e-6:
        return (0.0, pitch, math.atan2(-float(matrix[0, 1]), float(matrix[1, 1])))
    return (math.atan2(float(matrix[2, 1]), float(matrix[2, 2])), pitch,
            math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
