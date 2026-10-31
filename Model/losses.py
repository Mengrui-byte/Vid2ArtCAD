"""Losses for kinematics supervision and structured BRep diffusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rotations import geodesic_angle, rotation_6d_to_matrix


def kinematics_loss(pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    present = target["present"].float()
    active = target["active"].float()
    denom = present.sum().clamp_min(1.0)
    active_denom = active.sum().clamp_min(1.0)
    losses = {
        "present": F.binary_cross_entropy_with_logits(pred["present"], present),
        "parent": (F.cross_entropy(pred["parent"].transpose(1, 2), target["parent"], reduction="none") * present).sum() / denom,
        "joint_type": (F.cross_entropy(pred["joint_type"].transpose(1, 2), target["joint_type"], reduction="none") * present).sum() / denom,
        "axis": (F.smooth_l1_loss(pred["axis"], target["axis"], reduction="none") * active.unsqueeze(-1)).sum() / active_denom,
        "origin": (F.smooth_l1_loss(pred["origin"], target["origin"], reduction="none") * present.unsqueeze(-1)).sum() / denom,
        "rotation": (geodesic_angle(rotation_6d_to_matrix(pred["rotation6d"]),
                                    target["rotation"]) * present).sum() / denom,
        "limits": (F.smooth_l1_loss(pred["limits"], target["limits"], reduction="none") * active.unsqueeze(-1)).sum() / active_denom,
    }
    losses["total"] = sum(losses.values())
    return losses


def brep_diffusion_loss(pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
    valid = mask.bool()
    if not valid.any():
        raise ValueError("BRep target mask has no valid nodes")
    node_type = F.cross_entropy(pred["type_logits"].transpose(1, 2), target["node_type"].long(), reduction="none")
    parent = F.cross_entropy(pred["parent_logits"].transpose(1, 2), target["parent"].long(), reduction="none")
    losses = {
        "noise": pred["noise"],
        "node_type": node_type[valid].mean(),
        "parent": parent[valid].mean(),
    }
    losses["total"] = losses["noise"] + losses["node_type"] + losses["parent"]
    return losses


def obj_vertex_loss(pred: torch.Tensor, target: torch.Tensor, vertex_mask: torch.Tensor) -> torch.Tensor:
    mask = vertex_mask.unsqueeze(-1).to(pred.dtype)
    return ((pred - target).square() * mask).sum() / (mask.sum().clamp_min(1.0) * pred.shape[-1])
