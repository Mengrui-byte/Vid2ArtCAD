"""Deterministic coordinate and forward-kinematics helpers.

Matrices are 4x4 row-major arrays. `T_a_b @ v_b` maps a vector in b to a.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def homogenous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(unit)
    if norm == 0:
        raise ValueError("joint axis cannot be zero")
    unit /= norm
    skew = np.array(
        [
            [0.0, -unit[2], unit[1]],
            [unit[2], 0.0, -unit[0]],
            [-unit[1], unit[0], 0.0],
        ]
    )
    rotation = np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)
    return rotation


def joint_transform(joint_type: str, axis: np.ndarray, q: float) -> np.ndarray:
    if joint_type == "fixed":
        return np.eye(4)
    if joint_type == "revolute":
        return homogenous(axis_angle(axis, q), np.zeros(3))
    if joint_type == "prismatic":
        unit = np.asarray(axis, dtype=np.float64)
        unit = unit / np.linalg.norm(unit)
        return homogenous(np.eye(3), unit * q)
    raise ValueError(f"unsupported joint type: {joint_type}")


def joint_transform_dofs(
    joint_type: str,
    q: np.ndarray,
    axes: np.ndarray | None = None,
) -> np.ndarray:
    """Return a joint displacement for a canonical multi-DOF coordinate.

    `axes` is expressed in the joint frame.  For ball joints, ``q`` is a
    rotation vector; for planar joints the coordinates are translation along
    axes 0/1 followed by rotation about axis 2.  Pin-slot uses rotation about
    axis 0 and translation along axis 1.  Source adapters must resolve these
    axes before calling this function.
    """
    values = np.asarray(q, dtype=np.float64).reshape(-1)
    basis = None if axes is None else np.asarray(axes, dtype=np.float64)

    def require_axes(count: int) -> np.ndarray:
        if basis is None or basis.shape != (count, 3):
            raise ValueError(f"{joint_type} requires {count} joint-frame axes")
        norms = np.linalg.norm(basis, axis=1)
        if np.any(norms == 0):
            raise ValueError("joint axes cannot be zero")
        return basis / norms[:, None]

    if joint_type in {"fixed", "revolute", "prismatic"}:
        if values.size != (0 if joint_type == "fixed" else 1):
            raise ValueError(f"{joint_type} expects {0 if joint_type == 'fixed' else 1} q values")
        return np.eye(4) if joint_type == "fixed" else joint_transform(joint_type, require_axes(1)[0], values[0])
    if joint_type == "cylindrical":
        if values.size != 2:
            raise ValueError("cylindrical expects [rotation, translation]")
        axis = require_axes(1)[0]
        return joint_transform("revolute", axis, values[0]) @ joint_transform("prismatic", axis, values[1])
    if joint_type == "pin_slot":
        if values.size != 2:
            raise ValueError("pin_slot expects [rotation, translation]")
        basis = require_axes(2)
        return joint_transform("revolute", basis[0], values[0]) @ joint_transform("prismatic", basis[1], values[1])
    if joint_type == "ball":
        if values.size != 3:
            raise ValueError("ball expects a 3D rotation vector")
        angle = float(np.linalg.norm(values))
        rotation = np.eye(3) if angle == 0 else axis_angle(values / angle, angle)
        return homogenous(rotation, np.zeros(3))
    if joint_type == "planar":
        if values.size != 3:
            raise ValueError("planar expects [translation_0, translation_1, rotation]")
        basis = require_axes(3)
        out = joint_transform("prismatic", basis[0], values[0])
        out = out @ joint_transform("prismatic", basis[1], values[1])
        return out @ joint_transform("revolute", basis[2], values[2])
    raise ValueError(f"unsupported joint type: {joint_type}")


@dataclass(frozen=True)
class KinematicGraph:
    """Canonical tree with root as world; index order is the exported link order."""

    link_ids: tuple[str, ...]
    root: str
    parent_of: dict[str, str]
    joint_of_child: dict[str, str]


def fk_q_zero(
    T_parent_joint: np.ndarray,
    T_joint_child_ref: np.ndarray,
) -> np.ndarray:
    return T_parent_joint @ T_joint_child_ref


def fk(
    graph: KinematicGraph,
    links: dict[str, dict],
    joints: dict[str, dict],
    q: dict[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    """Compute world transforms. q is displacement/angle relative to q=0."""

    root = graph.root
    world: dict[str, np.ndarray] = {root: np.eye(4)}
    valid: dict[str, bool] = {root: True}

    def visit(link_id: str) -> None:
        for child in (name for name, parent in graph.parent_of.items() if parent == link_id):
            joint_id = graph.joint_of_child[child]
            spec = joints[joint_id]
            axis = np.asarray(spec["axis_in_joint"], dtype=np.float64)
            A = np.asarray(spec["T_parent_joint"], dtype=np.float64)
            B = np.asarray(spec["T_joint_child_ref"], dtype=np.float64)
            qj = q.get(joint_id, 0.0)
            q_valid = True
            if spec.get("type") != "fixed" and not spec.get("q_valid", True):
                qj = 0.0
                q_valid = False
            M = joint_transform(spec["type"], axis, qj)
            world[child] = world[link_id] @ A @ M @ B
            valid[child] = valid[link_id] and q_valid
            visit(child)

    visit(root)
    return world, valid
