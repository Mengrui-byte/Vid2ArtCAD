"""Fusion 360 Gallery adapter boundary.

The source adapter must keep original fields and emit canonical fields only after
coordinate/zero-position resolution. Do not collapse different source angle fields
into a generic `angle`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pipeline.core.schema import CanonicalAsset
from pipeline.kinematics.graph import canonical_joint_type


class FusionParser(Protocol):
    def parse(self, asset_dir: Path) -> tuple[CanonicalAsset, dict[str, Any]]: ...


class FusionAdapter:
    """Thin orchestration shell; concrete parsing is injected per dataset release."""

    def __init__(self, parser: FusionParser) -> None:
        self.parser = parser

    def to_canonical(self, asset_dir: Path) -> tuple[CanonicalAsset, dict[str, Any]]:
        asset, raw = self.parser.parse(asset_dir)
        self._verify(asset, raw)
        return asset, raw

    @staticmethod
    def _verify(asset: CanonicalAsset, raw: dict[str, Any]) -> None:
        ids = {link.id for link in asset.links}
        for joint in asset.joints:
            if joint.parent not in ids or joint.child not in ids:
                raise ValueError(f"joint {joint.id} references unknown link")
            if joint.type != "fixed" and joint.coordinate_status == "unresolved":
                # Geometry and q(t) may still be synthesized around q=0, but
                # source mechanical limits must not be silently inherited.
                if joint.mechanical_limits is not None:
                    raw.setdefault("warnings", []).append(
                        f"{joint.id}: unresolved source zero; source limits kept raw only"
                    )


def _vector(value: dict[str, Any]) -> np.ndarray:
    result = np.array([value.get(axis, 0.0) for axis in ("x", "y", "z")], dtype=np.float64)
    norm = np.linalg.norm(result)
    if norm == 0:
        raise ValueError("joint geometry contains a zero axis")
    return result / norm


def source_joint_dof_spec(raw_joint: dict[str, Any]) -> dict[str, Any]:
    """Extract canonical DOFs and joint-frame axes from Fusion joint JSON.

    Fusion's ``geometry_or_origin_one`` gives a stable joint frame.  Its
    primary axis is the hinge/slider axis; planar joints use the primary axis
    as the plane normal and secondary/tertiary axes as in-plane translations.
    The returned axes remain in the source joint frame and must be transformed
    by the source-to-canonical transform by the caller.
    """
    motion = raw_joint.get("joint_motion") or {}
    geometry = raw_joint.get("geometry_or_origin_one") or raw_joint.get("geometry_or_origin_two") or {}
    origin = geometry.get("origin") or {}
    joint_type = canonical_joint_type(motion.get("joint_type", ""))
    if joint_type == "fixed":
        axis_rows = []
    elif joint_type in {"revolute", "prismatic"}:
        axis_rows = [0]
    elif joint_type == "cylindrical":
        # Rotation and translation share the primary axis.
        axis_rows = [0]
    elif joint_type == "pin_slot":
        axis_rows = [0, 1]
    elif joint_type == "ball":
        axis_rows = [0, 1, 2]
    elif joint_type == "planar":
        axis_rows = [0, 1, 2]
    else:  # guarded by canonical_joint_type; retained for future extensions
        raise ValueError(f"no DOF mapping for {joint_type}")
    limits = {}
    for key in ("rotation_limits", "slide_limits", "pitch_limits", "yaw_limits", "roll_limits",
                "primary_slide_limits"):
        if key in motion:
            limits[key] = motion[key]
    resolved = bool(geometry.get("primary_axis_vector")) or joint_type == "fixed"
    warnings = []
    if not resolved:
        warnings.append("joint geometry is absent; coordinate resolution required")
    if joint_type == "fixed":
        selected_axes = []
    elif not resolved:
        selected_axes = []
    else:
        primary = _vector(geometry.get("primary_axis_vector") or {})
        all_axes = {
            0: primary,
            1: _vector(geometry.get("secondary_axis_vector") or {}) if joint_type != "cylindrical" and len(axis_rows) > 1 else primary,
            2: _vector(geometry.get("tertiary_axis_vector") or {}) if len(axis_rows) > 2 else None,
        }
        selected_axes = [all_axes[index].tolist() for index in axis_rows]
    return {
        "type": joint_type,
        "dof_names": tuple(f"{raw_joint.get('name', 'joint')}:{i}" for i in range(
            2 if joint_type in {"cylindrical", "pin_slot"} else len(axis_rows))),
        "axes_in_joint": selected_axes,
        "origin_source_units": [origin.get(axis, 0.0) for axis in ("x", "y", "z")],
        "coordinate_status": "resolved" if resolved else "unresolved",
        "warnings": warnings,
        "limits": limits,
        "source_joint": raw_joint,
    }
