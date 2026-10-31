"""Layered validation. Failures are fielded, never silently repaired."""

from __future__ import annotations

import math

import numpy as np

from pipeline.core.schema import QualityReport, SampleStatus


def rigid_transform_error(T: np.ndarray) -> float:
    R = T[:3, :3]
    orth = np.linalg.norm(R.T @ R - np.eye(3), ord="fro")
    det = abs(np.linalg.det(R) - 1.0)
    bottom = np.linalg.norm(T[3] - np.array([0, 0, 0, 1]))
    return max(orth, det, bottom)


def reference_pose_recovery(T_a: np.ndarray, T_b: np.ndarray, diagonal_d: float) -> tuple[float, float]:
    R_err = max(
        0.0,
        math.acos(np.clip((np.trace(T_a[:3, :3].T @ T_b[:3, :3]) - 1) / 2, -1.0, 1.0)),
    )
    t_err = np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])
    return float(t_err / diagonal_d), float(R_err)


def gate(clip_id: str, metrics: dict[str, float], failures: list[str]) -> QualityReport:
    limits = {
        "rigid_transform_error": 1e-5,
        "reference_translation_over_d": 1e-4,
        "reference_rotation_rad": 1e-4,
        "projection_median_px": 1.0,
    }
    for name, threshold in limits.items():
        if name in metrics and metrics[name] > threshold:
            failures.append(f"threshold_exceeded:{name}")
    if not failures:
        status = SampleStatus.ACCEPTED
    elif len(failures) == 1 and failures[0] in {"unknown_joint_limit", "low_visibility", "low_excitation"}:
        status = SampleStatus.PARTIAL
    else:
        status = SampleStatus.REVIEW
    return QualityReport(clip_id=clip_id, status=status, reason_codes=failures, metrics=metrics)
