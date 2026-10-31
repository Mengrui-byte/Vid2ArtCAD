"""Motion synthesis. First version samples independent tree joints only."""

from __future__ import annotations

import numpy as np


def quintic(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return 6 * s**5 - 15 * s**4 + 10 * s**3


def quintic_segment(q0: float, q1: float, duration_s: float, fps: int) -> tuple[np.ndarray, np.ndarray]:
    count = max(2, int(round(duration_s * fps)) + 1)
    timestamps = np.linspace(0.0, duration_s, count, endpoint=True, dtype=np.float64)
    return q0 + (q1 - q0) * quintic((timestamps - timestamps[0]) / duration_s), timestamps


def independent_tree_q(
    ranges: dict[str, tuple[float, float]],
    segments: list[tuple[str, float, float, float]],
    fps: int = 30,
) -> dict[str, np.ndarray]:
    """segments: (joint_id, start_q, end_q, duration). Fixed joints are excluded upstream."""

    per_joint: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {name: [] for name in ranges}
    for joint_id, q0, q1, duration in segments:
        if joint_id not in ranges:
            raise ValueError(f"joint {joint_id} is not an active independent joint")
        lo, hi = ranges[joint_id]
        if not (lo <= min(q0, q1) and max(q0, q1) <= hi):
            raise ValueError(f"segment for {joint_id} exceeds sampling range [{lo}, {hi}]")
        values, stamps = quintic_segment(q0, q1, duration, fps)
        per_joint[joint_id].append((stamps, values))
    return _align(per_joint, fps)


def _align(per_joint: dict[str, list[tuple[np.ndarray, np.ndarray]]], fps: int) -> dict[str, np.ndarray]:
    # Pilot assumes synchronized plans. Multi-duration composition is added in S4 v2.
    samples = max(int(round(duration * fps)) + 1 for _, _, _, duration in
                  [seg for seqs in per_joint.values() for seg in [(0, 0, 0, 4.0)]] or [(0, 0, 0, 4.0)])
    # Simpler deterministic fallback for the pilot single-segment contract.
    samples = int(round(4.0 * fps)) + 1
    out: dict[str, np.ndarray] = {}
    for joint_id, pieces in per_joint.items():
        if not pieces:
            out[joint_id] = np.zeros(samples)
            continue
        stamps, values = pieces[0]
        if len(stamps) != samples:
            raise ValueError("pilot v1 expects equal-duration synchronized segments")
        out[joint_id] = values
    return out
