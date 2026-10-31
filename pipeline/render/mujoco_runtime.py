"""Map canonical motion coordinates to MuJoCo joint coordinates."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def load_runtime(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"joints": [{"q_index": i, "sim_name": name, "kind": "scalar"}
                         for i, name in enumerate([])]}


def set_qpos(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray,
             joint_order: np.ndarray, runtime: dict | None = None) -> None:
    """Set MuJoCo qpos from canonical q using an explicit runtime map."""
    q = np.asarray(q, dtype=float).reshape(-1)
    if not np.isfinite(q).all():
        raise ValueError("canonical q must contain only finite values")
    names = [str(name) for name in joint_order]
    if len(names) != len(q) or len(set(names)) != len(names):
        raise ValueError("joint_order must contain one unique name per q coordinate")
    if runtime is None or not runtime.get("joints"):
        runtime = {"joints": [{"q_indices": [i], "sim_name": name, "kind": "scalar"}
                               for i, name in enumerate(names)]}
    if runtime.get("joint_order") is not None and list(runtime["joint_order"]) != names:
        raise ValueError("runtime joint_order does not match canonical joint_order")

    assignments = []
    mapped_indices = []
    sim_names = set()
    for item in runtime["joints"]:
        sim_name = str(item.get("sim_name", ""))
        if not sim_name or sim_name in sim_names:
            raise ValueError(f"runtime joint name is missing or duplicated: {sim_name!r}")
        sim_names.add(sim_name)
        sim_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, sim_name)
        if sim_id < 0:
            raise ValueError(f"runtime joint is missing from MuJoCo scene: {sim_name}")
        adr = int(model.jnt_qposadr[sim_id])
        try:
            indices = [int(index) for index in item["q_indices"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"runtime q_indices are invalid for {sim_name}") from exc
        if not indices or any(index < 0 or index >= len(q) for index in indices):
            raise ValueError(f"runtime q_indices are out of range for {sim_name}")
        mapped_indices.extend(indices)
        kind = item.get("kind", "scalar")
        joint_type = int(model.jnt_type[sim_id])
        if kind == "ball":
            if joint_type != int(mujoco.mjtJoint.mjJNT_BALL) or len(indices) != 3:
                raise ValueError(f"ball runtime joint must map three coordinates to a ball joint: {sim_name}")
            rotvec = q[indices]
            if item.get("inverse_motion"):
                rotvec = -rotvec
            quat = Rotation.from_rotvec(rotvec).as_quat()
            assignments.append((adr, quat[[3, 0, 1, 2]]))
        else:
            if kind != "scalar" or len(indices) != 1:
                raise ValueError(f"scalar runtime joint must map one coordinate: {sim_name}")
            if joint_type not in {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}:
                raise ValueError(f"scalar runtime joint must target a hinge or slide joint: {sim_name}")
            assignments.append((adr, np.array([q[indices[0]]])))

    if sorted(mapped_indices) != list(range(len(q))):
        raise ValueError("runtime mapping must cover each canonical q coordinate exactly once")
    for adr, value in assignments:
        data.qpos[adr:adr + len(value)] = value
