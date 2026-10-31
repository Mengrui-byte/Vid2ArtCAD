"""Compare generated URDF directions and FK poses against a target URDF."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .representation import _rpy_matrix


def _vector(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    return np.fromstring(text, sep=" ") if text else np.asarray(default, dtype=float)


def read_joints(path: Path) -> dict:
    joints = {}
    for joint in ET.parse(path).getroot().findall("joint"):
        child = joint.find("child")
        parent = joint.find("parent")
        if child is None or parent is None:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        xyz = _vector(origin.get("xyz") if origin is not None else None, (0, 0, 0))
        rpy = _vector(origin.get("rpy") if origin is not None else None, (0, 0, 0))
        joints[child.get("link")] = {
            "parent": parent.get("link"), "type": joint.get("type"),
            "xyz": xyz, "rotation": _rpy_matrix(rpy).astype(float),
            "axis": _vector(axis.get("xyz") if axis is not None else None, (0, 0, 1)),
        }
    return joints


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    relative = a.T @ b
    cosine = float(np.clip((np.trace(relative) - 1) / 2, -1, 1))
    return math.degrees(math.acos(cosine))


def fk(joints: dict) -> dict:
    poses = {"world": (np.eye(3), np.zeros(3))}
    visiting = set()

    def pose(link):
        if link in poses:
            return poses[link]
        if link in visiting:
            raise ValueError(f"cycle in predicted URDF at {link}")
        if link not in joints:
            raise ValueError(f"missing parent link {link}")
        visiting.add(link)
        joint = joints[link]
        parent_r, parent_t = pose(joint["parent"])
        poses[link] = (parent_r @ joint["rotation"], parent_t + parent_r @ joint["xyz"])
        visiting.remove(link)
        return poses[link]

    for link in joints:
        pose(link)
    return poses


def evaluate(target: Path, generated: Path) -> dict:
    reference = read_joints(target)
    predicted = read_joints(generated)
    common = sorted(reference.keys() & predicted.keys())
    report = {"target": str(target), "generated": str(generated),
              "target_joints": len(reference), "predicted_joints": len(predicted),
              "matching_links": len(common), "links": []}
    try:
        true_fk, pred_fk = fk(reference), fk(predicted)
    except ValueError as error:
        report["fk_error"] = str(error)
        return report
    for link in common:
        true, pred = reference[link], predicted[link]
        true_r, true_t = true_fk[link]
        pred_r, pred_t = pred_fk[link]
        true_axis = true_r @ true["axis"]
        pred_axis = pred_r @ pred["axis"]
        axis_cos = np.clip(np.dot(true_axis, pred_axis) /
                           max(np.linalg.norm(true_axis) * np.linalg.norm(pred_axis), 1e-12), -1, 1)
        report["links"].append({
            "link": link, "parent_correct": true["parent"] == pred["parent"],
            "joint_type_correct": true["type"] == pred["type"],
            "joint_rotation_degrees": angle_degrees(true["rotation"], pred["rotation"]),
            "fk_rotation_degrees": angle_degrees(true_r, pred_r),
            "fk_translation": float(np.linalg.norm(true_t - pred_t)),
            "world_axis_degrees": float(math.degrees(math.acos(axis_cos))),
        })
    report["mean_joint_rotation_degrees"] = float(np.mean([x["joint_rotation_degrees"] for x in report["links"]]))
    report["mean_fk_rotation_degrees"] = float(np.mean([x["fk_rotation_degrees"] for x in report["links"]]))
    report["mean_fk_translation"] = float(np.mean([x["fk_translation"] for x in report["links"]]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.target, args.generated)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "links"}, indent=2))


if __name__ == "__main__":
    main()
