"""Overfit one video to a fixed-topology OBJ assembly and kinematic tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .losses import kinematics_loss, obj_vertex_loss
from .model import VideoJEPAAssemblyModel
from .representation import load_urdf_target
from .rotations import matrix_to_rpy, rotation_6d_to_matrix
from .video import load_video


def _kin_target(target: dict, device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in target.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiu":
            dtype = torch.long if value.dtype.kind == "i" else torch.float32
            result[key] = torch.from_numpy(value).to(device=device, dtype=dtype).unsqueeze(0)
    return result


def _write_obj(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    names: list[str],
    scale: float,
    vertex_counts: np.ndarray,
    face_counts: np.ndarray,
) -> None:
    vertex_offset = 0
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Video-conditioned fixed-topology OBJ prediction\n")
        for part_idx, name in enumerate(names):
            count = int(vertex_counts[part_idx])
            valid_faces = faces[part_idx, :int(face_counts[part_idx])]
            if count == 0 or not len(valid_faces):
                continue
            stream.write(f"o {name}\n")
            for vertex in vertices[part_idx, :count] * scale:
                stream.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
            for face in valid_faces:
                stream.write(f"f {face[0] + vertex_offset + 1} {face[1] + vertex_offset + 1} {face[2] + vertex_offset + 1}\n")
            vertex_offset += count


def _write_part_objs(
    directory: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    names: list[str],
    scale: float,
    vertex_counts: np.ndarray,
    face_counts: np.ndarray,
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    filenames = []
    for part_idx, name in enumerate(names):
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", name) + ".obj"
        filenames.append(filename)
        count = int(vertex_counts[part_idx])
        valid_faces = faces[part_idx, :int(face_counts[part_idx])]
        with (directory / filename).open("w", encoding="utf-8") as stream:
            stream.write(f"o {name}\n")
            for vertex in vertices[part_idx, :count] * scale:
                stream.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
            for face in valid_faces:
                stream.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    return filenames


def _write_urdf(path: Path, kin_pred: dict[str, np.ndarray], names: list[str], mesh_files: list[str], scale: float) -> None:
    present_count = len(names)
    joint_names = ("fixed", "revolute", "prismatic")
    robot = ET.Element("robot", {"name": "video_generated_assembly"})
    ET.SubElement(robot, "link", {"name": "world"})
    for name, mesh_file in zip(names, mesh_files):
        link = ET.SubElement(robot, "link", {"name": name})
        visual = ET.SubElement(link, "visual")
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "mesh", {"filename": f"parts/{mesh_file}"})
    for child in range(present_count):
        parent_index = int(np.argmax(kin_pred["parent"][child]))
        parent_name = names[parent_index] if parent_index < present_count and parent_index != child else "world"
        joint_type = joint_names[int(np.argmax(kin_pred["joint_type"][child]))]
        joint = ET.SubElement(robot, "joint", {"name": f"joint_{child:02d}", "type": joint_type})
        ET.SubElement(joint, "parent", {"link": parent_name})
        ET.SubElement(joint, "child", {"link": names[child]})
        xyz = " ".join(f"{value * scale:.8g}" for value in kin_pred["origin"][child])
        matrix = rotation_6d_to_matrix(torch.from_numpy(kin_pred["rotation6d"][child])).numpy()
        rpy = " ".join(f"{value:.8g}" for value in matrix_to_rpy(matrix))
        ET.SubElement(joint, "origin", {"xyz": xyz, "rpy": rpy})
        if joint_type != "fixed":
            axis = " ".join(f"{value:.8g}" for value in kin_pred["axis"][child])
            ET.SubElement(joint, "axis", {"xyz": axis})
            lower, upper = kin_pred["limits"][child]
            if joint_type == "prismatic":
                lower *= scale
                upper *= scale
            ET.SubElement(joint, "limit", {
                "lower": f"{lower:.8g}", "upper": f"{upper:.8g}",
                "effort": "1", "velocity": "1",
            })
    ET.indent(robot)
    ET.ElementTree(robot).write(path, encoding="unicode", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-a", required=True)
    parser.add_argument("--video-b", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--mesh-target", required=True, help="NPZ made by Model.prepare_obj_target")
    parser.add_argument("--out", default="output/model_overfit")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--vjepa-repo", default="/root/vjepa2")
    parser.add_argument("--vjepa-checkpoint", default="/root/.cache/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt")
    parser.add_argument("--max-parts", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames_a = load_video(args.video_a, args.frames)
    frames_b = load_video(args.video_b, args.frames)
    kin_np = load_urdf_target(args.urdf, args.max_parts)
    kin_target = _kin_target(kin_np, device)
    with np.load(args.mesh_target) as archive:
        target_vertices = np.asarray(archive["vertices"], dtype=np.float32)
        vertex_mask = np.asarray(archive["vertex_mask"], dtype=bool)
        faces = np.asarray(archive["faces"], dtype=np.int32)
        face_counts = np.asarray(archive["face_counts"], dtype=np.int32)
        vertex_counts = np.asarray(archive["vertex_counts"], dtype=np.int32)
        link_names = archive["link_names"].astype(str).tolist()
        scale = float(archive["scale"])
    if target_vertices.shape[:2] != vertex_mask.shape or target_vertices.shape[0] != args.max_parts:
        raise ValueError("mesh target part/vertex dimensions do not match --max-parts")
    if len(link_names) != int(kin_np["present"].sum()):
        raise ValueError("mesh-target link order/count must match the URDF link order")
    target_v = torch.from_numpy(target_vertices).unsqueeze(0).to(device)
    target_mask = torch.from_numpy(vertex_mask).unsqueeze(0).to(device)

    model = VideoJEPAAssemblyModel(
        max_parts=args.max_parts,
        max_vertices=target_vertices.shape[1],
        vjepa_repo=args.vjepa_repo,
        vjepa_checkpoint=args.vjepa_checkpoint,
        vjepa_num_frames=args.frames,
    ).to(device)
    preprocessor = torch.hub.load(
        args.vjepa_repo, "vjepa2_preprocessor", source="local", pretrained=False, crop_size=384
    )
    video_a = preprocessor(frames_a)[0].unsqueeze(0).to(device)
    video_b = preprocessor(frames_b)[0].unsqueeze(0).to(device)
    model.train()
    with torch.no_grad():
        tokens_a, _ = model.encoder_a(video_a)
        tokens_b, _ = model.encoder_b(video_b)
    tokens_a = tokens_a.detach()
    tokens_b = tokens_b.detach()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    for step in range(1, args.steps + 1):
        outputs, jepa = model.forward_from_features(tokens_a, tokens_b)
        kin = kinematics_loss(outputs["kinematics"], kin_target)
        mesh_loss = obj_vertex_loss(outputs["mesh_vertices"], target_v, target_mask)
        total = kin["total"] + mesh_loss + 0.1 * jepa
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 1 or step % max(1, args.steps // 10) == 0:
            print(f"step={step:04d} total={float(total):.5f} mesh={float(mesh_loss):.5f} kin={float(kin['total']):.5f} vjepa={float(jepa):.5f}", flush=True)

    model.eval()
    with torch.no_grad():
        outputs, _ = model.forward_from_features(tokens_a, tokens_b)
    predicted_vertices = outputs["mesh_vertices"][0].cpu().numpy()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_obj(out / "generated_assembly.obj", predicted_vertices, faces, link_names, scale, vertex_counts, face_counts)
    mesh_files = _write_part_objs(out / "parts", predicted_vertices, faces, link_names, scale, vertex_counts, face_counts)

    kin_pred = {key: value[0].cpu().numpy() for key, value in outputs["kinematics"].items() if key != "slots"}
    kin_json = {
        "representation": "predicted_kinematic_tree",
        "links": link_names,
        "slots": [
            {
                "slot": i,
                "link_name": link_names[i] if i < len(link_names) else None,
                "present_probability": float(1.0 / (1.0 + np.exp(-kin_pred["present"][i]))),
                "parent_slot": int(np.argmax(kin_pred["parent"][i])),
                "joint_type_id": int(np.argmax(kin_pred["joint_type"][i])),
                "axis": kin_pred["axis"][i].tolist(),
                "origin": kin_pred["origin"][i].tolist(),
                "limits": kin_pred["limits"][i].tolist(),
            }
            for i in range(args.max_parts)
        ],
    }
    (out / "kinematic_tree.json").write_text(json.dumps(kin_json, indent=2) + "\n")
    _write_urdf(out / "generated_assembly.urdf", kin_pred, link_names, mesh_files, scale)
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    torch.save({
        "trainable_model": {name: value.detach().cpu() for name, value in model.state_dict().items()
                            if name in trainable_names},
        "vjepa_checkpoint": args.vjepa_checkpoint,
        "args": vars(args),
    }, out / "checkpoint.pt")
    print(f"wrote {out / 'generated_assembly.obj'} and {out / 'generated_assembly.urdf'}", flush=True)


if __name__ == "__main__":
    main()
