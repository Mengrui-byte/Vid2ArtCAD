"""Generate synchronized edit pairs from a published kinematic sample."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from pipeline.render.labels import export_labels
from pipeline.render.stereo_annotation import digest, dump
from pipeline.render.mujoco_runtime import set_qpos


def joint_indices(joint_order: np.ndarray, joint: str) -> list[int]:
    names = [str(value) for value in joint_order]
    indices = [index for index, name in enumerate(names)
               if name == joint or name.startswith(f'{joint}:')]
    if not indices:
        raise ValueError(f'joint is absent from motion.npz: {joint}')
    return indices


def edited_coordinates(q: np.ndarray, indices: list[int], scale: float) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('edit scale must be finite and greater than zero')
    result = np.array(q, dtype=np.float64, copy=True)
    result[:, indices] *= scale
    if np.allclose(result[:, indices], q[:, indices]):
        raise ValueError('edit scale does not change the selected trajectory')
    return result


def _render(stage: Path, q: np.ndarray, timestamps: np.ndarray,
            joint_order: np.ndarray, link_order: np.ndarray,
            runtime: dict | None) -> tuple[dict, np.ndarray]:
    camera = json.loads((stage / 'camera_cam01.json').read_text())
    width, height = map(int, camera['resolution_wh'])
    fps = float(camera['fps'])
    model = mujoco.MjModel.from_xml_path(str((stage / 'scene_multiview.xml').resolve()))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
                for name in link_order]
    if min(body_ids) < 0:
        raise ValueError('edit sample contains a link absent from scene_multiview.xml')
    poses = np.zeros((len(timestamps), len(body_ids), 4, 4), dtype=np.float64)
    poses[:, :, 3, 3] = 1.0
    writers = {
        camera_name: imageio.get_writer(
            stage / f'rgb_{camera_name}.mp4', fps=fps, codec='libx264',
            pixelformat='yuv420p', macro_block_size=1,
            ffmpeg_params=['-crf', '18', '-preset', 'fast'])
        for camera_name in ('cam01', 'cam02')
    }
    try:
        for frame_index, timestamp in enumerate(timestamps):
            set_qpos(model, data, q[frame_index], joint_order, runtime)
            data.time = float(timestamp)
            mujoco.mj_forward(model, data)
            poses[frame_index, :, :3, :3] = data.xmat[body_ids].reshape(-1, 3, 3)
            poses[frame_index, :, :3, 3] = data.xpos[body_ids]
            for camera_name, writer in writers.items():
                renderer.update_scene(data, camera=camera_name)
                writer.append_data(renderer.render())
    finally:
        for writer in writers.values():
            writer.close()
        renderer.close()
    return {
        camera_name: {'frame_count': len(timestamps), 'fps': fps,
                      'resolution_wh': [width, height],
                      'video': f'rgb_{camera_name}.mp4'}
        for camera_name in writers
    }, poses


def create_edit_pair(sample: Path, output: Path, joint: str, scale: float) -> dict:
    if output.exists():
        raise FileExistsError(output)
    loops = json.loads((sample / 'loop_constraints.json').read_text()).get('constraints', [])
    if loops:
        raise ValueError('edit_requires_loop_solver: sample contains closure constraints')
    motion = np.load(sample / 'motion.npz', allow_pickle=False)
    indices = joint_indices(motion['joint_order'], joint)
    q_edit = edited_coordinates(motion['q'], indices, scale)
    stage = output.with_name(output.name + '.building')
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copytree(
        sample, stage, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('labels', 'preview', 'checksums.json',
                                       'sample.json', 'quality.json', 'motion.npz',
                                       'rgb_cam01.mp4', 'rgb_cam02.mp4'))
    runtime = json.loads((sample / 'joint_runtime.json').read_text()) \
        if (sample / 'joint_runtime.json').exists() else None
    npz_values = {key: motion[key] for key in motion.files}
    npz_values['q'] = q_edit
    if 'q_all' in npz_values and 'all_joint_order' in npz_values:
        all_order = [str(value) for value in npz_values['all_joint_order']]
        for index, name in enumerate(motion['joint_order']):
            if str(name) in all_order:
                npz_values['q_all'][:, all_order.index(str(name))] = q_edit[:, index]
    np.savez_compressed(stage / 'motion.npz', **npz_values)
    videos, edited_poses = _render(
        stage, q_edit, motion['timestamps'], motion['joint_order'], motion['link_order'], runtime)
    npz_values['T_world_link'] = edited_poses
    if 'T_world_occurrence' in npz_values:
        npz_values['T_world_occurrence'] = edited_poses
    np.savez_compressed(stage / 'motion.npz', **npz_values)
    label_report = export_labels(stage)
    edit = {
        'schema_version': '1.0',
        'mode': 'kinematic_edit_pair',
        'source_sample': str(sample),
        'edited_joint': joint,
        'edited_q_indices': indices,
        'scale': scale,
        'source_of_geometry': 'dataset_reference',
        'source_of_motion': 'pipeline_policy',
        'synchronized_outputs': ['rgb_cam01.mp4', 'rgb_cam02.mp4',
                                 'labels/part_id_cam01', 'labels/part_id_cam02',
                                 'labels/depth_cam01', 'labels/depth_cam02'],
    }
    dump(stage / 'edit.json', edit)
    dump(stage / 'quality.json', {
        'status': 'edit_replay_passed',
        'available_tasks': ['rgb', 'motion', 'kinematics', 'depth', 'part_id', 'tracks', 'edit_pair'],
        'videos': videos,
        'labels': label_report,
    })
    sample_json = json.loads((sample / 'sample.json').read_text())
    sample_json.update({'mode': 'kinematic_edit_pair', 'edit': 'edit.json',
                        'source_sample': str(sample), 'quality': 'quality.json',
                        'labels': 'labels/quality.json'})
    dump(stage / 'sample.json', sample_json)
    dump(stage / 'checksums.json', {
        str(path.relative_to(stage)): digest(path)
        for path in sorted(stage.rglob('*'))
        if path.is_file() and path.name != 'checksums.json'
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(output)
    manifest_path = output.parent / 'edit_manifest.jsonl'
    with manifest_path.open('a') as stream:
        stream.write(json.dumps({**edit, 'output': str(output)}, ensure_ascii=False) + '\n')
    return edit


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--joint', required=True)
    parser.add_argument('--scale', type=float, default=1.5)
    args = parser.parse_args()
    print(json.dumps(create_edit_pair(args.sample, args.output, args.joint, args.scale),
                     ensure_ascii=False, indent=2))
