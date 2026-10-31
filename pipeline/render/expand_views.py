"""Add deterministic extra cameras to an existing synthetic multiview sample."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from pipeline.render.labels import export_labels
from pipeline.render.mujoco_runtime import set_qpos
from pipeline.render.stereo_annotation import digest, dump, project


def _camera_pose(base: np.ndarray, points: np.ndarray, K: np.ndarray, width: int,
                 height: int, offset_degrees: float, border: float = 0.04) -> np.ndarray:
    target = (points.min(axis=0) + points.max(axis=0)) / 2
    azimuth = np.deg2rad(offset_degrees)
    back = Rotation.from_rotvec([0, 0, azimuth]).apply(-base[:3, 2])
    back /= np.linalg.norm(back)
    right = np.cross([0, 0, 1], back)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross([0, 1, 0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    Rgl = np.column_stack([right, up, back])
    local = np.einsum('ki,ij->kj', points - target, Rgl)
    hx = (width - 1) / 2 - width * border
    hy = (height - 1) / 2 - height * border
    distance = max(np.max(local[:, 2] + np.abs(local[:, 0]) * K[0, 0] / hx),
                   np.max(local[:, 2] + np.abs(local[:, 1]) * K[1, 1] / hy))
    eye = target + back * distance
    transform = np.eye(4)
    transform[:3, :3] = Rgl @ np.diag([1, -1, -1])
    transform[:3, 3] = eye
    return transform


def _mesh_points(sample: Path, motion: np.lib.npyio.NpzFile) -> np.ndarray:
    corners = []
    for name in motion['link_order']:
        mesh = trimesh.load(sample / 'meshes' / f'{name}.obj', process=False)
        corners.append(trimesh.bounds.corners(mesh.bounds))
    corners = np.asarray(corners)
    poses = motion['T_world_link']
    return (np.einsum('tpij,pkj->tpki', poses[:, :, :3, :3], corners)
            + poses[:, :, :3, 3][:, :, None, :]).reshape(-1, 3)


def _camera_xml(transform: np.ndarray, name: str, fovy: float) -> ET.Element:
    camera = ET.Element('camera', name=name)
    camera.set('pos', ' '.join(f'{value:.16g}' for value in transform[:3, 3]))
    quat = Rotation.from_matrix(transform[:3, :3] @ np.diag([1, -1, -1])).as_quat()[[3, 0, 1, 2]]
    camera.set('quat', ' '.join(f'{value:.16g}' for value in quat))
    camera.set('fovy', str(fovy))
    return camera


def _write_contact_sheet(sample: Path, cameras: dict, frame_count: int) -> None:
    selected = sorted(set((0, min(165, frame_count - 1), min(285, frame_count - 1))))
    previews = {}
    preview_dir = sample / 'preview'
    preview_dir.mkdir(exist_ok=True)
    for name, spec in sorted(cameras.items()):
        reader = imageio.get_reader(sample / spec['video'])
        try:
            for index in selected:
                image = np.asarray(reader.get_data(index))
                previews[name, index] = image[::3, ::3]
                imageio.imwrite(preview_dir / f'{name}_{index:04d}.png', image)
        finally:
            reader.close()
    rows = [np.concatenate([previews[name, index] for name in sorted(cameras)], axis=1)
            for index in selected]
    imageio.imwrite(preview_dir / 'views_contact_sheet.png', np.concatenate(rows, axis=0))


def _expand_views_in_place(sample: Path, offsets_degrees: list[float] | tuple[float, ...],
                           overwrite: bool = False) -> dict:
    sample_data = json.loads((sample / 'sample.json').read_text())
    if sample_data.get('mode') != 'multiview_monocular':
        raise ValueError('expand-views requires a synthetic multiview sample')
    views_path = sample / str(sample_data.get('views', 'views.json'))
    views = json.loads(views_path.read_text())
    cameras = dict(views.get('cameras') or {})
    existing = sorted(cameras)
    if 'cam01' not in existing:
        raise ValueError('sample must contain cam01')
    camera01 = json.loads((sample / 'camera_cam01.json').read_text())
    width, height = map(int, camera01['resolution_wh'])
    K = np.asarray(camera01['K'], dtype=float)
    motion = np.load(sample / 'motion.npz', allow_pickle=False)
    timestamps = motion['timestamps']; q = motion['q']; joint_order = motion['joint_order']
    points = _mesh_points(sample, motion)
    scene_path = sample / 'scene_multiview.xml'
    xml = ET.parse(scene_path)
    world = xml.getroot().find('worldbody')
    if world is None:
        raise ValueError('scene has no worldbody')
    requested_names = {f'cam{index:02d}' for index in range(3, 3 + len(offsets_degrees))}
    for child in list(world):
        if child.tag == 'camera' and child.get('name') in requested_names:
            if not overwrite:
                raise FileExistsError(f"{child.get('name')} camera already exists in the scene; use --overwrite")
            world.remove(child)
    runtime_path = sample / 'joint_runtime.json'
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else None
    added = []
    framing = dict(json.loads((sample / 'quality.json').read_text()).get('framing') or {})
    for index, offset in enumerate(offsets_degrees, start=3):
            name = f'cam{index:02d}'
            video_path = sample / f'rgb_{name}.mp4'
            camera_path = sample / f'camera_{name}.json'
            arrays_path = sample / f'camera_{name}.npz'
            if not overwrite and any(path.exists() for path in (video_path, camera_path, arrays_path)):
                raise FileExistsError(f'{name} already exists; use --overwrite')
            transform = _camera_pose(np.asarray(camera01['T_world_camera']), points, K,
                                     width, height, float(offset))
            uv, depth = project(points, transform, K)
            low, high = uv.min(axis=0), uv.max(axis=0)
            if depth.min() <= 0 or np.any(low < 20) or np.any(high >= np.array([width, height]) - 20):
                raise ValueError(f'{name} camera framing is outside the image')
            world.append(_camera_xml(transform, name, float(camera01['fovy_degrees'])))
            ET.indent(xml)
            xml.write(scene_path, encoding='utf-8', xml_declaration=True)
            model = mujoco.MjModel.from_xml_path(str(scene_path))
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height, width)
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if camera_id < 0:
                renderer.close()
                raise ValueError(f'camera was not compiled: {name}')
            writer = imageio.get_writer(video_path, fps=float(camera01['fps']), codec='libx264',
                                        pixelformat='yuv420p', macro_block_size=1,
                                        ffmpeg_params=['-crf', '18', '-preset', 'fast'])
            try:
                for frame, timestamp in enumerate(timestamps):
                    set_qpos(model, data, q[frame], joint_order, runtime)
                    data.time = float(timestamp)
                    mujoco.mj_forward(model, data)
                    actual = np.eye(4)
                    actual[:3, :3] = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
                    actual[:3, 3] = data.cam_xpos[camera_id]
                    if not np.allclose(actual, transform, atol=1e-8, rtol=0):
                        raise ValueError(f'{name} camera pose drifted during replay')
                    renderer.update_scene(data, camera=name)
                    writer.append_data(renderer.render())
            finally:
                writer.close()
                renderer.close()
            dump(camera_path, dict(camera01, T_world_camera=transform.tolist(),
                                   framing=dict(minimum_pixel=low.tolist(), maximum_pixel=high.tolist()),
                                   azimuth_offset_degrees=float(offset)))
            np.savez_compressed(arrays_path, K=np.repeat(K[None], len(timestamps), axis=0),
                                T_world_camera=np.repeat(transform[None], len(timestamps), axis=0),
                                timestamps=timestamps)
            cameras[name] = dict(video=video_path.name, calibration=camera_path.name,
                                  calibration_arrays=arrays_path.name)
            framing[name] = dict(minimum_pixel=low.tolist(), maximum_pixel=high.tolist())
            added.append(name)
    ET.indent(xml)
    xml.write(scene_path, encoding='utf-8', xml_declaration=True)
    views.update({'mode': 'multi_aimed_monocular_cameras', 'cameras': cameras,
                  'added_azimuth_offsets_degrees': {
                      name: float(offsets_degrees[index]) for index, name in enumerate(added)
                  }})
    dump(views_path, views)
    label_report = export_labels(sample, cameras=tuple(sorted(cameras)))
    quality_path = sample / 'quality.json'
    quality = json.loads(quality_path.read_text())
    quality['videos'] = {}
    for name, spec in sorted(cameras.items()):
        reader = imageio.get_reader(sample / spec['video'])
        try:
            meta = reader.get_meta_data()
            count = sum(1 for _ in reader)
        finally:
            reader.close()
        quality['videos'][name] = {'frame_count': count, 'fps': float(meta.get('fps', camera01['fps'])),
                                   'resolution_wh': [width, height]}
    quality['framing'] = framing
    quality['view_count'] = len(cameras)
    quality['available_tasks'] = ['rgb', 'motion', 'kinematics', 'depth', 'part_id', 'tracks']
    dump(quality_path, quality)
    _write_contact_sheet(sample, cameras, len(timestamps))
    sample_data['label_report'] = label_report
    dump(sample / 'sample.json', sample_data)
    dump(sample / 'checksums.json', {
        str(path.relative_to(sample)): digest(path)
        for path in sorted(sample.rglob('*')) if path.is_file() and path.name != 'checksums.json'
    })
    return {'sample': str(sample), 'added_views': added, 'view_count': len(cameras),
            'frame_count': len(timestamps)}


def expand_views(sample: Path, offsets_degrees: list[float] | tuple[float, ...] = (90.0, 270.0),
                 overwrite: bool = False) -> dict:
    """Expand a sample in a staged copy and atomically publish only a validated result."""
    sample = sample.resolve()
    if not sample.is_dir():
        raise FileNotFoundError(sample)
    if overwrite:
        raise ValueError('transactional expansion does not support overwrite; use a fresh sample copy')
    stage = Path(tempfile.mkdtemp(prefix=f'.{sample.name}.building-', dir=sample.parent))
    shutil.rmtree(stage)
    backup = sample.parent / f'.{sample.name}.pre-expand'
    if backup.exists():
        raise FileExistsError(f'preserved pre-expansion backup already exists: {backup}')
    try:
        shutil.copytree(sample, stage)
        result = _expand_views_in_place(stage, offsets_degrees, overwrite=False)
        from pipeline.render.validate_sample import validate_sample
        validation = validate_sample(stage, check_video=True)
        if validation.get('status') != 'accepted':
            raise ValueError(f"expanded sample failed validation: {validation.get('reason_codes')}")
        if result['view_count'] < 4:
            raise ValueError('expanded sample must contain at least four views')
        validation['sample_id'] = sample.name
        os.replace(sample, backup)
        try:
            os.replace(stage, sample)
        except BaseException:
            os.replace(backup, sample)
            raise
        result.update({'sample': str(sample), 'validation': validation,
                       'preserved_backup': str(backup)})
        return result
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Add extra deterministic cameras to a synthetic sample')
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--offset-degrees', type=float, nargs='+', default=[90.0, 270.0])
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    print(json.dumps(expand_views(args.sample, args.offset_degrees, args.overwrite), ensure_ascii=False, indent=2))
