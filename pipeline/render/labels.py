"""Derive deterministic per-frame labels from an approved MuJoCo sample."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from pipeline.render.mujoco_runtime import set_qpos


def _segmentation_ids(image: np.ndarray, geom_ids: set[int]) -> np.ndarray:
    """Decode MuJoCo's segmentation RGB triplets into geom IDs.

    MuJoCo uses channel 0 for object type and channel 1 for object id.  Pixels
    belonging to non-geometry objects are assigned background zero.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 2:
        raise ValueError(f'unexpected segmentation image shape: {image.shape}')
    first = image[..., 0].astype(np.int64)
    second = image[..., 1].astype(np.int64)
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    # MuJoCo versions have emitted both (type,id) and (id,type) channel
    # ordering.  Do not use ``not any(first == geom_type)``: a real geom can
    # legitimately have the same numeric id as the GEOM enum.
    first_is_type = np.count_nonzero(first == geom_type)
    second_is_type = np.count_nonzero(second == geom_type)
    if second_is_type > first_is_type:
        types, ids = second, first
    elif first_is_type > second_is_type:
        types, ids = first, second
    else:
        # In a tie, the type channel has very low cardinality while the id
        # channel varies across visible geoms.
        first_cardinality = np.unique(first).size
        second_cardinality = np.unique(second).size
        if second_cardinality <= first_cardinality:
            types, ids = second, first
        else:
            types, ids = first, second
    result = np.zeros(types.shape, dtype=np.uint16)
    valid = (types == geom_type) & np.isin(ids, np.fromiter(geom_ids, dtype=np.int64))
    result[valid] = ids[valid].astype(np.uint16) + 1
    return result


def _write_depth(path: Path, depth: np.ndarray) -> None:
    values = np.asarray(depth, dtype=np.float32)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1) as archive:
        with archive.open('depth.npy', 'w') as stream:
            np.lib.format.write_array(stream, values, allow_pickle=False)


def _frame_metadata(part_id: np.ndarray, depth: np.ndarray, body_for_label: dict[int, int],
                    width: int, height: int) -> dict:
    labels = {}
    for label, link_index in sorted(body_for_label.items()):
        ys, xs = np.nonzero(part_id == label)
        if len(xs) == 0:
            labels[str(link_index)] = dict(part_id=label, link_index=link_index,
                                        visible=False, pixel_count=0,
                                        bbox_xyxy=None, centroid_xy=None, depth_median=None)
            continue
        labels[str(link_index)] = dict(
            part_id=label, link_index=link_index, visible=True, pixel_count=int(len(xs)),
            visible_fraction=float(len(xs) / (width * height)),
            bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            centroid_xy=[float(xs.mean()), float(ys.mean())],
            depth_median=float(np.nanmedian(depth[ys, xs])) if np.isfinite(depth[ys, xs]).any() else None,
        )
    return labels


def export_labels(sample_dir: Path, cameras: tuple[str, ...] = ('cam01', 'cam02'),
                  frame_stride: int = 1, write_depth: bool = True) -> dict:
    """Render labels from the sample scene without modifying RGB artifacts."""
    motion = np.load(sample_dir / 'motion.npz', allow_pickle=False)
    q = motion['q']; timestamps = motion['timestamps']; joint_order = motion['joint_order']
    runtime = json.loads((sample_dir / 'joint_runtime.json').read_text()) if (sample_dir / 'joint_runtime.json').exists() else None
    scene = sample_dir / 'scene_multiview.xml'
    if not scene.exists():
        raise FileNotFoundError(scene)
    camera_json = json.loads((sample_dir / 'camera_cam01.json').read_text())
    width, height = map(int, camera_json['resolution_wh'])
    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    geom_ids = set(range(model.ngeom))
    geom_body = {geom_id: int(model.geom_bodyid[geom_id]) for geom_id in geom_ids}
    # Each exported link body is named link_NN.  Internal kinematic bodies are
    # excluded from labels; their visible geoms are assigned to their nearest
    # exported link body by walking the MuJoCo body parent chain.
    link_body_ids = {}
    for index, name in enumerate(motion['link_order']):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
        if body_id >= 0:
            link_body_ids[index] = body_id
    labels_root = sample_dir / 'labels'; labels_root.mkdir(exist_ok=True)
    report = {'schema_version': '1.1', 'source_frame_count': int(len(q)),
              'frame_stride': frame_stride, 'cameras': {}}
    try:
        for camera in cameras:
            part_dir = labels_root / f'part_id_{camera}'; part_dir.mkdir(exist_ok=True)
            depth_dir = labels_root / f'depth_{camera}' if write_depth else None
            if depth_dir: depth_dir.mkdir(exist_ok=True)
            track_path = labels_root / f'tracks_{camera}.jsonl'
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            if camera_id < 0: raise ValueError(f'missing camera {camera}')
            frame_count = 0
            visible_labels = set()
            with track_path.open('w') as tracks:
                for frame_index in range(0, len(q), frame_stride):
                    set_qpos(model, data, q[frame_index], joint_order, runtime)
                    data.time = float(timestamps[frame_index]); mujoco.mj_forward(model, data)
                    renderer.enable_segmentation_rendering()
                    renderer.update_scene(data, camera=camera)
                    seg = renderer.render()
                    part_id = _segmentation_ids(seg, geom_ids)
                    renderer.disable_segmentation_rendering()
                    renderer.enable_depth_rendering()
                    renderer.update_scene(data, camera=camera)
                    depth = renderer.render().astype(np.float32)
                    renderer.disable_depth_rendering()
                    # Convert geom IDs to stable exported link labels. Label 0
                    # is background; labels are link index + 1.
                    stable = np.zeros_like(part_id, dtype=np.uint16)
                    body_to_link = {body: link for link, body in link_body_ids.items()}
                    for body_id in range(model.nbody):
                        current = body_id
                        while current not in body_to_link and current > 0:
                            current = int(model.body_parentid[current])
                        if current in body_to_link:
                            body_to_link[body_id] = body_to_link[current]
                    for geom_id, body_id in geom_body.items():
                        link = body_to_link.get(body_id)
                        if link is not None:
                            stable[part_id == geom_id + 1] = link + 1
                    imageio.imwrite(part_dir / f'{frame_index:06d}.png', stable)
                    if depth_dir: _write_depth(depth_dir / f'{frame_index:06d}.npz', depth)
                    bodies = {link + 1: link for link in link_body_ids}
                    record = dict(frame_index=frame_index, timestamp_s=float(timestamps[frame_index]),
                                  objects=_frame_metadata(stable, depth, bodies, width, height))
                    visible_labels.update(int(value) for value in np.unique(stable) if value)
                    tracks.write(json.dumps(record, ensure_ascii=False) + '\n')
                    frame_count += 1
            report['cameras'][camera] = dict(frame_count=frame_count, width=width, height=height,
                                              visible_label_count=len(visible_labels),
                                              visible_part_ids=sorted(visible_labels),
                                              part_id_dir=str(part_dir.relative_to(sample_dir)),
                                              depth_dir=str(depth_dir.relative_to(sample_dir)) if depth_dir else None,
                                              tracks=str(track_path.relative_to(sample_dir)))
    finally:
        renderer.close()
    (labels_root / 'quality.json').write_text(json.dumps(
             dict(status='labels_derived', source_frame_count=len(q), cameras=list(cameras),
             label_convention='0 background; link_index+1 foreground; uint16 PNG',
             depth_convention='camera z-depth in meters; float32 compressed NPZ (key: depth)', report=report),
        ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sample', type=Path)
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()
    export_labels(args.sample, frame_stride=args.stride)
