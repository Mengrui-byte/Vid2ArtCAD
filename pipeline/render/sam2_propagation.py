"""Optional SAM 2 video propagation into the Vid2ArtCAD pseudo-label contract."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from pipeline.render.file_lock import file_lock


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def _refresh_checksums(sample: Path, changed: list[Path]) -> None:
    manifest = sample / 'checksums.json'
    checksums = json.loads(manifest.read_text()) if manifest.exists() else {}
    for path in changed:
        checksums[str(path.relative_to(sample))] = hashlib.sha256(path.read_bytes()).hexdigest()
    _atomic_json(manifest, checksums)


def _load_predictor(model_id: str | None, config: Path | None, checkpoint: Path | None,
                    device: str | None):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("SAM2 requires PyTorch; install the optional SAM2 environment first") from exc
    try:
        if model_id:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            predictor = SAM2VideoPredictor.from_pretrained(model_id)
        else:
            if config is None or checkpoint is None:
                raise ValueError('SAM2 requires --model-id or both --config and --checkpoint')
            from sam2.build_sam import build_sam2_video_predictor
            target_device = device or ('cuda' if torch.cuda.is_available() else
                                       'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
                                       else 'cpu')
            predictor = build_sam2_video_predictor(str(config), str(checkpoint), device=target_device)
    except ImportError as exc:
        raise RuntimeError(
            "SAM2 package is not installed; install Meta's official SAM2 repository and its dependencies") from exc
    return predictor


def _video_frame_dir(video: Path, directory: Path) -> tuple[int, int, int]:
    reader = imageio.get_reader(video)
    count = 0
    width = height = 0
    try:
        for index, frame in enumerate(reader):
            frame = np.asarray(frame)
            if frame.ndim != 3 or frame.shape[2] < 3:
                raise ValueError(f'video frame must be RGB: {frame.shape}')
            height, width = frame.shape[:2]
            imageio.imwrite(directory / f'{index:06d}.jpg', frame[:, :, :3])
            count += 1
    finally:
        reader.close()
    if not count:
        raise ValueError('video contains no frames')
    return count, width, height


def propagate_sam2(sample: Path, seed_frame: int, model_id: str | None = None,
                   config: Path | None = None, checkpoint: Path | None = None,
                   device: str | None = None, view: str = 'cam01',
                   predictor=None, overwrite_pseudo: bool = False) -> dict:
    """Propagate reviewed masks with SAM2 while preserving the human-review gate."""
    sample = sample.resolve()
    lock_path = sample.parent / f'.{sample.name}.annotation.lock'
    with file_lock(lock_path):
        return _propagate_sam2_locked(sample, seed_frame, model_id, config, checkpoint,
                                      device, view, predictor, overwrite_pseudo)


def _propagate_sam2_locked(sample: Path, seed_frame: int, model_id: str | None = None,
                           config: Path | None = None, checkpoint: Path | None = None,
                           device: str | None = None, view: str = 'cam01',
                           predictor=None, overwrite_pseudo: bool = False) -> dict:
    sample = sample.resolve()
    video_name = 'rgb.mp4' if view == 'cam01' and (sample / 'rgb.mp4').exists() else f'rgb_{view}.mp4'
    video = sample / video_name
    if not video.is_file():
        raise FileNotFoundError(video)
    review_path = sample / ('review.json' if view == 'cam01' else f'review_{view}.json')
    if not review_path.exists():
        raise FileNotFoundError(review_path)
    review = json.loads(review_path.read_text())
    if seed_frame not in {int(value) for value in review.get('reviewed_frames', [])}:
        raise ValueError('seed frame must be human-reviewed before SAM2 propagation')
    label_dir = sample / 'labels' / f'part_id_{view}'
    seed_path = label_dir / f'{seed_frame:06d}.png'
    if not seed_path.is_file():
        raise FileNotFoundError(seed_path)
    seed = np.asarray(imageio.imread(seed_path))
    if seed.ndim != 2 or seed.dtype != np.uint16:
        raise ValueError('seed mask must be a uint16 single-channel part-id image')
    object_ids = [int(value) for value in np.unique(seed)
                  if 0 < int(value) < 65535]
    if not object_ids:
        raise ValueError('seed frame contains no foreground part IDs')
    predictor = predictor or _load_predictor(model_id, config, checkpoint, device)

    with tempfile.TemporaryDirectory(prefix='vid2artcad-sam2-') as temporary:
        frame_dir = Path(temporary) / 'frames'
        frame_dir.mkdir()
        frame_count, width, height = _video_frame_dir(video, frame_dir)
        state = predictor.init_state(video_path=str(frame_dir), offload_video_to_cpu=True)
        for object_id in object_ids:
            predictor.add_new_mask(state, frame_idx=seed_frame, obj_id=object_id,
                                   mask=(seed == object_id))
        outputs = {}
        for frame_index, output_ids, logits in predictor.propagate_in_video(state):
            scores = logits.detach().float().cpu().numpy()
            labels = np.zeros((height, width), dtype=np.uint16)
            confidence = np.zeros((height, width), dtype=np.float32)
            for slot, object_id in enumerate(output_ids):
                score = scores[slot]
                if score.ndim == 3:
                    score = score[0]
                if score.shape != (height, width):
                    try:
                        import torch
                        score = torch.nn.functional.interpolate(
                            torch.from_numpy(score)[None, None],
                            size=(height, width), mode='bilinear', align_corners=False
                        )[0, 0].numpy()
                    except ImportError as exc:
                        raise RuntimeError('SAM2 logits resolution differs from video and PyTorch is unavailable') from exc
                mask = score > 0.0
                better = mask & (score > confidence)
                labels[better] = int(object_id)
                confidence[better] = score[better]
            outputs[int(frame_index)] = (labels, confidence)
        if hasattr(predictor, 'reset_state'):
            predictor.reset_state(state)

    reviewed = {int(value) for value in review.get('reviewed_frames', [])}
    changed = []
    pseudo = review.setdefault('pseudo_frames', {})
    for frame_index, (labels, confidence) in sorted(outputs.items()):
        if frame_index in reviewed:
            continue
        path = label_dir / f'{frame_index:06d}.png'
        if path.exists() and not overwrite_pseudo and str(frame_index) not in pseudo:
            continue
        temporary = path.with_name(path.stem + '.tmp.png')
        imageio.imwrite(temporary, labels.astype(np.uint16))
        temporary.replace(path)
        pseudo[str(frame_index)] = {
            'source': 'sam2_pseudo', 'seed_frame': seed_frame,
            'object_ids': object_ids,
            'confidence_mean': float(np.mean(confidence[labels > 0])) if np.any(labels > 0) else 0.0,
            'confidence_foreground_fraction': float(np.mean(labels > 0)),
            'review_required': True,
        }
        changed.append(path)
    run = {
        'method': 'sam2_video_predictor', 'model_id': model_id,
        'config': str(config) if config else None,
        'checkpoint': str(checkpoint) if checkpoint else None,
        'device': device, 'view': view, 'seed_frame': seed_frame,
        'frame_count': frame_count, 'resolution_wh': [width, height],
        'object_ids': object_ids, 'reviewed_frames_unchanged': True,
    }
    review.setdefault('sam2_propagation_runs', []).append(run)
    _atomic_json(review_path, review)
    changed.append(review_path)
    _refresh_checksums(sample, changed)
    return {'status': 'pseudo_labels_generated', 'view': view,
            'written_frames': sorted(int(key) for key in pseudo if int(key) not in reviewed),
            'reviewed_frames': sorted(reviewed), 'object_ids': object_ids, 'method': run['method']}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Propagate reviewed masks with SAM2')
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--seed-frame', type=int, required=True)
    parser.add_argument('--model-id')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device')
    parser.add_argument('--view', default='cam01')
    parser.add_argument('--overwrite-pseudo', action='store_true')
    args = parser.parse_args()
    print(json.dumps(propagate_sam2(
        args.sample, args.seed_frame, args.model_id, args.config, args.checkpoint,
        args.device, args.view, overwrite_pseudo=args.overwrite_pseudo),
        ensure_ascii=False, indent=2))
