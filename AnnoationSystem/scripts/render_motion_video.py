from __future__ import annotations
import argparse, json, math
from pathlib import Path
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f'cannot load triangle mesh: {path}')
    return mesh


def vec(value, fallback=None):
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return np.asarray(value, dtype=float)
    if isinstance(value, dict) and all(k in value for k in ('x','y','z')):
        return np.asarray([value['x'], value['y'], value['z']], dtype=float)
    return fallback


def joint_axis(assembly, design):
    ref = design.get('joint_or_axis_reference', '')
    joint = (assembly.get('joints') or {}).get(ref)
    if joint:
        frame = joint.get('geometry_or_origin_one') or joint.get('geometry_or_origin_two') or {}
        origin = vec(frame.get('origin'))
        axis = vec(frame.get('primary_axis_vector'))
        if origin is not None and axis is not None and np.linalg.norm(axis) > 1e-9:
            return origin, axis / np.linalg.norm(axis), 'source_joint'
    origin = vec(design.get('axis_origin'))
    axis = vec(design.get('axis_direction'))
    if origin is None or axis is None or np.linalg.norm(axis) <= 1e-9:
        raise ValueError('design needs axis_origin and axis_direction, or a valid source joint reference')
    return origin, axis / np.linalg.norm(axis), 'design_hypothesis'


def transform_mesh(mesh, matrix):
    out = mesh.copy()
    out.apply_transform(matrix)
    return out


def keyframe_value(keyframes, t):
    if not keyframes:
        return 0.0
    if t <= keyframes[0]['time_seconds']:
        return float(keyframes[0]['coordinate'])
    for left, right in zip(keyframes, keyframes[1:]):
        t0, t1 = float(left['time_seconds']), float(right['time_seconds'])
        if t <= t1:
            alpha = (t - t0) / max(t1 - t0, 1e-9)
            smooth = alpha * alpha * (3 - 2 * alpha)
            return float(left['coordinate']) * (1 - smooth) + float(right['coordinate']) * smooth
    return float(keyframes[-1]['coordinate'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', type=Path, required=True)
    ap.add_argument('--design', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--frames', type=int, default=60)
    ap.add_argument('--fps', type=int, default=20)
    args = ap.parse_args()
    assembly = json.loads((args.asset / 'assembly.json').read_text())
    design = json.loads(args.design.read_text())
    if design.get('motion_type') not in ('revolute', 'prismatic'):
        raise ValueError('only revolute and prismatic designs can be rendered as motion previews')
    names = {bid: body.get('name', bid) for bid, body in (assembly.get('bodies') or {}).items()}
    mesh_by_id = {}
    for bid, body in (assembly.get('bodies') or {}).items():
        obj = args.asset / body.get('obj', f'{bid}.obj')
        if obj.exists():
            mesh_by_id[bid] = load_mesh(obj)
    fixed = set(design.get('fixed_body_ids') or [])
    moving = set(design.get('moving_body_ids') or [])
    selected = fixed | moving
    selected &= set(mesh_by_id)
    if not fixed or not moving or not selected:
        raise ValueError('design must select existing fixed and moving mesh bodies')
    origin, axis, axis_source = joint_axis(assembly, design)
    qrange = design.get('coordinate_range') or {}
    lo, hi = float(qrange['minimum']), float(qrange['maximum'])
    keyframes = design.get('trajectory_keyframes') or [{'time_seconds': 0, 'coordinate': lo}, {'time_seconds': 1, 'coordinate': hi}, {'time_seconds': 2, 'coordinate': lo}]
    duration = float(design.get('duration_seconds') or max(float(keyframes[-1]['time_seconds']), 1.0))
    all_mesh = trimesh.util.concatenate([mesh_by_id[bid] for bid in selected])
    center = all_mesh.bounds.mean(axis=0)
    radius = max(float(np.linalg.norm(all_mesh.bounds[1] - all_mesh.bounds[0])) / 2, 1e-6) * 1.2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.output), fps=args.fps, codec='libx264')
    fig = plt.figure(figsize=(7, 7), dpi=100)
    for frame in range(args.frames):
        t = duration * frame / max(args.frames - 1, 1)
        q = keyframe_value(keyframes, t)
        rotation = trimesh.transformations.rotation_matrix(q, axis, origin)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off(); ax.set_box_aspect((1,1,1))
        ax.set_xlim(center[0]-radius, center[0]+radius); ax.set_ylim(center[1]-radius, center[1]+radius); ax.set_zlim(center[2]-radius, center[2]+radius)
        ax.view_init(elev=20, azim=35 + 8 * math.sin(2 * math.pi * frame / max(args.frames, 1)))
        for bid in selected:
            mesh = mesh_by_id[bid]
            shown = transform_mesh(mesh, rotation) if bid in moving and design.get('motion_type') == 'revolute' else mesh
            if design.get('motion_type') == 'prismatic' and bid in moving:
                offset = axis * q
                shown = mesh.copy(); shown.vertices += offset
            if len(shown.faces) > 1800:
                shown = shown.submesh([np.linspace(0, len(shown.faces)-1, 1800, dtype=int)], append=True, repair=False)
            color = '#d97757' if bid in moving else '#6f8f82'
            ax.plot_trisurf(shown.vertices[:,0], shown.vertices[:,1], shown.vertices[:,2], triangles=shown.faces, color=color, linewidth=0, antialiased=False)
        ax.set_title(f'{names.get(next(iter(moving)), "moving part")} | q={q:.3f} | design preview', fontsize=10)
        fig.canvas.draw(); writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()); fig.clear()
    writer.close(); plt.close(fig)
    metadata = {
        'status': 'synthetic_design_preview', 'provenance': 'llm_motion_design', 'collision_validation': 'not_run',
        'axis_source': axis_source, 'axis_origin': origin.tolist(), 'axis_direction': axis.tolist(),
        'coordinate_range': {'minimum': lo, 'maximum': hi, 'unit': qrange.get('unit')},
        'fixed_body_ids': sorted(fixed), 'moving_body_ids': sorted(moving), 'output': str(args.output.resolve()),
        'limitations': ['This video visualizes the saved design proposal.', 'It is not collision-validated and does not establish a mechanical limit.'],
    }
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata))

if __name__ == '__main__':
    main()
