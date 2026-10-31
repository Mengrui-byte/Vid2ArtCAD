"""Render a review-only motion hypothesis for a body-level assembly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def _load_mesh(asset: Path, body_id: str) -> trimesh.Trimesh:
    mesh = trimesh.load(asset / f'{body_id}.obj', process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f'body {body_id} has no triangle mesh')
    return mesh


def render_hypothesis(asset: Path, moving_body_name: str, axis_body_name: str,
                      amplitude_rad: float, output: Path) -> dict:
    source = json.loads((asset / 'assembly.json').read_text())
    bodies = source.get('bodies') or {}
    body_by_name = {value.get('name'): body_id for body_id, value in bodies.items()}
    if moving_body_name not in body_by_name or axis_body_name not in body_by_name:
        raise ValueError('requested body names were not found in assembly.json')

    meshes = {body_id: _load_mesh(asset, body_id) for body_id in bodies}
    moving_id = body_by_name[moving_body_name]
    axis_id = body_by_name[axis_body_name]
    axis_mesh = meshes[axis_id]
    centered = axis_mesh.vertices - axis_mesh.vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    axis /= np.linalg.norm(axis)
    pivot = axis_mesh.vertices.mean(axis=0)

    combined = trimesh.util.concatenate(list(meshes.values()))
    root_visibility = (source.get('root') or {}).get('bodies') or {}
    visible_ids = {
        body_id for body_id in meshes
        if (root_visibility.get(body_id) or {}).get('is_visible', True)
    }
    if moving_id not in visible_ids:
        raise ValueError(f'moving body {moving_body_name} is marked invisible in root metadata')
    visible_meshes = {body_id: meshes[body_id] for body_id in visible_ids}
    visible_combined = trimesh.util.concatenate(list(visible_meshes.values()))
    center = visible_combined.bounds.mean(axis=0)
    extent = np.maximum(visible_combined.extents, 1e-5)
    poses = (-amplitude_rad, 0.0, amplitude_rad)
    figure = plt.figure(figsize=(15, 5), dpi=100)
    moving_mesh = meshes[moving_id]
    moving_center = moving_mesh.centroid
    centers = []
    for index, angle in enumerate(poses, start=1):
        rotation = trimesh.transformations.rotation_matrix(angle, axis, pivot)
        moved = moving_mesh.copy()
        moved.apply_transform(rotation)
        centers.append(moved.centroid.tolist())
        axes = figure.add_subplot(1, 3, index, projection='3d')
        axes.set_title(f'{angle:+.2f} rad' if angle else 'reference pose')
        axes.set_axis_off()
        axes.set_box_aspect(extent)
        for body_id, mesh in visible_meshes.items():
            shown = moved if body_id == moving_id else mesh
            faces = shown.faces
            if len(faces) > 1800:
                faces = faces[np.linspace(0, len(faces) - 1, 1800, dtype=int)]
            axes.plot_trisurf(
                shown.vertices[:, 0], shown.vertices[:, 1], shown.vertices[:, 2],
                triangles=faces,
                color='#dc7654' if body_id == moving_id else '#78958a',
                linewidth=0, antialiased=False,
            )
        pad = extent * 0.56
        axes.set_xlim(center[0] - pad[0], center[0] + pad[0])
        axes.set_ylim(center[1] - pad[1], center[1] + pad[1])
        axes.set_zlim(center[2] - pad[2], center[2] + pad[2])
        axes.view_init(elev=18, azim=38)

    figure.suptitle(f'{asset.name}: review-only motion hypothesis', fontsize=13)
    figure.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / 'motion_hypothesis.png', bbox_inches='tight')
    plt.close(figure)

    peak_to_peak = float(np.linalg.norm(np.asarray(centers[0]) - np.asarray(centers[2])))
    report = {
        'asset_id': asset.name,
        'status': 'hypothesis_preview_human_review_required',
        'provenance': 'synthetic_hypothesis',
        'moving_body': moving_body_name,
        'visible_body_count': len(visible_ids),
        'hidden_body_names': [bodies[body_id].get('name', body_id)
                              for body_id in meshes if body_id not in visible_ids],
        'axis_reference_body': axis_body_name,
        'axis_estimation': 'principal_component_1_of_axis_reference_mesh_vertices',
        'axis_direction': axis.tolist(),
        'pivot': pivot.tolist(),
        'amplitude_rad': amplitude_rad,
        'moving_body_centroid_at_minus_reference_plus': centers,
        'centroid_peak_to_peak_source_units': peak_to_peak,
        'centroid_peak_to_peak_over_asset_diagonal': peak_to_peak / float(np.linalg.norm(extent)),
        'limitations': [
            'Axis and pivot are geometric hypotheses, not source joint definitions.',
            'No collision/interpenetration or mechanical-limit validation was performed.',
            'Image is a three-pose review sheet, not an accepted video annotation.',
        ],
        'preview_image': 'motion_hypothesis.png',
    }
    (output / 'motion_hypothesis.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--moving-body', required=True)
    parser.add_argument('--axis-body', required=True)
    parser.add_argument('--amplitude-rad', type=float, default=0.15)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = render_hypothesis(args.asset, args.moving_body, args.axis_body,
                               args.amplitude_rad, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
