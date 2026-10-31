"""Build a canonical kinematic model from Fusion-style assembly JSON.

This module is intentionally renderer-independent.  It is the single place
where source occurrence frames, source joint geometry, graph topology, and
canonical joint DOFs are combined.  Renderers consume the resulting
``KinematicSolver`` and must not reinterpret source JSON themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from pipeline.fusion.adapter import source_joint_dof_spec
from pipeline.kinematics.graph import canonical_joint_type
from pipeline.kinematics.solver import JointSpec, KinematicSolver


@dataclass(frozen=True)
class SourceEdge:
    id: str
    name: str
    parent: str
    child: str
    type: str
    source: dict
    spec: JointSpec


@dataclass(frozen=True)
class SourceKinematicModel:
    links: tuple[str, ...]
    edges: tuple[SourceEdge, ...]
    tree: tuple[SourceEdge, ...]
    loops: tuple[SourceEdge, ...]
    roots: tuple[str, ...]
    solver: KinematicSolver


class _UF:
    def __init__(self, values: Iterable[str]):
        self.p = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.p[value]
        if parent != value:
            self.p[value] = self.find(parent)
        return self.p[value]

    def union(self, left: str, right: str) -> bool:
        left, right = self.find(left), self.find(right)
        if left == right:
            return False
        self.p[right] = left
        return True


def _raw_vec(value: dict | None) -> np.ndarray:
    value = value or {}
    return np.asarray([value.get(axis, 0.0) for axis in ('x', 'y', 'z')], dtype=float)


def _unit(value: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1e-10:
        raise ValueError(f'{label} is a zero vector')
    return value / norm


def _orthogonal(axis: np.ndarray) -> np.ndarray:
    candidates = np.eye(3)
    candidate = candidates[int(np.argmin(np.abs(candidates @ axis)))]
    return _unit(candidate - axis * np.dot(candidate, axis), 'generated joint basis')


def _basis(raw_type: str, geometry: dict, occurrence_transform: np.ndarray) -> np.ndarray:
    primary = _unit(occurrence_transform[:3, :3] @ _raw_vec(geometry.get('primary_axis_vector')), 'primary joint axis')
    secondary = geometry.get('secondary_axis_vector')
    tertiary = geometry.get('tertiary_axis_vector')
    if raw_type == 'planar':
        # Fusion describes the plane normal as primary; canonical planar DOFs
        # are translation along two in-plane axes followed by normal rotation.
        axes = [_raw_vec(secondary), _raw_vec(tertiary), _raw_vec(geometry.get('primary_axis_vector'))]
    elif raw_type in {'ball'}:
        axes = [_raw_vec(geometry.get('primary_axis_vector')), _raw_vec(secondary), _raw_vec(tertiary)]
    elif raw_type == 'cylindrical':
        axes = [_raw_vec(geometry.get('primary_axis_vector'))]
    elif raw_type == 'pin_slot':
        axes = [_raw_vec(geometry.get('primary_axis_vector')), _raw_vec(secondary)]
    else:
        axes = [_raw_vec(geometry.get('primary_axis_vector'))]
    result = []
    for index, axis in enumerate(axes):
        if np.linalg.norm(axis) < 1e-10:
            if index == 0:
                axis = primary
            else:
                axis = _orthogonal(result[0])
        axis = _unit(occurrence_transform[:3, :3] @ axis, f'joint axis {index}')
        for previous in result:
            axis = axis - previous * np.dot(axis, previous)
        axis = _unit(axis, f'joint axis {index}')
        result.append(axis)
    if len(result) == 3 and np.dot(np.cross(result[0], result[1]), result[2]) < 0:
        result[2] *= -1
    return np.asarray(result)


def _joint_frame(raw: dict, canonical_occurrence: dict[str, np.ndarray], parent: str,
                 source_to_canonical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    geometry = raw.get('geometry_or_origin_one') or raw.get('geometry_or_origin_two') or {}
    origin = _raw_vec(geometry.get('origin')) * 0.01
    rotation = canonical_occurrence[parent][:3, :3]
    # Fusion joint origins are assembly/world coordinates, while axis vectors
    # are occurrence-local.  Mixing those frames is the most common source of
    # plausible-looking but numerically wrong annotations.
    world_origin = source_to_canonical[:3, :3] @ origin + source_to_canonical[:3, 3]
    frame = np.eye(4)
    frame[:3, 3] = world_origin
    type_name = canonical_joint_type((raw.get('joint_motion') or {}).get('joint_type', ''))
    if type_name != 'fixed':
        axes = _basis(type_name, geometry, canonical_occurrence[parent])
        if len(axes) == 1:
            axes = np.asarray([axes[0], _orthogonal(axes[0]), np.cross(axes[0], _orthogonal(axes[0]))])
        elif len(axes) == 2:
            axes = np.asarray([axes[0], axes[1], np.cross(axes[0], axes[1])])
        frame[:3, :3] = np.column_stack(axes)
    else:
        frame[:3, :3] = canonical_occurrence[parent][:3, :3]
    return frame, _basis(type_name, geometry, canonical_occurrence[parent]) if type_name != 'fixed' else np.zeros((0, 3))


def build_source_model(
    link_ids: Iterable[str],
    occurrence_transforms: dict[str, np.ndarray],
    raw_joints: dict[str, dict],
    root: str,
    source_to_canonical: np.ndarray | None = None,
) -> SourceKinematicModel:
    links = tuple(link_ids)
    known = set(links)
    if root not in known:
        raise ValueError(f'unknown kinematic root: {root}')
    source_to_canonical = np.eye(4) if source_to_canonical is None else np.asarray(source_to_canonical, dtype=float)
    edges: list[SourceEdge] = []
    for joint_id, raw in sorted(raw_joints.items()):
        parent, child = raw.get('occurrence_one'), raw.get('occurrence_two')
        if parent not in known or child not in known:
            raise ValueError(f'joint {joint_id} references unknown occurrence')
        type_name = canonical_joint_type((raw.get('joint_motion') or {}).get('joint_type', ''))
        spec = source_joint_dof_spec(raw)
        if spec['coordinate_status'] == 'unresolved' and type_name != 'fixed':
            raise ValueError(f'{joint_id} has unresolved joint geometry')
        frame, _resolved_axes = _joint_frame(raw, occurrence_transforms, parent, source_to_canonical)
        parent_T = np.linalg.inv(occurrence_transforms[parent]) @ frame
        child_ref = np.linalg.inv(frame) @ occurrence_transforms[child]
        names = tuple(spec['dof_names'])
        # The joint frame already carries the resolved world orientation.  The
        # solver axes are therefore coordinates *inside* that frame, not world
        # vectors.  This distinction matters for translated and rotated joints.
        axis_count = {'fixed': 0, 'cylindrical': 1, 'revolute': 1, 'prismatic': 1,
                      'pin_slot': 2, 'ball': 3, 'planar': 3}[type_name]
        axes = np.eye(axis_count, 3, dtype=float)
        joint_spec = JointSpec(joint_id, parent, child, type_name, parent_T, child_ref, axes, names)
        edges.append(SourceEdge(joint_id, raw.get('name', joint_id), parent, child, type_name, raw, joint_spec))

    uf = _UF(links)
    tree_ids: set[str] = set()
    for edge in edges:
        if uf.union(edge.parent, edge.child):
            tree_ids.add(edge.id)
    components: dict[str, list[str]] = {}
    for link in links:
        rep = uf.find(link)
        components.setdefault(rep, []).append(link)
    if root not in known:
        raise ValueError(f'unknown kinematic root: {root}')
    roots = (root,) + tuple(
        members[0] for members in components.values()
        if uf.find(members[0]) != uf.find(root)
    )
    tree = tuple(edge for edge in edges if edge.id in tree_ids)
    loops = tuple(edge for edge in edges if edge.id not in tree_ids)
    # Orient the forest away from the configured root.  Reverse traversal is
    # represented by inverse joint motion, preserving the source zero pose.
    adjacency = {link: [] for link in links}
    for edge in tree:
        adjacency[edge.parent].append((edge.child, edge, False))
        adjacency[edge.child].append((edge.parent, edge, True))
    oriented: list[JointSpec] = []
    seen = set(roots)
    queue = list(roots)
    while queue:
        parent = queue.pop(0)
        for child, edge, reverse in adjacency[parent]:
            if child in seen:
                continue
            seen.add(child)
            queue.append(child)
            source = edge.spec
            if not reverse:
                oriented.append(source)
            else:
                oriented.append(JointSpec(source.id, parent, child, source.type,
                    np.linalg.inv(source.T_joint_child_ref), np.linalg.inv(source.T_parent_joint),
                    source.axes_in_joint, source.dof_names, True))
    if len(seen) != len(links):
        raise ValueError('kinematic forest does not cover every source occurrence')
    loop_specs = [edge.spec for edge in loops]
    root_transforms = {component_root: occurrence_transforms[component_root]
                       for component_root in roots}
    solver = KinematicSolver(links, roots, oriented, loop_specs,
        root_transforms=root_transforms)
    return SourceKinematicModel(
        links, tuple(edges),
        tuple(next(e for e in tree if e.id == j.id) for j in oriented),
        loops, roots, solver)
