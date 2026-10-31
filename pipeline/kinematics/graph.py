"""Topology normalization for arbitrary assembly graphs.

The renderer may serialize a spanning tree, but this module deliberately keeps
the source graph and closure edges.  A tree is an export choice, not the
kinematic model.
"""

from __future__ import annotations

from dataclasses import dataclass


_TYPE_MAP = {
    "RigidJointType": "fixed",
    "RevoluteJointType": "revolute",
    "SliderJointType": "prismatic",
    "CylindricalJointType": "cylindrical",
    "PinSlotJointType": "pin_slot",
    "BallJointType": "ball",
    "PlanarJointType": "planar",
}

_DOF_COUNT = {
    "fixed": 0,
    "revolute": 1,
    "prismatic": 1,
    "cylindrical": 2,
    "pin_slot": 2,
    "ball": 3,
    "planar": 3,
}


def canonical_joint_type(raw_type: str) -> str:
    """Map Fusion-style names to the canonical type vocabulary."""
    try:
        return _TYPE_MAP[raw_type]
    except KeyError as exc:
        raise ValueError(f"unsupported source joint type: {raw_type}") from exc


@dataclass(frozen=True)
class GraphJoint:
    id: str
    parent: str
    child: str
    type: str
    source: dict

    @property
    def dof_count(self) -> int:
        return _DOF_COUNT[self.type]


@dataclass(frozen=True)
class KinematicGraphModel:
    """Normalized graph plus an explicit spanning forest for serialization."""

    link_ids: tuple[str, ...]
    component_of: dict[str, str]
    joints: tuple[GraphJoint, ...]
    tree_joints: tuple[str, ...]
    loop_joints: tuple[str, ...]
    roots: tuple[str, ...]
    dof_order: tuple[str, ...]

    def joint(self, joint_id: str) -> GraphJoint:
        return next(j for j in self.joints if j.id == joint_id)


class _UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def normalize_graph(link_ids: list[str], raw_joints: dict[str, dict]) -> KinematicGraphModel:
    """Normalize a possibly disconnected, cyclic source assembly.

    Fixed joints are merged into rigid components.  Non-fixed joints are kept
    as graph edges; a deterministic spanning forest is selected for URDF-like
    export and every remaining edge is retained as a loop constraint.
    """
    links = list(dict.fromkeys(link_ids))
    known = set(links)
    uf = _UnionFind(links)
    parsed: list[GraphJoint] = []
    for joint_id, raw in raw_joints.items():
        parent, child = raw.get("occurrence_one"), raw.get("occurrence_two")
        if parent not in known or child not in known:
            raise ValueError(f"joint {joint_id} references unknown link")
        joint = GraphJoint(
            id=joint_id,
            parent=parent,
            child=child,
            type=canonical_joint_type((raw.get("joint_motion") or {}).get("joint_type")),
            source=raw,
        )
        parsed.append(joint)
        if joint.type == "fixed":
            uf.union(parent, child)

    components = {link: uf.find(link) for link in links}
    component_ids = tuple(dict.fromkeys(components.values()))
    # Keep one representative link per rigid component.  Consumers can map all
    # source occurrences back through component_of.
    component_of = {link: components[link] for link in links}
    active = [j for j in parsed if j.type != "fixed" and components[j.parent] != components[j.child]]

    tree: list[str] = []
    loops: list[str] = []
    # Stable ordering makes manifests reproducible across Python/process runs.
    forest = _UnionFind(list(component_ids))
    adjacency = {component: [] for component in component_ids}
    for joint in sorted(active, key=lambda item: item.id):
        a, b = components[joint.parent], components[joint.child]
        if forest.find(a) != forest.find(b):
            forest.union(a, b)
            tree.append(joint.id)
            adjacency[a].append((b, joint.id))
            adjacency[b].append((a, joint.id))
        else:
            loops.append(joint.id)

    roots: list[str] = []
    visited: set[str] = set()
    for start in component_ids:
        if start in visited:
            continue
        stack = [start]
        group: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            group.append(current)
            stack.extend(neighbor for neighbor, _ in adjacency[current])
        roots.append(min(group))
    dof_order = tuple(
        f"{joint.id}:{index}"
        for joint in sorted(active, key=lambda item: item.id)
        for index in range(joint.dof_count)
    )
    return KinematicGraphModel(
        link_ids=component_ids,
        component_of=component_of,
        joints=tuple(parsed),
        tree_joints=tuple(tree),
        loop_joints=tuple(loops),
        roots=tuple(roots),
        dof_order=dof_order,
    )
