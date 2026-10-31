"""Forward kinematics and closure solving for normalized assembly graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from pipeline.kinematics.frames import joint_transform_dofs


@dataclass(frozen=True)
class JointSpec:
    id: str
    parent: str
    child: str
    type: str
    T_parent_joint: np.ndarray
    T_joint_child_ref: np.ndarray
    axes_in_joint: np.ndarray
    dof_names: tuple[str, ...]
    inverse_motion: bool = False

    @property
    def dof_count(self) -> int:
        return len(self.dof_names)


@dataclass(frozen=True)
class SolveResult:
    q: np.ndarray
    transforms: dict[str, np.ndarray]
    residual: np.ndarray
    success: bool
    message: str


class KinematicSolver:
    """Solve a spanning tree plus arbitrary closure joints.

    The q vector is global and follows ``dof_order``. Tree joints determine
    link poses; loop joints contribute residual equations and may be solved as
    dependent coordinates. Fixed/root placement is represented by the initial
    world transforms supplied to ``forward``.
    """

    def __init__(
        self,
        link_ids: Iterable[str],
        roots: Iterable[str],
        tree_joints: Iterable[JointSpec],
        loop_joints: Iterable[JointSpec] = (),
        root_transforms: dict[str, np.ndarray] | None = None,
    ):
        self.link_ids = tuple(link_ids)
        self.roots = tuple(roots)
        self.tree_joints = tuple(tree_joints)
        self.loop_joints = tuple(loop_joints)
        self.root_transforms = root_transforms or {}
        self.joints = self.tree_joints + self.loop_joints
        self.dof_order = tuple(name for joint in self.joints for name in joint.dof_names)
        self._offsets: dict[str, slice] = {}
        cursor = 0
        for joint in self.joints:
            self._offsets[joint.id] = slice(cursor, cursor + joint.dof_count)
            cursor += joint.dof_count
        self.dof_count = cursor
        self._children: dict[str, list[JointSpec]] = {link: [] for link in self.link_ids}
        for joint in self.tree_joints:
            if joint.parent not in self._children or joint.child not in self._children:
                raise ValueError(f"tree joint {joint.id} references unknown link")
            self._children[joint.parent].append(joint)
        if set(self.roots) - set(self.link_ids):
            raise ValueError("root is not a known link")

    def _joint_motion(self, joint: JointSpec, q: np.ndarray) -> np.ndarray:
        values = q[self._offsets[joint.id]]
        motion = joint_transform_dofs(joint.type, values, joint.axes_in_joint)
        return np.linalg.inv(motion) if joint.inverse_motion else motion

    def forward(self, q: np.ndarray) -> dict[str, np.ndarray]:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size != self.dof_count:
            raise ValueError(f"expected {self.dof_count} q values, got {q.size}")
        world = {link: np.eye(4) for link in self.link_ids}
        for root in self.roots:
            if root in self.root_transforms:
                world[root] = np.asarray(self.root_transforms[root], dtype=np.float64)

        def visit(parent: str) -> None:
            for joint in self._children[parent]:
                world[joint.child] = (
                    world[parent]
                    @ joint.T_parent_joint
                    @ self._joint_motion(joint, q)
                    @ joint.T_joint_child_ref
                )
                visit(joint.child)

        for root in self.roots:
            visit(root)
        return world

    @staticmethod
    def _pose_residual(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
        delta = np.linalg.inv(expected) @ actual
        return np.r_[delta[:3, 3], Rotation.from_matrix(delta[:3, :3]).as_rotvec()]

    def closure_residual(self, q: np.ndarray) -> np.ndarray:
        world = self.forward(q)
        residual = []
        for joint in self.loop_joints:
            predicted = (
                world[joint.parent]
                @ joint.T_parent_joint
                @ self._joint_motion(joint, q)
                @ joint.T_joint_child_ref
            )
            residual.append(self._pose_residual(world[joint.child], predicted))
        return np.concatenate(residual) if residual else np.zeros(0, dtype=np.float64)

    def solve(
        self,
        q_initial: np.ndarray | None = None,
        prescribed: dict[str, float] | None = None,
        max_nfev: int = 400,
    ) -> SolveResult:
        q0 = np.zeros(self.dof_count, dtype=np.float64) if q_initial is None else np.asarray(q_initial, dtype=np.float64).copy()
        if q0.size != self.dof_count:
            raise ValueError(f"expected {self.dof_count} q values, got {q0.size}")
        prescribed = prescribed or {}
        fixed: dict[int, float] = {}
        for name, value in prescribed.items():
            try:
                fixed[self.dof_order.index(name)] = float(value)
            except ValueError as exc:
                raise ValueError(f"unknown prescribed DOF: {name}") from exc
        variable = [index for index in range(self.dof_count) if index not in fixed]

        def assemble(values: np.ndarray) -> np.ndarray:
            q = q0.copy()
            q[variable] = values
            for index, value in fixed.items():
                q[index] = value
            return q

        def objective(values: np.ndarray) -> np.ndarray:
            return self.closure_residual(assemble(values))

        if not self.loop_joints:
            q = assemble(np.asarray([q0[index] for index in variable]))
            return SolveResult(q, self.forward(q), np.zeros(0), True, "no loop constraints")
        if not variable:
            q = assemble(np.zeros(0, dtype=np.float64))
            residual = self.closure_residual(q)
            return SolveResult(
                q, self.forward(q), residual,
                bool(np.allclose(residual, 0.0, atol=1e-9)),
                "all DOFs prescribed",
            )
        result = least_squares(
            objective, np.asarray([q0[index] for index in variable]),
            max_nfev=max_nfev, ftol=1e-13, xtol=1e-13, gtol=1e-13,
        )
        q = assemble(result.x)
        residual = self.closure_residual(q)
        return SolveResult(q, self.forward(q), residual, bool(result.success), result.message)
