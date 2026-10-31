"""Core data contracts shared by all pipeline stages."""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional, Literal


class StrEnum(str, Enum):
    """Compatibility implementation for Python 3.9."""


class Source(StrEnum):
    SYNTHETIC_GT = "synthetic_gt"
    DATASET_REFERENCE = "dataset_reference"
    MEASURED = "measured"
    HUMAN_REFERENCE = "human_reference"
    PIPELINE_POLICY = "pipeline_policy"
    PSEUDO = "pseudo"
    UNKNOWN = "unknown"


class JointType(StrEnum):
    FIXED = "fixed"
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    CYLINDRICAL = "cylindrical"
    PIN_SLOT = "pin_slot"
    BALL = "ball"
    PLANAR = "planar"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    REVIEW = "review"
    FAILED = "failed"


class SampleStatus(StrEnum):
    ACCEPTED = "accepted"
    PARTIAL = "partial"
    REVIEW = "review"
    FAILED = "failed"


@dataclass
class SourceField:
    value: Any
    source: Source
    method: str | None = None
    confidence: Optional[float] = None
    observability: Optional[str] = None


@dataclass
class Limits:
    lower: float | None
    upper: float | None
    lower_enabled: bool = False
    upper_enabled: bool = False
    unit: Literal["rad", "m"] = "rad"


@dataclass
class Joint:
    id: str
    type: JointType
    parent: str
    child: str
    T_parent_joint: list[list[float]]
    T_joint_child_ref: list[list[float]]
    axis_in_joint: tuple[float, float, float] = (0.0, 0.0, 1.0)
    mechanical_limits: Limits | None = None
    coordinate_status: Literal["resolved", "unresolved"] = "unresolved"
    source: Source = Source.DATASET_REFERENCE


@dataclass
class Link:
    id: str
    part_id: int = 1
    geometry: str | None = None
    source_occurrences: list[str] = field(default_factory=list)


@dataclass
class CanonicalAsset:
    asset_id: str
    design_family_id: str
    links: list[Link]
    joints: list[Joint]
    root_link: str
    source_asset_id: str | None = None
    source_dataset: str | None = None
    units: tuple[str, str, str] = ("m", "rad", "s")
    transform_convention: str = "T_a_b maps column vectors from b to a"
    reference_pose: str = "canonical q=0 matches imported assembly"
    source_to_canonical: list[list[float]] | None = None
    cad_program: str | None = None


@dataclass
class MotionPlan:
    asset_id: str
    clip_id: str
    seed: int
    fps: int = 30
    duration_s: float = 4.0
    motion_policy: str = "quintic_segments_v1"
    sampling_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    executed_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class QualityReport:
    clip_id: str
    status: SampleStatus
    reason_codes: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    available_tasks: list[str] = field(default_factory=list)


@dataclass
class StageRecord:
    task_key: str
    stage: str
    asset_id: str
    clip_id: str | None = None
    status: StageStatus = StageStatus.PENDING
    config_hash: str | None = None
    input_paths: list[str] = field(default_factory=list)
    output_paths: list[str] = field(default_factory=list)
    error: str | None = None
