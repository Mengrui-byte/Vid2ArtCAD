"""Simple deterministic task state machine and cache key."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline.core.schema import StageRecord, StageStatus


def cache_key(stage: str, input_hashes: list[str], config: dict, seed: int = 0) -> str:
    payload = json.dumps(
        {"stage": stage, "inputs": input_hashes, "config": config, "seed": seed},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class TaskLedger:
    def __init__(self, root: Path) -> None:
        self.path = root / "ledger.jsonl"

    def append(self, record: StageRecord) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record.__dict__, default=str) + "\n")

    def latest(self, task_key: str) -> StageRecord | None:
        if not self.path.exists():
            return None
        found = None
        with self.path.open() as f:
            for line in f:
                rec = StageRecord(**json.loads(line))
                if rec.task_key == task_key:
                    found = rec
        return found

    @staticmethod
    def can_reuse(record: StageRecord | None, expected_key: str) -> bool:
        return record is not None and record.status == StageStatus.DONE and record.config_hash == expected_key
