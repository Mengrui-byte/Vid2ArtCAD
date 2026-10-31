"""Extract official BrepGen surface/edge targets from local Fusion STEP parts.

Run with pythonocc-core and occwl installed and BrepGen/data_process on
PYTHONPATH. Assembly STEP is excluded: each link must remain a separate part.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import types
from pathlib import Path


def process_parts(source: Path, output: Path) -> dict:
    # occwl 3.0 uses the old deprecate(target=...) API; the PyPI package has
    # since changed it. This decorator only emits warnings, not CAD transforms.
    sys.modules["deprecate"] = types.SimpleNamespace(deprecated=lambda **_kwargs: lambda fn: fn)
    from OCC.Core import TopoDS

    # BrepGen imports the pre-7.9 aliases; OCC 7.9 exposes the same casts on topods.
    TopoDS.topods_Solid = TopoDS.topods.Solid
    TopoDS.topods_Shell = TopoDS.topods.Shell
    from occwl.io import load_step
    from occwl.solid import Solid
    from process_brep import load_step_with_occ, parse_solid

    output.mkdir(parents=True, exist_ok=True)
    records = []
    for step in sorted(source.glob("*/*.step")):
        if step.name == "assembly.step" or step.name.startswith("."):
            continue
        record = {"sample": step.parent.name, "body": step.stem, "source": str(step)}
        destination = output / step.parent.name / f"{step.stem}.pkl"
        try:
            if destination.is_file():
                with destination.open("rb") as stream:
                    target = pickle.load(stream)
                record.update(status="ready", target=str(destination), faces=len(target["surf_ncs"]),
                              edges=len(target["edge_ncs"]))
                records.append(record)
                continue
            solids = load_step(str(step))
            if not solids:
                solids = [Solid(shape) for shape in load_step_with_occ(str(step))]
            if len(solids) != 1:
                raise ValueError(f"expected one solid, found {len(solids)}")
            target = parse_solid(solids[0])
            if target is None:
                raise ValueError("exceeds BrepGen MAX_FACE threshold")
            target["uid"] = f"{step.parent.name}_{step.stem}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as stream:
                pickle.dump(target, stream, protocol=pickle.HIGHEST_PROTOCOL)
            record.update(status="ready", target=str(destination), faces=len(target["surf_ncs"]),
                          edges=len(target["edge_ncs"]))
        except Exception as error:
            record.update(status="failed", reason=f"{type(error).__name__}: {error}")
        records.append(record)
        print(f"{record['sample']}/{record['body']}: {record['status']}", flush=True)
    summary = {"ready": sum(r["status"] == "ready" for r in records),
               "failed": sum(r["status"] == "failed" for r in records), "parts": records}
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = process_parts(args.source, args.out)
    print(f"BrepGen targets: {summary['ready']} ready, {summary['failed']} failed")


if __name__ == "__main__":
    main()
