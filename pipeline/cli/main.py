"""Phase-based CLI for the annotation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def init(root: Path) -> None:
    for path in (
        root / "raw",
        root / "staging" / "assets",
        root / "staging" / "clips",
        root / "published",
        root / "reports",
    ):
        path.mkdir(parents=True, exist_ok=True)
    print(f"initialized pipeline root: {root}")


def run(root: Path, asset: str, stages: str, data_root: Path | None = None,
        output_root: Path | None = None, preview: bool = False, strict: bool = False) -> None:
    requested = [item.strip() for item in stages.split(",") if item.strip()]
    supported = ["fusion", "motion", "render", "labels", "validate", "package"]
    if requested != supported:
        raise SystemExit("run currently executes the complete pipeline; use the default --stages")
    source_root = data_root or root / "raw"
    if not source_root.is_dir():
        raise SystemExit(f"asset data root does not exist: {source_root}; run ingest first")
    if asset and not (source_root / asset / "assembly.json").is_file():
        raise SystemExit(f"asset not found under {source_root}: {asset}")
    from pipeline.render.auto_annotation import run_dataset
    run_dataset(source_root, output_root or Path("annoation"), [asset] if asset else None,
                revolute=.5, slider=.01, max_active=2, preview=preview, strict=strict)


def status(root: Path) -> None:
    ledger = root / "ledger.jsonl"
    if not ledger.exists():
        print("no tasks")
        return
    counts: dict[str, int] = {}
    for line in ledger.read_text().splitlines():
        if '"status"' not in line:
            continue
        value = line.split('"status": "')[1].split('"')[0]
        counts[value] = counts.get(value, 0) + 1
    print(", ".join(f"{key}={value}" for key, value in sorted(counts.items())))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser("vid2artcad")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser("init")
    p_init.add_argument("--root", type=Path, default=Path("data"))

    p_ingest = subparsers.add_parser("ingest")
    p_ingest.add_argument("--root", type=Path, default=Path("data"))
    p_ingest.add_argument("--source", type=Path, required=True)
    p_ingest.add_argument("--force", action="store_true")

    p_run = subparsers.add_parser("run")
    p_run.add_argument("--root", type=Path, default=Path("data"))
    p_run.add_argument("--asset")
    p_run.add_argument("--stages", default="fusion,motion,render,labels,validate,package")
    p_run.add_argument("--data-root", type=Path)
    p_run.add_argument("--output-root", type=Path)
    p_run.add_argument("--preview", action="store_true")
    p_run.add_argument("--strict", action="store_true")

    p_status = subparsers.add_parser("status")
    p_status.add_argument("--root", type=Path, default=Path("data"))

    p_audit = subparsers.add_parser("audit")
    p_audit.add_argument("--data-root", type=Path, default=Path("data/raw/batch_001"))
    p_audit.add_argument("--output", type=Path, default=Path("annoation/dataset_audit.json"))

    p_package = subparsers.add_parser("package")
    p_package.add_argument("--output-root", type=Path, default=Path("annoation"))

    p_minimal = subparsers.add_parser("package-minimal")
    p_minimal.add_argument("--manifest-root", type=Path, required=True,
                           help="Published annotation root containing dataset_manifest.json")
    p_minimal.add_argument("--cad-root", type=Path, required=True,
                           help="Original CAD asset root containing assembly.step")
    p_minimal.add_argument("--destination", type=Path, required=True)

    p_validate = subparsers.add_parser("validate")
    p_validate.add_argument("--sample", type=Path, required=True)
    p_validate.add_argument("--no-video", action="store_true")

    p_edit = subparsers.add_parser("edit")
    p_edit.add_argument("--sample", type=Path, required=True)
    p_edit.add_argument("--output", type=Path, required=True)
    p_edit.add_argument("--joint", required=True)
    p_edit.add_argument("--scale", type=float, default=1.5)

    p_video = subparsers.add_parser("video-import")
    p_video.add_argument("--video", type=Path, required=True)
    p_video.add_argument("--output", type=Path, required=True)
    p_video.add_argument("--masks", type=Path)
    p_video.add_argument("--tracks", type=Path)
    p_video.add_argument("--camera", type=Path)
    p_video.add_argument("--source-label", default="pseudo")
    p_video.add_argument("--design-family-id")

    p_multiview = subparsers.add_parser("video-import-multiview")
    p_multiview.add_argument("--view", action="append", required=True,
                             help="named video as view=path; repeat for cam01/cam02")
    p_multiview.add_argument("--mask-view", action="append",
                             help="named mask directory as view=path")
    p_multiview.add_argument("--camera-view", action="append",
                             help="named camera JSON as view=path")
    p_multiview.add_argument("--tracks-view", action="append",
                             help="named tracks JSONL as view=path")
    p_multiview.add_argument("--output", type=Path, required=True)
    p_multiview.add_argument("--source-label", default="pseudo")
    p_multiview.add_argument("--design-family-id")

    p_review = subparsers.add_parser("review")
    p_review.add_argument("--sample", type=Path, required=True)
    p_review.add_argument("--video", default="rgb.mp4")
    p_review.add_argument("--view", help="multiview name, e.g. cam02")
    p_review.add_argument("--host", default="127.0.0.1")
    p_review.add_argument("--port", type=int, default=8765)

    p_propagate = subparsers.add_parser("propagate")
    p_propagate.add_argument("--sample", type=Path, required=True)
    p_propagate.add_argument("--seed-frame", type=int, required=True)
    p_propagate.add_argument("--radius", type=int, default=30)
    p_propagate.add_argument("--view", default="cam01")

    p_decode = subparsers.add_parser("decode-motion")
    p_decode.add_argument("--reference", type=Path, required=True)
    p_decode.add_argument("--masks", type=Path, action="append", required=True)
    p_decode.add_argument("--camera", type=Path, action="append", required=True)
    p_decode.add_argument("--output", type=Path, required=True)
    p_decode.add_argument("--label-map", type=Path)
    p_decode.add_argument("--residual-threshold-px", type=float, default=40.0)
    p_decode.add_argument("--joint", action="append", dest="active_joints")

    p_sam2 = subparsers.add_parser("sam2-propagate")
    p_sam2.add_argument("--sample", type=Path, required=True)
    p_sam2.add_argument("--seed-frame", type=int, required=True)
    p_sam2.add_argument("--model-id")
    p_sam2.add_argument("--config", type=Path)
    p_sam2.add_argument("--checkpoint", type=Path)
    p_sam2.add_argument("--device")
    p_sam2.add_argument("--view", default="cam01")
    p_sam2.add_argument("--overwrite-pseudo", action="store_true")

    p_track = subparsers.add_parser("cotracker-track")
    p_track.add_argument("--sample", type=Path, required=True)
    p_track.add_argument("--view", default="cam01")
    p_track.add_argument("--query-frame", type=int, default=0)
    p_track.add_argument("--points-per-part", type=int, default=16)
    p_track.add_argument("--grid-size", type=int, default=10)
    p_track.add_argument("--device")

    p_queue = subparsers.add_parser("queue")
    p_queue.add_argument("--output-root", type=Path, default=Path("annoation"))
    p_queue.add_argument("--data-root", type=Path, default=Path("data/raw/batch_001"))
    p_queue.add_argument("--queue-path", type=Path)
    p_queue.add_argument("--hourly-rate-usd", type=float, default=25.0)

    p_claim = subparsers.add_parser("queue-claim")
    p_claim.add_argument("--queue", type=Path, required=True)
    p_claim.add_argument("--task-id", required=True)
    p_claim.add_argument("--worker", required=True)

    p_complete = subparsers.add_parser("queue-complete")
    p_complete.add_argument("--queue", type=Path, required=True)
    p_complete.add_argument("--task-id", required=True)
    p_complete.add_argument("--status", choices=["done", "review", "failed"], default="done")
    p_complete.add_argument("--duration-s", type=float)
    p_complete.add_argument("--notes")

    p_reopen = subparsers.add_parser("queue-reopen")
    p_reopen.add_argument("--queue", type=Path, required=True)
    p_reopen.add_argument("--task-id", required=True)
    p_reopen.add_argument("--notes")

    p_queue_server = subparsers.add_parser("queue-server")
    p_queue_server.add_argument("--output-root", type=Path, default=Path("annoation"))
    p_queue_server.add_argument("--data-root", type=Path, default=Path("data/raw/batch_001"))
    p_queue_server.add_argument("--queue-path", type=Path)
    p_queue_server.add_argument("--hourly-rate-usd", type=float, default=25.0)
    p_queue_server.add_argument("--host", default="127.0.0.1")
    p_queue_server.add_argument("--port", type=int, default=8766)

    p_acceptance = subparsers.add_parser("acceptance-audit")
    p_acceptance.add_argument("--data-root", type=Path, default=Path("data/raw/batch_001"))
    p_acceptance.add_argument("--output-root", type=Path, default=Path("annoation"))
    p_acceptance.add_argument("--output", type=Path, default=Path("annoation/acceptance_audit.json"))
    p_acceptance.add_argument("--check-video", action="store_true")

    p_expand = subparsers.add_parser("expand-views")
    p_expand.add_argument("--sample", type=Path, required=True)
    p_expand.add_argument("--offset-degrees", type=float, nargs="+", default=[90.0, 270.0])
    p_expand.add_argument("--overwrite", action="store_true")

    p_preview_review = subparsers.add_parser("preview-review")
    p_preview_review.add_argument("--data-root", type=Path, default=Path("data/raw/batch_001"))
    p_preview_review.add_argument("--output-root", type=Path, default=Path("annoation"))
    p_preview_review.add_argument("--asset", required=True)
    p_preview_review.add_argument("--decision", choices=["approved", "rejected"], required=True)
    p_preview_review.add_argument("--reviewer", required=True)
    p_preview_review.add_argument("--notes", required=True)

    args = parser.parse_args(argv)
    if args.command == "init":
        init(args.root)
    elif args.command == "ingest":
        from pipeline.render.ingest import ingest
        report = ingest(args.source, args.root, args.force)
        print(f"ingested assets={len(report['assets'])} output={args.root / 'ingest_manifest.json'}")
    elif args.command == "run":
        if not args.asset and args.data_root is None:
            raise SystemExit("run requires --asset or --data-root")
        run(args.root, args.asset, args.stages, args.data_root, args.output_root,
            args.preview, args.strict)
    elif args.command == "audit":
        from pipeline.render.dataset_audit import audit_dataset
        report = audit_dataset(args.data_root, args.output)
        print(f"audited assets={report['asset_count']} output={args.output}")
    elif args.command == "package":
        from pipeline.render.package_dataset import package_dataset
        report = package_dataset(args.output_root)
        print(f"packaged samples={report['sample_count']} output={args.output_root}")
    elif args.command == "package-minimal":
        from pipeline.render.minimal_dataset import package_minimal_dataset
        report = package_minimal_dataset(args.manifest_root, args.cad_root, args.destination)
        print(f"packaged minimal samples={report['sample_count']} output={args.destination}")
    elif args.command == "validate":
        from pipeline.render.validate_sample import validate_sample
        report = validate_sample(args.sample, not args.no_video)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report['status'] != 'accepted':
            raise SystemExit(1)
    elif args.command == "edit":
        from pipeline.render.edit_pairs import create_edit_pair
        report = create_edit_pair(args.sample, args.output, args.joint, args.scale)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "video-import":
        from pipeline.render.video_annotation import import_video
        report = import_video(args.video, args.output, args.masks, args.tracks,
                              args.camera, args.source_label, args.design_family_id)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "video-import-multiview":
        def named_paths(items):
            result = {}
            for item in items or []:
                if '=' not in item:
                    raise SystemExit('named view arguments must use view=path')
                name, path = item.split('=', 1)
                if not name or not path or name in result:
                    raise SystemExit(f'invalid or duplicate named view: {item}')
                result[name] = Path(path)
            return result
        from pipeline.render.video_annotation import import_multiview
        report = import_multiview(
            named_paths(args.view), args.output, named_paths(args.mask_view),
            named_paths(args.camera_view), named_paths(args.tracks_view),
            args.source_label, args.design_family_id)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "review":
        from pipeline.render.review_server import serve
        video = args.video
        if args.view:
            video = f'rgb_{args.view}.mp4'
        serve(args.sample, video, args.host, args.port)
    elif args.command == "propagate":
        from pipeline.render.video_propagation import propagate
        print(json.dumps(propagate(args.sample, args.seed_frame, args.radius, view=args.view),
                         ensure_ascii=False, indent=2))
    elif args.command == "decode-motion":
        if len(args.masks) != len(args.camera):
            raise SystemExit("decode-motion requires one --camera for each --masks")
        from pipeline.render.motion_decode import decode_motion
        print(json.dumps(decode_motion(
            args.reference, args.masks, args.camera, args.output,
            args.label_map, args.residual_threshold_px, args.active_joints),
            ensure_ascii=False, indent=2))
    elif args.command == "sam2-propagate":
        from pipeline.render.sam2_propagation import propagate_sam2
        print(json.dumps(propagate_sam2(
            args.sample, args.seed_frame, args.model_id, args.config,
            args.checkpoint, args.device, args.view,
            overwrite_pseudo=args.overwrite_pseudo),
            ensure_ascii=False, indent=2))
    elif args.command == "cotracker-track":
        from pipeline.render.cotracker_tracks import track_cotracker
        print(json.dumps(track_cotracker(
            args.sample, args.view, args.query_frame, args.points_per_part,
            args.grid_size, args.device), ensure_ascii=False, indent=2))
    elif args.command == "queue":
        from pipeline.render.task_queue import build_queue
        print(json.dumps(build_queue(args.output_root, args.queue_path, args.hourly_rate_usd,
                                     args.data_root),
                         ensure_ascii=False, indent=2))
    elif args.command == "queue-claim":
        from pipeline.render.task_queue import claim_task
        print(json.dumps(claim_task(args.queue, args.task_id, args.worker),
                         ensure_ascii=False, indent=2))
    elif args.command == "queue-complete":
        from pipeline.render.task_queue import complete_task
        print(json.dumps(complete_task(args.queue, args.task_id, args.status,
                                       args.duration_s, args.notes), ensure_ascii=False, indent=2))
    elif args.command == "queue-reopen":
        from pipeline.render.task_queue import reopen_task
        print(json.dumps(reopen_task(args.queue, args.task_id, args.notes),
                         ensure_ascii=False, indent=2))
    elif args.command == "queue-server":
        from pipeline.render.queue_server import serve
        serve(args.output_root, args.queue_path, args.host, args.port, args.hourly_rate_usd,
              args.data_root)
    elif args.command == "acceptance-audit":
        from pipeline.render.acceptance_audit import audit_acceptance
        print(json.dumps(audit_acceptance(args.data_root, args.output_root, args.output,
                                           args.check_video), ensure_ascii=False, indent=2))
    elif args.command == "expand-views":
        from pipeline.render.expand_views import expand_views
        print(json.dumps(expand_views(args.sample, args.offset_degrees, args.overwrite),
                         ensure_ascii=False, indent=2))
    elif args.command == "preview-review":
        from pipeline.render.auto_annotation import record_preview_decision
        print(json.dumps(record_preview_decision(
            args.data_root, args.output_root, args.asset, args.decision,
            args.reviewer, args.notes), ensure_ascii=False, indent=2))
    else:
        status(args.root)


if __name__ == "__main__":
    main()
