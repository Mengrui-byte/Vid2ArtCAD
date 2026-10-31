"""Rank joint-free CAD assemblies for human review of motion hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MOTION_CUES = (
    'lid', 'cover', 'door', 'hinge', 'wheel', 'roller', 'slider', 'handle',
    'lever', 'arm', 'pendulum', 'button', 'switch', 'knob', 'shaft', 'drawer',
    'flap', 'fold', 'pinion', 'gear', 'ram', 'jaw', 'screw', 'pedal', 'disc',
)

REVIEW_CONTRACT = {
    'provenance': 'synthetic_hypothesis',
    'release_requires_human_approval': True,
    'required_annotation_fields': [
        'fixed_occurrences', 'moving_occurrences', 'motion_type',
        'axis_or_direction_world', 'pivot_world', 'amplitude',
        'evidence', 'reviewer', 'approved',
    ],
    'preliminary_visual_gates': {
        'projected_peak_to_peak_motion_px_at_1080p': {'minimum': 40, 'status': 'pilot_threshold'},
        'moving_part_visible_fraction_of_frames': {'minimum': 0.8, 'status': 'pilot_threshold'},
        'return_to_start_error_fraction_of_travel': {'maximum': 0.01, 'status': 'pilot_threshold'},
        'collision_or_interpenetration': 'must_pass_geometry_and_frame_review',
        'motion_semantics': 'reviewer_confirms_motion_is_meaningful_and_not_incidental',
    },
    'threshold_note': 'Pilot thresholds require calibration; they are not validated dataset-wide acceptance limits.',
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def inspect_asset(asset: Path) -> dict | None:
    try:
        source = _load(asset / 'assembly.json')
    except (OSError, ValueError, TypeError):
        return None
    if source.get('joints'):
        return None

    occurrences = source.get('occurrences') or {}
    if len(occurrences) < 2:
        return None
    names = {key: str(value.get('name') or key)
             for key, value in occurrences.items() if isinstance(value, dict)}
    cues = sorted({cue for name in names.values() for cue in MOTION_CUES
                   if cue in name.casefold()})
    body_to_occurrence = {
        body_id: occurrence_id
        for occurrence_id, occurrence in occurrences.items()
        if isinstance(occurrence, dict)
        for body_id in (occurrence.get('bodies') or {})
    }

    cross_occurrence_contact_pairs: dict[tuple[str, str], int] = {}
    for contact in source.get('contacts') or []:
        if not isinstance(contact, dict):
            continue
        first = (contact.get('entity_one') or {}).get('occurrence')
        second = (contact.get('entity_two') or {}).get('occurrence')
        if first in names and second in names and first != second:
            pair = tuple(sorted((first, second)))
            cross_occurrence_contact_pairs[pair] = cross_occurrence_contact_pairs.get(pair, 0) + 1

    mesh_body_ids = {
        body_id for body_id in body_to_occurrence
        if (asset / f'{body_id}.obj').is_file()
    }
    occurrence_body_count = len(body_to_occurrence)
    root_bodies = (source.get('root') or {}).get('bodies') or {}
    hole_count = len(source.get('holes') or [])
    contact_pair_count = len(cross_occurrence_contact_pairs)
    score = (
        3 * len(cues)
        + min(contact_pair_count, 8)
        + min(hole_count, 8)
        + (2 if occurrence_body_count and len(mesh_body_ids) == occurrence_body_count else 0)
    )
    reasons = []
    if not cues:
        reasons.append('no_semantic_motion_cue_in_occurrence_names')
    if not cross_occurrence_contact_pairs:
        reasons.append('no_cross_occurrence_contact_evidence')
    if not hole_count:
        reasons.append('no_hole_or_axis_feature_evidence')
    if occurrence_body_count != len(mesh_body_ids):
        reasons.append('one_or_more_occurrence_meshes_missing')
    if root_bodies:
        reasons.append('root_bodies_need_explicit_rigid_group_policy')
    if any('Component' in name and not cues for name in names.values()):
        reasons.append('generic_component_names_limit_semantic_confidence')

    return {
        'asset_id': asset.name,
        'score': score,
        'occurrence_count': len(occurrences),
        'occurrence_names': list(names.values()),
        'name_cues': cues,
        'cross_occurrence_contact_pair_count': contact_pair_count,
        'cross_occurrence_contact_pairs': [
            {'parts': [names[first], names[second]], 'contact_count': count}
            for (first, second), count in sorted(
                cross_occurrence_contact_pairs.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        'hole_feature_count': hole_count,
        'occurrence_body_count': occurrence_body_count,
        'occurrence_mesh_count': len(mesh_body_ids),
        'root_body_count': len(root_bodies),
        'review_reasons': reasons,
        'motion_hypothesis': None,
        'status': 'human_review_required',
    }


def build_report(data_root: Path) -> dict:
    candidates = [record for asset in sorted(data_root.iterdir())
                  if asset.is_dir() and (record := inspect_asset(asset)) is not None]
    candidates.sort(key=lambda record: (-record['score'], record['asset_id']))
    return {
        'schema_version': '1.0',
        'data_root': str(data_root.resolve()),
        'policy': 'evidence ranks candidates only; no motion axis or trajectory is inferred',
        'review_contract': REVIEW_CONTRACT,
        'summary': {
            'candidate_count': len(candidates),
            'with_name_cues': sum(bool(item['name_cues']) for item in candidates),
            'with_cross_occurrence_contacts': sum(
                bool(item['cross_occurrence_contact_pairs']) for item in candidates
            ),
            'with_hole_features': sum(bool(item['hole_feature_count']) for item in candidates),
            'all_occurrence_meshes_present': sum(
                item['occurrence_body_count'] > 0
                and item['occurrence_body_count'] == item['occurrence_mesh_count']
                for item in candidates
            ),
        },
        'candidates': candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output.resolve()), **report['summary']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
