"""Apply the approved caption-only wording revision without changing video data."""
import hashlib
import json
from pathlib import Path

from pipeline.render.batch_tree_annotation import CONFIGS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('annoation')
    captions = {c['id']: c['description'] for c in CONFIGS}
    revised = []
    for path in sorted(root.rglob('motion_description.json')):
        data = json.loads(path.read_text())
        aid = data.get('asset_id')
        if aid not in captions:
            continue
        text_path = path.with_suffix('.txt')
        before = {p.name: sha(p) for p in [path, text_path] if p.exists()}
        data['description_cn'] = captions[aid]
        json_text = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
        text = captions[aid] + '\n'
        if path.read_text() == json_text and text_path.exists() and text_path.read_text() == text:
            continue
        path.write_text(json_text)
        text_path.write_text(text)
        after = {p.name: sha(p) for p in [path, text_path]}
        revision = path.parent / 'caption_revision.json'
        revision.write_text(json.dumps(dict(reason='User requested visible motion descriptions only',
            before_sha256=before, after_sha256=after, scope='caption text only; video, motion and calibration unchanged'),
            ensure_ascii=False, indent=2) + '\n')
        checksum = path.parent / 'checksums.json'
        if checksum.exists():
            hashes = json.loads(checksum.read_text())
            hashes.update(after)
            hashes[revision.name] = sha(revision)
            checksum.write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + '\n')
            for f in [path, text_path, revision]:
                assert sha(f) == hashes[f.name]
        assert json.loads(path.read_text())['description_cn'] == text_path.read_text().strip()
        revised.append(str(path.parent))
    print(json.dumps(dict(updated_directories=len(revised), paths=revised), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
