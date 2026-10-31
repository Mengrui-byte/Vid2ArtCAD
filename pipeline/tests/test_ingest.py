import json
import tempfile
import unittest
from pathlib import Path

from pipeline.render.ingest import ingest


class IngestTest(unittest.TestCase):
    def test_copies_asset_and_records_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source' / 'asset_a'
            source.mkdir(parents=True)
            (source / 'assembly.json').write_text('{}')
            report = ingest(root / 'source', root / 'dataset')
            self.assertEqual(report['assets'][0]['asset_id'], 'asset_a')
            self.assertTrue((root / 'dataset' / 'raw' / 'asset_a' / 'assembly.json').exists())
            self.assertTrue((root / 'dataset' / 'ingest_manifest.json').exists())
            self.assertIn('assembly.json', json.loads(
                (root / 'dataset' / 'ingest_manifest.json').read_text()
            )['assets'][0]['sha256'])


if __name__ == '__main__':
    unittest.main()
