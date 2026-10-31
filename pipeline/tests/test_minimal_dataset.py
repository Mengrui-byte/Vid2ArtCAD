import json
import tempfile
import unittest
from pathlib import Path

from pipeline.render.minimal_dataset import package_minimal_dataset


class MinimalDatasetTest(unittest.TestCase):
    def test_copies_only_runtime_files_and_urdf_mesh_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            published = root / "published"
            source = published / "samples" / "asset_multiview"
            source.mkdir(parents=True)
            (source / "meshes").mkdir()
            (source / "rgb_cam01.mp4").write_bytes(b"one")
            (source / "rgb_cam02.mp4").write_bytes(b"two")
            (source / "meshes" / "link.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
            )
            (source / "assembly.urdf").write_text(
                '<robot name="asset"><link name="x"><visual><geometry>'
                '<mesh filename="meshes/link.obj" /></geometry></visual></link></robot>'
            )
            (source / "quality.json").write_text("discard")
            manifest = {"accepted": [{"sample_id": "asset_multiview", "path": "samples/asset_multiview"}], "rejected": []}
            (published / "dataset_manifest.json").write_text(json.dumps(manifest))
            cad = root / "cad" / "asset"
            cad.mkdir(parents=True)
            (cad / "assembly.step").write_bytes(b"step")

            output = root / "minimal"
            output.mkdir()
            result = package_minimal_dataset(published, root / "cad", output)

            self.assertEqual(result["sample_count"], 1)
            self.assertEqual(result["source_cad_root"], "cad")
            self.assertNotIn("/tmp/", result["source_manifest"])
            self.assertTrue((output / "asset" / "meshes" / "link.obj").is_file())
            self.assertTrue((output / "asset" / "assembly.step").is_file())
            self.assertTrue((output / "asset" / "video_cam01.mp4").is_file())
            self.assertTrue((output / "asset" / "video_cam02.mp4").is_file())
            self.assertFalse((output / "asset" / "rgb_cam01.mp4").exists())
            self.assertFalse((output / "asset" / "quality.json").exists())

            (source / "meshes" / "link.obj").unlink()
            with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                package_minimal_dataset(published, root / "cad", root / "broken")
            self.assertFalse((root / "broken").exists())

            occupied = root / "occupied"
            occupied.mkdir()
            sentinel = occupied / "keep.txt"
            sentinel.write_text("preserve")
            with self.assertRaisesRegex(ValueError, "empty directory"):
                package_minimal_dataset(published, root / "cad", occupied)
            self.assertEqual(sentinel.read_text(), "preserve")

            (source / "assembly.urdf").write_text(
                '<robot><mesh filename="../escape.obj" /></robot>'
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                package_minimal_dataset(published, root / "cad", root / "unsafe")
            self.assertFalse((root / "unsafe").exists())
