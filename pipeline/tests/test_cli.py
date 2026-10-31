import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.cli.main import main
from pipeline.render.task_queue import build_queue, claim_task, complete_task


class CliTest(unittest.TestCase):
    def test_init_and_run(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            main(["init", "--root", str(root)])
            asset = root / "raw" / "demo"
            asset.mkdir()
            (asset / "assembly.json").write_text("{}")
            with patch("pipeline.render.auto_annotation.run_dataset") as run_dataset:
                main(["run", "--root", str(root), "--asset", "demo"])
            self.assertEqual(run_dataset.call_args.args[:3], (root / "raw", Path("annoation"), ["demo"]))
            self.assertTrue((root / "ledger.jsonl").exists() is False)
            self.assertTrue((root / "staging" / "assets").exists())

    def test_queue_reopen_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / 'real' / 'clip'
            (sample / 'labels' / 'part_id_cam01').mkdir(parents=True)
            (sample / 'sample.json').write_text(json.dumps({
                'mode': 'real_video_annotation',
                'views': {'cam01': {'part_id': 'labels/part_id_cam01'}},
            }))
            (sample / 'review.json').write_text(json.dumps({'status': 'needs_review'}))
            queue_path = root / 'queue.json'
            build_queue(root, queue_path)
            task_id = 'clip:cam01:review'
            claim_task(queue_path, task_id, 'alice')
            complete_task(queue_path, task_id, status='failed', duration_s=4)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(['queue-reopen', '--queue', str(queue_path), '--task-id', task_id,
                      '--notes', 'fixed input'])
            task = json.loads(output.getvalue())
            self.assertEqual(task['status'], 'pending')
            self.assertEqual(task['attempt_history'][0]['duration_s'], 4)

