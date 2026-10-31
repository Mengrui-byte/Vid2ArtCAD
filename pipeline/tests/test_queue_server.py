import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from pipeline.render.queue_server import make_handler
from pipeline.render.task_queue import build_queue, claim_task, complete_task


class QueueServerTest(unittest.TestCase):
    def test_dashboard_reopens_failed_task_and_escapes_task_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            sample = output_root / 'real' / "clip_'onmouseover='x"
            (sample / 'labels' / 'part_id_cam01').mkdir(parents=True)
            (sample / 'sample.json').write_text(json.dumps({
                'mode': 'real_video_annotation',
                'views': {'cam01': {'part_id': 'labels/part_id_cam01'}},
            }))
            (sample / 'review.json').write_text(json.dumps({'status': 'needs_review'}))
            queue_path = output_root / 'queue.json'
            build_queue(output_root, queue_path)
            task_id = f"{sample.name}:cam01:review"
            claim_task(queue_path, task_id, 'alice')
            complete_task(queue_path, task_id, status='failed', duration_s=5)

            server = ThreadingHTTPServer(
                ('127.0.0.1', 0), make_handler(output_root, queue_path, 25.0, None),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_address[1]}'
            try:
                page = urlopen(base + '/').read().decode()
                self.assertIn('重新打开', page)
                self.assertIn('data-task=', page)
                self.assertNotIn('onclick=', page)
                request = Request(
                    base + '/api/reopen', data=json.dumps({
                        'task_id': task_id, 'notes': 'input repaired',
                    }).encode(), method='POST',
                    headers={'Content-Type': 'application/json'},
                )
                reopened = json.loads(urlopen(request).read())
                self.assertEqual(reopened['status'], 'pending')
                self.assertEqual(reopened['attempt_history'][0]['duration_s'], 5)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
