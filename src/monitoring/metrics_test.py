import tempfile
import json
import threading
import unittest
from pathlib import Path
from src.monitoring.metrics import RunMetrics, distribution, save_json


class MetricsTest(unittest.TestCase):
    def test_percentile_calculation(self):
        self.assertEqual(distribution([10, 2, 4, 6])["p95"], 10)
        self.assertIsNone(distribution([])["p99"])

    def test_thread_safe_counters(self):
        m = RunMetrics(max_samples=10)
        def work():
            for _ in range(1000): m.add("work", duration_ms=1)
        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        snap = m.snapshot()
        self.assertEqual(snap["counts"]["work"], 4000)
        self.assertEqual(snap["latency_ms"]["work"]["count"], 10)

    def test_json_report(self):
        with tempfile.TemporaryDirectory() as root:
            filename = Path(root) / "reports" / "test.json"
            save_json(filename, {"valid": True})
            self.assertEqual(json.loads(filename.read_text()), {"valid": True})


if __name__ == "__main__": unittest.main()
