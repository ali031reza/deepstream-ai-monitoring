import queue
import tempfile
import threading
import unittest

import numpy as np

from src.monitoring.metrics import RunMetrics
from src.pipeline.run_event_pipeline import (
    _base_option, make_event_model, MultiSentinelQueue, make_worker_supervisor,
    pin_verifier_start_offsets,
)
from src.storage.event_frame_store import EventFrameStore
from src.storage.event_policy import EventPolicy


SESSION = "0123456789abcdef0123456789abcdef"


class FakeModel:
    def __init__(self, *args, **kwargs): pass

    def infer(self, frame, *, frame_id):
        return {"model_id": "mock_model_a", "frame_id": frame_id,
                "detections": [{"class_name": "person", "confidence": .93}]}


class PipelineWrapperTest(unittest.TestCase):
    def test_event_model_saves_image(self):
        with tempfile.TemporaryDirectory() as root:
            store = EventFrameStore(root)
            cls = make_event_model(FakeModel, model_key="model_a", camera_id="cam",
                                   session_id=SESSION, store=store,
                                   policy=EventPolicy(.8, 1), metrics=RunMetrics())
            instance = cls()
            output = instance.infer(np.ones((16, 16, 3), dtype=np.uint8), frame_id=2)
            self.assertTrue(output["is_event"])
            self.assertEqual(output["image_size_bytes"], store.stats()["bytes_written"])

    def test_two_model_wrappers_share_image(self):
        with tempfile.TemporaryDirectory() as root:
            store = EventFrameStore(root)
            models = [make_event_model(FakeModel, model_key=f"model_{i}", camera_id="cam",
                                       session_id=SESSION, store=store,
                                       policy=EventPolicy(.8, 1), metrics=RunMetrics())()
                      for i in range(2)]
            image = np.ones((16, 16, 3), dtype=np.uint8)
            a, b = [m.infer(image, frame_id=2) for m in models]
            self.assertEqual(a["storage_key"], b["storage_key"])
            self.assertEqual(store.stats()["saved"], 1)

    def test_non_event_keeps_metadata_empty(self):
        with tempfile.TemporaryDirectory() as root:
            cls = make_event_model(FakeModel, model_key="model_a", camera_id="cam",
                                   session_id=SESSION, store=EventFrameStore(root),
                                   policy=EventPolicy(.98, 1), metrics=RunMetrics())
            out = cls().infer(np.ones((8, 8, 3), dtype=np.uint8), frame_id=2)
            self.assertFalse(out["is_event"])
            self.assertNotIn("storage_key", out)

    def test_multisentinal_queue(self):
        q = MultiSentinelQueue(maxsize=10, workers=3)
        q.put(None)
        self.assertEqual([q.get_nowait() for _ in range(3)], [None]*3)
        for _ in range(3): q.task_done()
        self.assertEqual(q.unfinished_tasks, 0)

    def test_base_option(self):
        self.assertEqual(_base_option(["--source-id", "cam"], "--source-id", "x"), "cam")
        self.assertEqual(_base_option(["--source-id=cam"], "--source-id", "x"), "cam")
        self.assertEqual(_base_option([], "--source-id", "x"), "x")

    def test_verifier_pins_start_to_pre_publish_tail(self):
        class Partition:
            def __init__(self, topic, partition, offset=-1001):
                self.topic, self.partition, self.offset = topic, partition, offset
        class Consumer:
            def get_watermark_offsets(self, part, timeout, cached):
                return (0, 17 if part.topic == "model-a-results" else 9)
            def assign(self, parts):
                self.chosen = parts
        fake = Consumer()
        result = pin_verifier_start_offsets(fake, [Partition("model-a-results", 0),
                                                  Partition("model-b-results", 1)], Partition)
        self.assertEqual([x.offset for x in result], [17, 9])
        self.assertIs(fake.chosen, result)

    def test_worker_supervisor_handles_three_sentinels(self):
        tasks = MultiSentinelQueue(maxsize=10, workers=3)
        received = []
        lock = threading.Lock()
        ready = threading.Event()
        def fake_worker(*args):
            key, q, local_ready = args
            local_ready.set()
            while True:
                x = q.get()
                try:
                    if x is None: return
                    with lock: received.append(x)
                finally:
                    q.task_done()
        # Signature mirrors original worker positional args 0..14.
        # We use a shim to bridge the test's fake worker.
        def original(*args): fake_worker(args[0], args[1], args[13])
        wrapper = make_worker_supervisor(original, 3, 1)
        args = [None]*15
        args[0], args[1], args[13] = "model_a", tasks, ready
        thread = threading.Thread(target=wrapper, args=tuple(args))
        thread.start()
        self.assertTrue(ready.wait(3))
        for i in range(10): tasks.put(i)
        tasks.put(None)
        tasks.join()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(sorted(received), list(range(10)))


if __name__ == "__main__":
    unittest.main()
