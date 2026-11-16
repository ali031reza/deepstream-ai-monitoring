import hashlib
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np

from src.storage.event_frame_store import EventFrameStore
from src.storage.event_policy import EventPolicy


SESSION = "0123456789abcdef0123456789abcdef"


class EventFrameStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = EventFrameStore(self.temp.name)
        self.frame = np.full((16, 24, 3), 180, dtype=np.uint8)

    def put(self):
        return self.store.put_rgb(camera_id="camera", source_session_id=SESSION,
                                  frame_sequence=0, rgb_array=self.frame)

    def test_real_jpeg_and_sha256(self):
        meta = self.put()
        target = self.store.verify(storage_key=meta.storage_key,
                                   image_size_bytes=meta.image_size_bytes,
                                   image_sha256=meta.image_sha256)
        self.assertTrue(target.read_bytes().startswith(b"\xff\xd8"))
        self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), meta.image_sha256)
        self.assertEqual(self.store.stats()["saved"], 1)

    def test_two_models_reuse_one_image(self):
        a = self.put()
        b = self.put()
        self.assertEqual(a, b)
        self.assertEqual(self.store.stats()["saved"], 1)
        self.assertEqual(self.store.stats()["reused"], 1)

    def test_concurrent_model_writes_reuse_image(self):
        outcomes = []
        threads = [threading.Thread(target=lambda: outcomes.append(self.put())) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(len(outcomes), 10)
        self.assertEqual(len(set(outcomes)), 1)
        self.assertEqual(self.store.stats()["saved"], 1)

    def test_conflicting_image_is_not_silently_overwritten(self):
        self.put()
        self.frame[:] = 0
        with self.assertRaisesRegex(ValueError, "collision"):
            self.put()

    def test_integrity_detects_tampering(self):
        meta = self.put()
        self.store.path_for_key(meta.storage_key).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            self.store.verify(storage_key=meta.storage_key,
                              image_size_bytes=meta.image_size_bytes,
                              image_sha256=meta.image_sha256)

    def test_rejects_unsafe_storage_key(self):
        for bad in ("../x.jpg", "/tmp/x.jpg", "a//x.jpg", "a\\b.jpg", "a/x.png"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.store.path_for_key(bad)

    def test_rejects_non_rgb_array(self):
        with self.assertRaises(ValueError):
            self.store.put_rgb(camera_id="camera", source_session_id=SESSION,
                               frame_sequence=0, rgb_array=np.zeros((4, 4), dtype=np.uint8))

    def test_key_is_deterministic_and_camera_specific(self):
        k = self.store.key("camera", SESSION, 15)
        self.assertEqual(k, self.store.key("camera", SESSION, 15))
        self.assertNotEqual(k, self.store.key("other-camera", SESSION, 15))


class EventPolicyTest(unittest.TestCase):
    def test_event_requires_threshold(self):
        policy = EventPolicy(.7, 1)
        self.assertTrue(policy.matches({"detections": [{"class_name": "person", "confidence": .8}]}, 1))
        self.assertFalse(policy.matches({"detections": [{"class_name": "person", "confidence": .6}]}, 1))

    def test_event_every_n_frames(self):
        policy = EventPolicy(.1, 5)
        detection = {"detections": [{"class_name": "person", "confidence": .7}]}
        self.assertTrue(policy.matches(detection, 5))
        self.assertFalse(policy.matches(detection, 4))

    def test_event_class_filter(self):
        policy = EventPolicy(.5, 1, "vehicle")
        self.assertFalse(policy.matches({"detections": [{"class_name": "person", "confidence": .99}]}, 1))

    def test_rejects_invalid_policy(self):
        with self.assertRaises(ValueError): EventPolicy(2, 1)
        with self.assertRaises(ValueError): EventPolicy(.5, 0)


if __name__ == "__main__":
    unittest.main()
