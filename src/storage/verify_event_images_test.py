import tempfile
import unittest
import numpy as np
from src.storage.event_frame_store import EventFrameStore
from src.storage.verify_event_images import verify_session

SESSION = "0123456789abcdef0123456789abcdef"

class Cursor:
    def __init__(self, models, image_rows):
        self.models = models; self.image_rows = image_rows; self.calls = 0
    def execute(self, sql, params):
        assert params == (SESSION,)
        self.calls += 1
    def fetchall(self):
        return self.models if self.calls == 1 else self.image_rows
    def __enter__(self): return self
    def __exit__(self, *args): pass

class Connection:
    def __init__(self, cursor): self.c = cursor
    def cursor(self): return self.c

class DB:
    def __init__(self, cursor): self.c = cursor
    class Transaction:
        def __init__(self, conn): self.conn = conn
        def __enter__(self): return self.conn
        def __exit__(self, *args): pass
    def transaction(self): return DB.Transaction(Connection(self.c))

class VerifyTest(unittest.TestCase):
    def test_verified_image_and_model_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EventFrameStore(tmp)
            meta = store.put_rgb(camera_id="camera", source_session_id=SESSION,
                                 frame_sequence=1, rgb_array=np.ones((8,8,3), dtype=np.uint8))
            cursor = Cursor([("mock_model_a", 1, 1, 0), ("mock_model_b", 1, 1, 0)],
                            [(meta.storage_key, meta.image_size_bytes, meta.image_sha256,
                              meta.image_format, "camera", f"{SESSION}:000000000001")])
            report = verify_session(DB(cursor), store, SESSION)
            self.assertEqual(report["verified_unique_images"], 1)
            self.assertEqual(report["models"]["mock_model_b"]["rows"], 1)

    def test_missing_foreign_key_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            cursor = Cursor([("mock_model_a", 1, 1, 1)], [])
            with self.assertRaisesRegex(AssertionError, "without image ID"):
                verify_session(DB(cursor), EventFrameStore(tmp), SESSION)

if __name__ == "__main__": unittest.main()
