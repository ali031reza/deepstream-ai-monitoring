import unittest
from src.benchmark.kafka_load import check_database


class FakeCursor:
    def __init__(self, rows): self.rows = rows
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def execute(self, sql, params):
        assert params == ("test-session",)
        assert "source_session_id" in sql
    def fetchall(self): return self.rows

class FakeDB:
    class Context:
        def __init__(self, obj): self.obj = obj
        def __enter__(self): return self.obj
        def __exit__(self, *args): pass
    class Conn:
        def __init__(self, rows): self.rows = rows
        def cursor(self): return FakeDB.Context(FakeCursor(self.rows))
    def __init__(self, rows): self.rows = rows
    def transaction(self): return FakeDB.Context(FakeDB.Conn(self.rows))

class LoadVerifyTest(unittest.TestCase):
    def test_both_models_persisted_all_frames(self):
        rows = [("mock_model_a", 20, 20), ("mock_model_b", 20, 20)]
        result = check_database(FakeDB(rows), "test-session", 20, 1)
        self.assertEqual(result["mock_model_b"]["distinct_frames"], 20)

    def test_detects_too_many_rows(self):
        rows = [("mock_model_a", 21, 21), ("mock_model_b", 20, 20)]
        with self.assertRaises(AssertionError):
            check_database(FakeDB(rows), "test-session", 20, 1)

if __name__ == "__main__": unittest.main()
