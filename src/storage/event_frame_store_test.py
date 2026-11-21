"""Partial event uploads stay unpublished until the clean and boxed pair is durable."""
import unittest

import numpy as np
from botocore.exceptions import ClientError

from src.storage.event_frame_store import EventFrameStore


class _Body:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._payload


class _FakeClient:
    def __init__(self):
        self.objects = {}
        self.fail_keys = set()

    def put_object(self, *, Bucket, Key, Body, ContentType, IfNoneMatch):
        if Key in self.fail_keys:
            raise ClientError(
                {"Error": {"Code": "InternalError", "Message": "upload failed"},
                 "ResponseMetadata": {"HTTPStatusCode": 500}},
                "PutObject",
            )
        if Key in self.objects:
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed", "Message": "exists"},
                 "ResponseMetadata": {"HTTPStatusCode": 412}},
                "PutObject",
            )
        self.objects[Key] = bytes(Body)

    def get_object(self, *, Bucket, Key):
        payload = self.objects[Key]
        return {"Body": _Body(payload), "ContentLength": len(payload)}


class EventFrameStorePairTest(unittest.TestCase):
    def test_failed_boxed_upload_retries_without_publishing_a_partial_pair(self):
        client = _FakeClient()
        store = EventFrameStore("s3://camera-events/events/camera-01", client=client)
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[8:40, 10:50] = (180, 40, 40)
        boxes = ((10, 8, 50, 40),)
        boxed_key = "events/camera-01/session/000000000007.boxed.jpg"
        client.fail_keys.add(boxed_key)

        published = False
        with self.assertRaises(ClientError):
            store.put_pair(
                source_session_id="session", frame_sequence=7,
                rgb_array=frame, boxes=boxes,
            )
            published = True

        self.assertFalse(published)
        self.assertIn("events/camera-01/session/000000000007.jpg", client.objects)
        self.assertNotIn(boxed_key, client.objects)
        self.assertEqual(store.stats()["saved"], 1)
        self.assertEqual(store.stats()["boxed_saved"], 0)

        client.fail_keys.clear()
        clean, boxed = store.put_pair(
            source_session_id="session", frame_sequence=7,
            rgb_array=frame, boxes=boxes,
        )
        published = True

        self.assertTrue(published)
        self.assertEqual(clean.storage_key, "events/camera-01/session/000000000007.jpg")
        self.assertEqual(boxed.storage_key, boxed_key)
        self.assertEqual(client.objects[clean.storage_key], _jpeg_bytes(clean, client))
        store.verify(
            storage_key=clean.storage_key,
            image_size_bytes=clean.image_size_bytes,
            image_sha256=clean.image_sha256,
        )
        store.verify(
            storage_key=boxed.storage_key,
            image_size_bytes=boxed.image_size_bytes,
            image_sha256=boxed.image_sha256,
        )
        self.assertEqual(store.stats()["reused"], 1)
        self.assertEqual(store.stats()["boxed_saved"], 1)
        self.assertEqual(store.stats()["pair_latency"]["count"], 1)

    def test_two_models_share_the_clean_jpeg_and_keep_separate_boxes(self):
        client = _FakeClient()
        store = EventFrameStore("s3://camera-events/events/camera-01", client=client)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        boxes = ((1, 1, 8, 8),)
        first_clean, first_boxed = store.put_pair(
            source_session_id="session", frame_sequence=7, rgb_array=frame,
            boxes=boxes, model_id="yolo11n-person",
        )
        second_clean, second_boxed = store.put_pair(
            source_session_id="session", frame_sequence=7, rgb_array=frame,
            boxes=((2, 2, 9, 9),), model_id="yolo11n-fire",
        )
        self.assertEqual(first_clean.storage_key, second_clean.storage_key)
        self.assertEqual(first_clean.storage_key, "events/camera-01/session/000000000007.jpg")
        self.assertEqual(
            first_boxed.storage_key,
            "events/camera-01/session/000000000007.yolo11n-person.boxed.jpg",
        )
        self.assertEqual(
            second_boxed.storage_key,
            "events/camera-01/session/000000000007.yolo11n-fire.boxed.jpg",
        )
        self.assertNotEqual(first_boxed.storage_key, second_boxed.storage_key)
        self.assertEqual(store.stats()["saved"], 1)
        self.assertEqual(store.stats()["reused"], 1)
        self.assertEqual(store.stats()["boxed_saved"], 2)


def _jpeg_bytes(image, client) -> bytes:
    return client.objects[image.storage_key]


if __name__ == "__main__":
    unittest.main()
