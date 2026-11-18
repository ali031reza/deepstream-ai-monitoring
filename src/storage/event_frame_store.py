"""Store event JPEGs as immutable S3 objects in the camera's configured prefix."""
from dataclasses import dataclass
import hashlib
from io import BytesIO
import os
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class EventImage:
    storage_key: str
    image_format: str
    image_size_bytes: int
    image_sha256: str


class EventFrameStore:
    def __init__(self, root: str, *, quality: int = 88):
        uri = urlsplit(root)
        self.prefix = uri.path.strip("/")
        if uri.scheme != "s3" or not uri.netloc or not self.prefix or uri.query or uri.fragment:
            raise ValueError(f"Expected s3://bucket/prefix, got {root!r}")
        self.bucket = uri.netloc
        self.root = root.rstrip("/")
        self.quality = quality
        self.client = boto3.client(
            "s3", endpoint_url=os.getenv("OBJECT_STORE_ENDPOINT", "http://object-store:9000"),
            aws_access_key_id="camera-store", aws_secret_access_key=os.environ["OBJECT_STORE_PASSWORD"],
            region_name="us-east-1", config=Config(s3={"addressing_style": "path"}),
        )
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError as exc:
            if exc.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
                raise
            try:
                self.client.create_bucket(Bucket=self.bucket)
            except ClientError as conflict:
                if conflict.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
                    raise
        self._stats = dict(saved=0, reused=0, boxed_saved=0, boxed_reused=0, bytes_written=0)

    @staticmethod
    def boxed_key(clean_key):
        return str(PurePosixPath(clean_key).with_suffix(".boxed.jpg"))

    def put_rgb(self, *, source_session_id, frame_sequence, rgb_array, boxes=None):
        key = f"{self.prefix}/{source_session_id}/{frame_sequence:012d}.jpg"
        image = Image.fromarray(rgb_array)
        counter = ""
        if boxes is not None:
            key = self.boxed_key(key)
            counter = "boxed_"
            draw = ImageDraw.Draw(image)
            for box in boxes:
                draw.rectangle(box, outline=(0, 255, 80), width=max(2, round(min(image.size) / 360)))
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=self.quality, subsampling=0)
        payload = buf.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=payload,
                                   ContentType="image/jpeg", IfNoneMatch="*")
        except ClientError as exc:
            if exc.response["ResponseMetadata"]["HTTPStatusCode"] != 412:
                raise
            self.verify(storage_key=key, image_size_bytes=len(payload), image_sha256=digest)
            self._stats[counter + "reused"] += 1
        else:
            self._stats[counter + "saved"] += 1
            self._stats["bytes_written"] += len(payload)
        return EventImage(key, "jpeg", len(payload), digest)

    def verify(self, *, storage_key, image_size_bytes, image_sha256):
        if not storage_key.startswith(self.prefix + "/"):
            raise ValueError("Object is outside the configured camera prefix")
        response = self.client.get_object(Bucket=self.bucket, Key=storage_key)
        with response["Body"] as body:
            digest = hashlib.sha256(body.read()).hexdigest()
        if response["ContentLength"] != image_size_bytes or digest != image_sha256:
            raise ValueError(f"Stored JPEG integrity mismatch: {storage_key}")

    def stats(self):
        return self._stats.copy()
