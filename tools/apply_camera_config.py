"""Validate a camera YAML file and apply it as one PostgreSQL revision.

Applying the same values again keeps the current revision number. A real change
inserts the next immutable revision and points the camera at it in one transaction.
Editing this file does not affect a pipeline session that has already started.

    python -m tools.apply_camera_config infra/postgres/cameras_counfig/camera-01.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from configs.settings import load_settings
from src.database.camera_config import apply_camera_config, load_camera_config_file
from src.database.client import DatabaseClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", type=Path)
    args = parser.parse_args(argv)
    desired = load_camera_config_file(args.config_file)
    database = DatabaseClient(**load_settings().database.connection_kwargs())
    with database.transaction() as connection:
        result = apply_camera_config(connection, desired)
    state = "created" if result.created else "unchanged"
    print(
        f"camera_id={result.camera_id} config_version={result.config_version} {state}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"camera config apply failed: {exc}", file=sys.stderr)
        sys.exit(1)
