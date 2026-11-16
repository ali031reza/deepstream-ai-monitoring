from __future__ import annotations

from contextlib import contextmanager

from typing import Iterator

import psycopg


class DatabaseClient:

    def __init__(
        self,
        *,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        connect_timeout: int = 5,
    ):

        if not host:
            raise ValueError("Database host is required")

        if not dbname:
            raise ValueError("Database name is required")

        if not user:
            raise ValueError("Database user is required")

        if not password:
            raise ValueError("Database password is required")

        self._config = {
            "host": host,
            "port": int(port),
            "dbname": dbname,
            "user": user,
            "password": password,
            "connect_timeout": int(connect_timeout),
            "application_name": "deepstream-results-processor",
        }

    @contextmanager
    def transaction(self) -> Iterator:

        # Psycopg automatically commits on successful context exit
        # and rolls back when an exception occurs.

        with psycopg.connect(
            **self._config
        ) as connection:

            yield connection

    def check_connection(self):

        with self.transaction() as connection:

            with connection.cursor() as cursor:

                cursor.execute("SELECT 1")

                return cursor.fetchone()[0] == 1