"""PostgreSQL fixtures for the backend contract tests.

The PostgreSQL half of each contract test needs a server, named by
``JIG_TEST_POSTGRES_URL``. Without one those cases skip — unless
``JIG_REQUIRE_POSTGRES_TESTS=1`` (CI sets it), which turns the skip into a
failure so the PostgreSQL backends can never ship on a green run that
silently never exercised them.
"""
from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

POSTGRES_URL_ENV = "JIG_TEST_POSTGRES_URL"
REQUIRE_POSTGRES_ENV = "JIG_REQUIRE_POSTGRES_TESTS"


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    dsn = os.environ.get(POSTGRES_URL_ENV)
    if dsn:
        return dsn
    message = f"{POSTGRES_URL_ENV} is not set; PostgreSQL backend cases cannot run"
    if os.environ.get(REQUIRE_POSTGRES_ENV) == "1":
        pytest.fail(f"{message} and {REQUIRE_POSTGRES_ENV}=1")
    pytest.skip(message)


@pytest.fixture
def pg_schema(pg_dsn: str) -> Iterator[str]:
    """A fresh schema per test, dropped afterwards — tests never share rows."""
    import psycopg

    schema = f"jig_test_{uuid.uuid4().hex[:12]}"
    try:
        yield schema
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
