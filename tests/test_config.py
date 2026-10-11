import pytest

from gateway.config import Settings


@pytest.mark.parametrize("given, expected", [
    ("postgres://u:p@host:5432/db", "postgresql+asyncpg://u:p@host:5432/db"),
    ("postgresql://u:p@host/db", "postgresql+asyncpg://u:p@host/db"),
    ("postgresql+asyncpg://u:p@host/db", "postgresql+asyncpg://u:p@host/db"),
    ("sqlite+aiosqlite:///./gateway.db", "sqlite+aiosqlite:///./gateway.db"),
])
def test_database_url_gets_the_async_driver(given, expected):
    assert Settings(database_url=given).database_url == expected
