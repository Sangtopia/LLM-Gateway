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


def test_redis_url_is_trimmed():
    assert Settings(redis_url='  "redis://default:pw@host:1"  ').redis_url == "redis://default:pw@host:1"


def test_bad_redis_url_fails_without_echoing_the_secret():
    with pytest.raises(ValueError) as error:
        Settings(redis_url="redis-cli -u redis://default:SECRETPW@host:1")
    assert "must start with redis://" in str(error.value)
    assert "SECRETPW" not in str(error.value)
