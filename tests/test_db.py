import asyncio
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from gateway.db import RequestLogger, requests_table


def record(**overrides) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC),
        "model": "claude-haiku-5-5",
        "provider": "anthropic",
        "stream": False,
        "status_code": 200,
        "outcome": "ok",
        "latency_ms": 12.5,
        "cache_hit": False,
        "fallback_used": False,
        **overrides,
    }


def test_writes_queued_records_and_flushes_on_stop(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        logger = RequestLogger(engine)
        await logger.start()
        for i in range(25):
            logger.write(record(input_tokens=i, cost_usd=0.001))
        await logger.stop()

        async with engine.connect() as conn:
            rows = (await conn.execute(select(requests_table))).mappings().all()
        await engine.dispose()
        return rows

    rows = asyncio.run(run())
    assert len(rows) == 25
    assert sorted(r["input_tokens"] for r in rows) == list(range(25))
    assert rows[0]["model"] == "claude-haiku-5-5"


def test_drops_records_instead_of_blocking_when_queue_is_full(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        logger = RequestLogger(engine, max_queue=3)
        for _ in range(5):  # Not started, so nothing drains the queue.
            logger.write(record())
        await engine.dispose()
        return logger._queue.qsize()

    assert asyncio.run(run()) == 3
