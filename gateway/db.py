"""Database tables (API keys, request log) and the background request log writer."""

import asyncio
import logging

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, MetaData, String, Table, insert
from sqlalchemy.ext.asyncio import AsyncEngine

log = logging.getLogger(__name__)

metadata = MetaData()

requests_table = Table(
    "requests",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("api_key_id", String, index=True),  # None when the request had no valid key.
    Column("model", String, index=True),
    Column("provider", String, nullable=False),
    Column("stream", Boolean, nullable=False),
    Column("status_code", Integer, nullable=False),
    Column("outcome", String, nullable=False),  # "ok", "error" or "incomplete"
    Column("error_type", String),
    Column("stop_reason", String),
    Column("input_tokens", Integer),
    Column("output_tokens", Integer),
    Column("cache_read_tokens", Integer),
    Column("cache_write_tokens", Integer),
    Column("cost_usd", Float),
    Column("latency_ms", Float, nullable=False),  # Request arrival to last byte sent.
    Column("ttft_ms", Float),  # Request arrival to first content delta (streams only).
    Column("upstream_ms", Float),  # Time waiting for upstream response headers.
    Column("cache_hit", Boolean, nullable=False, default=False),
    Column("fallback_used", Boolean, nullable=False, default=False),
    Column("upstream_request_id", String),
)

api_keys_table = Table(
    "api_keys",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("key_hash", String(64), nullable=False, unique=True),  # SHA-256 hex; the key itself is never stored.
    Column("key_prefix", String, nullable=False),  # First characters, to tell keys apart in listings.
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("revoked_at", DateTime(timezone=True)),
    # Limits; None means unlimited.
    Column("rpm_limit", Integer),
    Column("tpm_limit", Integer),
    Column("monthly_budget_usd", Float),
)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)


class RequestLogger:
    """Writes request records in the background so logging never delays a response.

    Records are queued in memory and inserted in batches. If the queue fills up
    (the database is down or too slow), new records are dropped with a warning
    rather than slowing down or failing live traffic.
    """

    def __init__(self, engine: AsyncEngine, max_queue: int = 10_000, max_batch: int = 500):
        self._engine = engine
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=max_queue)
        self._max_batch = max_batch
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Flush everything queued so far, then stop."""
        if self._task is None:
            return
        await self._queue.put(None)
        await self._task
        self._task = None

    def write(self, record: dict) -> None:
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            log.warning("request log queue full, dropping record %s", record.get("id"))

    async def _run(self) -> None:
        stopping = False
        while not stopping:
            batch = [await self._queue.get()]
            while not self._queue.empty() and len(batch) < self._max_batch:
                batch.append(self._queue.get_nowait())
            if None in batch:
                stopping = True
                batch = [r for r in batch if r is not None]
            if not batch:
                continue
            try:
                async with self._engine.begin() as conn:
                    await conn.execute(insert(requests_table), batch)
            except Exception:
                log.exception("failed to write %d request log records", len(batch))
