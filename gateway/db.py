"""Database tables (API keys, request log) and the background request log writer."""

import asyncio
import logging
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    func,
    insert,
    inspect,
    select,
    text,
)
from sqlalchemy.ext.asyncio import AsyncEngine

log = logging.getLogger(__name__)

metadata = MetaData()

requests_table = Table(
    "requests",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("endpoint", String),  # None for /v1/messages; "count_tokens" for the token counter.
    Column("api_key_id", String, index=True),  # None when the request had no valid key.
    Column("model", String, index=True),  # As requested by the app.
    Column("served_model", String, index=True),  # The model that answered (differs after a fallback).
    Column("route", String),  # "alias:strategy" when the app asked for an alias like "fast".
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
    Column("prompt_cache_savings_usd", Float),  # Saved by Anthropic prompt caching (negative: unread writes).
    Column("prompt_cache_added", Boolean),  # The gateway added cache_control to this request.
    Column("response_cache_saved_usd", Float),  # On a response cache hit: what the original request cost.
    Column("cache_kind", String),  # "exact" or "semantic" on a response cache hit.
    Column("cache_similarity", Float),  # Cosine similarity of a semantic hit.
    Column("latency_ms", Float, nullable=False),  # Request arrival to last byte sent.
    Column("ttft_ms", Float),  # Request arrival to first content delta (streams only).
    Column("upstream_ms", Float),  # Time waiting for upstream response headers.
    Column("cache_hit", Boolean, nullable=False, default=False),
    Column("fallback_used", Boolean, nullable=False, default=False),
    Column("refusal_fallback", Boolean),  # Served by another model after the first one declined.
    Column("upstream_request_id", String),
    Column("attempts", JSON),  # One entry per upstream try: model, status or error, timing, retry wait.
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
        await conn.run_sync(_add_missing_columns)


def _add_missing_columns(sync_conn) -> None:
    """Add columns introduced after a table was first created.

    create_all only creates missing tables. This covers the simple case of new
    nullable columns without a migration tool; anything more involved (renames,
    type changes) would need Alembic.
    """
    inspector = inspect(sync_conn)
    for table in metadata.sorted_tables:
        existing = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name not in existing and column.nullable:
                col_type = column.type.compile(dialect=sync_conn.dialect)
                sync_conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{column.name}" {col_type}'))
                log.info("added column %s.%s", table.name, column.name)


async def spend_since(engine: AsyncEngine, key_id: str, since: datetime) -> float:
    """Dollars a key has spent since `since`, from the request log."""
    t = requests_table
    query = select(func.coalesce(func.sum(t.c.cost_usd), 0.0)).where(t.c.api_key_id == key_id, t.c.created_at >= since)
    async with engine.connect() as conn:
        return float((await conn.execute(query)).scalar_one())


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
