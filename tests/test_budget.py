import asyncio
from datetime import UTC, datetime, timedelta

import fakeredis
from sqlalchemy.ext.asyncio import create_async_engine

from gateway.db import init_db, spend_since
from gateway.keys import ApiKey
from gateway.ratelimit import RateLimiter, month_start, next_month_start

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
KEY = ApiKey("key_a", "a", monthly_budget_usd=1.00)


class Clock:
    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class SpendLog:
    """Stands in for the request log: returns a fixed spend and records calls."""

    def __init__(self, spent: float = 0.0):
        self.spent = spent
        self.calls: list[tuple[str, datetime]] = []

    async def __call__(self, key_id: str, since: datetime) -> float:
        self.calls.append((key_id, since))
        return self.spent


def run(coro):
    return asyncio.run(coro)


def test_month_boundaries():
    assert month_start(NOW) == datetime(2026, 10, 1, tzinfo=UTC)
    assert next_month_start(NOW) == datetime(2026, 11, 1, tzinfo=UTC)
    assert next_month_start(datetime(2026, 12, 31, 23, 59, tzinfo=UTC)) == datetime(2027, 1, 1, tzinfo=UTC)


def test_allows_until_spend_reaches_budget_then_rejects_until_next_month():
    async def go():
        limiter = RateLimiter(fakeredis.FakeAsyncRedis(), load_spend=SpendLog(), clock=Clock(NOW))
        before = await limiter.admit(KEY)
        limiter.record_usage(KEY, 0, 0.60)
        await limiter.drain()
        middle = await limiter.admit(KEY)
        limiter.record_usage(KEY, 0, 0.45)  # Now $1.05: soft cap, the in-flight request still counted.
        await limiter.drain()
        after = await limiter.admit(KEY)
        return before, middle, after

    before, middle, after = run(go())
    assert before.allowed and middle.allowed
    assert not after.allowed
    assert after.limit == "monthly budget"
    assert after.retry_after_s == int((datetime(2026, 11, 1, tzinfo=UTC) - NOW).total_seconds())
    assert "$1.00" in after.message and "$1.05" in after.message and "2026-11-01" in after.message


def test_missing_counter_is_rebuilt_from_request_log_once():
    async def go():
        spend_log = SpendLog(spent=1.20)
        limiter = RateLimiter(fakeredis.FakeAsyncRedis(), load_spend=spend_log, clock=Clock(NOW))
        first = await limiter.admit(KEY)
        second = await limiter.admit(KEY)
        return first, second, spend_log.calls

    first, second, calls = run(go())
    assert not first.allowed and not second.allowed  # Redis was empty, but the log shows $1.20 spent.
    assert calls == [("key_a", datetime(2026, 10, 1, tzinfo=UTC))]  # Second check used the rebuilt counter.


def test_charge_before_rebuild_does_not_create_a_counter_from_zero():
    async def go():
        redis = fakeredis.FakeAsyncRedis()
        limiter = RateLimiter(redis, load_spend=SpendLog(spent=0.90), clock=Clock(NOW))
        limiter.record_usage(KEY, 0, 0.05)  # No counter yet: skipped, the request log has it.
        await limiter.drain()
        exists_after_charge = await redis.exists("gw:spend:key_a:2026-10")
        decision = await limiter.admit(KEY)  # Rebuilds from the log ($0.90).
        stored = int(await redis.get("gw:spend:key_a:2026-10"))
        return exists_after_charge, decision, stored

    exists_after_charge, decision, stored = run(go())
    assert exists_after_charge == 0
    assert decision.allowed
    assert stored == 900_000  # Micro-dollars.


def test_new_month_starts_a_new_counter():
    async def go():
        clock = Clock(NOW)
        spend_log = SpendLog()
        limiter = RateLimiter(fakeredis.FakeAsyncRedis(), load_spend=spend_log, clock=clock)
        await limiter.admit(KEY)
        limiter.record_usage(KEY, 0, 2.00)
        await limiter.drain()
        october = await limiter.admit(KEY)
        clock.now = datetime(2026, 11, 1, 0, 0, 1, tzinfo=UTC)
        november = await limiter.admit(KEY)
        return october, november, spend_log.calls[-1]

    october, november, last_load = run(go())
    assert not october.allowed
    assert november.allowed
    assert last_load == ("key_a", datetime(2026, 11, 1, tzinfo=UTC))


def test_budget_and_rate_limit_checked_in_one_call():
    async def go():
        limiter = RateLimiter(fakeredis.FakeAsyncRedis(), load_spend=SpendLog(), clock=Clock(NOW))
        key = ApiKey("key_a", "a", rpm_limit=2, monthly_budget_usd=1.00)
        return [await limiter.admit(key) for _ in range(3)]

    decisions = run(go())
    assert [d.allowed for d in decisions] == [True, True, False]
    assert decisions[-1].limit == "requests"


def test_spend_since_sums_this_months_costs_for_one_key(tmp_path):
    from gateway.db import requests_table
    from sqlalchemy import insert

    def row(key_id, cost, created_at):
        return {
            "id": f"{key_id}-{cost}-{created_at:%d}",
            "created_at": created_at,
            "api_key_id": key_id,
            "provider": "anthropic",
            "stream": False,
            "status_code": 200,
            "outcome": "ok",
            "latency_ms": 1.0,
            "cost_usd": cost,
        }

    async def go():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'spend.db'}")
        await init_db(engine)
        async with engine.begin() as conn:
            await conn.execute(
                insert(requests_table),
                [
                    row("key_a", 0.25, NOW),
                    row("key_a", 0.50, NOW - timedelta(days=1)),
                    row("key_a", 9.00, datetime(2026, 9, 30, tzinfo=UTC)),  # Last month.
                    row("key_b", 7.00, NOW),  # Another key.
                    row("key_a", None, NOW),  # Failed request with no cost.
                ],
            )
        total = await spend_since(engine, "key_a", month_start(NOW))
        none = await spend_since(engine, "key_zzz", month_start(NOW))
        await engine.dispose()
        return total, none

    total, none = run(go())
    assert total == 0.75
    assert none == 0.0
