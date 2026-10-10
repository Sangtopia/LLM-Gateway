import asyncio
import time

import fakeredis
import pytest
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

from gateway.keys import ApiKey
from gateway.ratelimit import RateLimiter, tokens_for_rate_limit


def unreachable_redis() -> Redis:
    return Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.1, retry=Retry(NoBackoff(), 0))


def run(coro):
    return asyncio.run(coro)


async def admit_n(limiter: RateLimiter, key: ApiKey, n: int):
    return [await limiter.admit(key) for _ in range(n)]


def test_requests_per_minute_allows_burst_up_to_limit_then_rejects():
    async def go():
        limiter = RateLimiter(fakeredis.FakeAsyncRedis())
        return await admit_n(limiter, ApiKey("key_a", "a", rpm_limit=3), 4)

    decisions = run(go())
    assert [d.allowed for d in decisions] == [True, True, True, False]
    rejected = decisions[-1]
    assert rejected.limit == "requests"
    assert 19 <= rejected.retry_after_s <= 20  # One request refills every 60/3 = 20s.


def test_keys_have_separate_buckets():
    async def go():
        limiter = RateLimiter(fakeredis.FakeAsyncRedis())
        await admit_n(limiter, ApiKey("key_a", "a", rpm_limit=1), 1)
        return await limiter.admit(ApiKey("key_b", "b", rpm_limit=1))

    assert run(go()).allowed


def test_tokens_per_minute_charges_after_the_fact_and_can_go_negative():
    async def go():
        redis = fakeredis.FakeAsyncRedis()
        limiter = RateLimiter(redis)
        key = ApiKey("key_a", "a", tpm_limit=1_000)
        first = await limiter.admit(key)  # Bucket is full, so admitted.
        limiter.charge_tokens(key, 5_000)  # Response used far more than the limit.
        await limiter.drain()
        second = await limiter.admit(key)
        ttl_ms = await redis.pttl("gw:rl:key_a:tpm")
        return first, second, ttl_ms

    first, second, ttl_ms = run(go())
    assert first.allowed
    assert not second.allowed and second.limit == "tokens"
    # Level is about -4000; refilling to 1 at 1000/min takes about 4 minutes.
    assert 239 <= second.retry_after_s <= 241
    # The bucket must outlive its debt, or the overspend would be forgiven.
    assert ttl_ms > 290_000


def test_rejected_request_takes_nothing_from_other_buckets():
    async def go():
        redis = fakeredis.FakeAsyncRedis()
        limiter = RateLimiter(redis)
        key = ApiKey("key_a", "a", rpm_limit=10, tpm_limit=1_000)
        await limiter.admit(key)
        limiter.charge_tokens(key, 5_000)
        await limiter.drain()
        rejected = await limiter.admit(key)
        level = float(await redis.hget("gw:rl:key_a:rpm", "level"))
        return rejected, level

    rejected, rpm_level = run(go())
    assert rejected.limit == "tokens"
    assert rpm_level == pytest.approx(9, abs=0.01)  # Only the first, admitted request was counted.


def test_unlimited_key_never_calls_redis():
    async def go():
        broken = unreachable_redis()
        limiter = RateLimiter(broken)
        decision = await limiter.admit(ApiKey("key_a", "a"))
        limiter.charge_tokens(ApiKey("key_a", "a"), 1_000)
        await limiter.drain()
        await broken.aclose()
        return decision

    assert run(go()).allowed


def test_fails_open_when_redis_is_down_and_stops_trying_for_a_while():
    async def go():
        broken = unreachable_redis()
        limiter = RateLimiter(broken)
        key = ApiKey("key_a", "a", rpm_limit=1, tpm_limit=10)
        first = await limiter.admit(key)  # Pays for the failed connection attempt.
        start = time.perf_counter()
        rest = await admit_n(limiter, key, 50)
        limiter.charge_tokens(key, 1_000)
        await limiter.drain()  # Must not raise.
        elapsed = time.perf_counter() - start
        await broken.aclose()
        return [first, *rest], elapsed

    decisions, elapsed = run(go())
    assert all(d.allowed for d in decisions)
    assert elapsed < 0.05  # Later requests skip Redis instead of each waiting on it.


def test_cache_reads_do_not_count_toward_token_limit():
    usage = {"input_tokens": 10, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 50_000, "output_tokens": 5}
    assert tokens_for_rate_limit(usage) == 115
