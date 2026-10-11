import asyncio

import fakeredis
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

from gateway.responsecache import MAX_ENTRY_BYTES, ResponseCache, cache_ttl


def test_ttl_header_is_clamped():
    assert cache_ttl({}) == 3_600
    assert cache_ttl({"x-gateway-cache-ttl": "60"}) == 60
    assert cache_ttl({"x-gateway-cache-ttl": "999999"}) == 86_400
    assert cache_ttl({"x-gateway-cache-ttl": "0"}) == 1
    assert cache_ttl({"x-gateway-cache-ttl": "soon"}) == 3_600


def test_entries_expire_and_oversized_bodies_are_skipped():
    async def go():
        redis = fakeredis.FakeAsyncRedis()
        cache = ResponseCache(redis)
        stored = await cache.put("k", b"{}", "application/json", "m", 0.01, 60)
        ttl = await redis.ttl("k")
        too_big = await cache.put("big", b"x" * (MAX_ENTRY_BYTES + 1), "application/json", "m", 0.01, 60)
        return stored, ttl, too_big, await cache.get("k"), await cache.get("big")

    stored, ttl, too_big, entry, big = asyncio.run(go())
    assert stored and 0 < ttl <= 60
    assert entry == {"body": "{}", "content_type": "application/json", "model": "m", "cost_usd": 0.01}
    assert not too_big and big is None


def test_redis_down_means_no_cache_not_an_error():
    async def go():
        broken = Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.1, retry=Retry(NoBackoff(), 0))
        cache = ResponseCache(broken)
        result = await cache.get("k"), await cache.put("k", b"{}", "application/json", "m", 0.0, 60)
        await broken.aclose()
        return result

    assert asyncio.run(go()) == (None, False)
