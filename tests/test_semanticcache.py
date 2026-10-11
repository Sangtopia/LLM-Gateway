import asyncio

import fakeredis
import numpy as np

from gateway.semanticcache import SemanticCache, semantic_scope, semantic_threshold

H = {"anthropic-version": "2023-06-01"}


def unit(*values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def request(text, **extra):
    return {"model": "claude-haiku-5-5", "max_tokens": 100, "messages": [{"role": "user", "content": text}], **extra}


def test_scope_ignores_only_the_user_text():
    a, text_a = semantic_scope(request("reset my password"), H, "key_a")
    b, text_b = semantic_scope(request("how do I reset my password?"), H, "key_a")
    assert a == b and text_a != text_b
    assert semantic_scope(request("x", max_tokens=200), H, "key_a")[0] != a
    assert semantic_scope(request("x", system="other"), H, "key_a")[0] != a
    assert semantic_scope(request("x"), H, "key_b")[0] != a
    assert semantic_scope(request("x"), {**H, "x-gateway-cache-scope": "user-7"}, "key_a")[0] != a


def test_only_single_turn_text_requests_qualify():
    assert semantic_scope({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}, H, "k") is None
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}}
    assert semantic_scope({"messages": [{"role": "user", "content": [image]}]}, H, "k") is None
    blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert semantic_scope({"messages": [{"role": "user", "content": blocks}]}, H, "k")[1] == "a\nb"
    assert semantic_scope({"messages": [{"role": "user", "content": "   "}]}, H, "k") is None


def test_threshold_header_is_clamped():
    assert semantic_threshold({}) == 0.95
    assert semantic_threshold({"x-gateway-cache-threshold": "0.9"}) == 0.9
    assert semantic_threshold({"x-gateway-cache-threshold": "0.1"}) == 0.80
    assert semantic_threshold({"x-gateway-cache-threshold": "lots"}) == 0.95


def test_lookup_returns_closest_entry_above_threshold():
    async def go():
        cache = SemanticCache(fakeredis.FakeAsyncRedis(), embedder=None)
        cache.store_later("s", unit(1, 0, 0), b'{"a":1}', "application/json", "m", 0.01, 60)
        cache.store_later("s", unit(0, 1, 0), b'{"b":2}', "application/json", "m", 0.02, 60)
        await cache.drain()
        near_a = await cache.lookup("s", unit(1, 0.1, 0), 0.95)
        between = await cache.lookup("s", unit(1, 1, 0), 0.95)
        other_scope = await cache.lookup("t", unit(1, 0, 0), 0.5)
        return near_a, between, other_scope

    near_a, between, other_scope = asyncio.run(go())
    entry, similarity = near_a
    assert entry["body"] == '{"a":1}' and similarity > 0.99
    assert between is None  # 0.71 to both: below the threshold, so a miss.
    assert other_scope is None


def test_scope_list_is_capped():
    from gateway import semanticcache

    async def go():
        redis = fakeredis.FakeAsyncRedis()
        cache = SemanticCache(redis, embedder=None)
        for i in range(semanticcache.MAX_ENTRIES_PER_SCOPE + 5):
            cache.store_later("s", unit(1, i, 0), b"{}", "application/json", "m", 0.0, 60)
            await cache.drain()  # One at a time: 205 at once would exhaust the fake's connection pool.
        return await redis.llen("s")

    assert asyncio.run(go()) == semanticcache.MAX_ENTRIES_PER_SCOPE
