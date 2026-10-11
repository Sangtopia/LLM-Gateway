"""Exact-match response cache, in Redis.

Claude 5 models don't accept sampling parameters, so no request is guaranteed to
produce the same answer twice and the gateway can't decide on its own that a
response is safe to reuse. Apps opt in per request with `x-gateway-cache: on`
when a previous answer to the identical request is acceptable (optionally
`x-gateway-cache-ttl: <seconds>`). `x-gateway-cache: semantic` also matches
requests that mean the same thing (see semanticcache.py).

`x-gateway-cache-scope` narrows sharing further, e.g. to one end user: the gateway
can't tell which parts of a prompt are personal, and identical text from two
different users often needs two different answers.

The key is a hash of the request body (JSON with sorted keys, so field order
doesn't matter), the anthropic-version and anthropic-beta headers, and the
gateway key: one key's cached answers are never served to another.
Only complete, successful responses are stored. Streams are stored as their raw
event bytes and replayed as a stream. Redis errors are ignored: a cache that's
down just means every request goes upstream.
"""

import asyncio
import hashlib
import json
import logging
import time

from redis.asyncio import Redis
from redis.exceptions import RedisError

log = logging.getLogger(__name__)

DEFAULT_TTL_S = 3_600
MAX_TTL_S = 86_400
MAX_ENTRY_BYTES = 256 * 1024


def cache_mode(headers) -> str | None:
    """None (no caching), "exact", or "semantic" (exact first, then by meaning)."""
    value = headers.get("x-gateway-cache", "").lower()
    return {"on": "exact", "exact": "exact", "semantic": "semantic"}.get(value)


def cache_ttl(headers) -> int:
    try:
        ttl = int(headers.get("x-gateway-cache-ttl", DEFAULT_TTL_S))
    except ValueError:
        return DEFAULT_TTL_S
    return max(1, min(ttl, MAX_TTL_S))


def cache_key(request: dict, headers, key_id: str) -> str:
    material = json.dumps(
        {
            "body": request,
            "version": headers.get("anthropic-version", ""),
            "beta": headers.get("anthropic-beta", ""),
            "key": key_id,
            "scope": headers.get("x-gateway-cache-scope", ""),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "gw:cache:" + hashlib.sha256(material.encode()).hexdigest()


class ResponseCache:
    def __init__(self, redis: Redis):
        self._redis = redis
        self._down_until = 0.0
        self._background: set[asyncio.Task] = set()

    def put_later(self, *args) -> None:
        """Store in the background, so the write never delays the response."""
        task = asyncio.create_task(self.put(*args))
        self._background.add(task)  # Keep a reference so the task isn't garbage collected.
        task.add_done_callback(self._background.discard)

    async def drain(self) -> None:
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)

    async def get(self, key: str) -> dict | None:
        if time.monotonic() < self._down_until:
            return None
        try:
            raw = await self._redis.get(key)
        except RedisError:
            self._down_until = time.monotonic() + 5.0
            log.warning("response cache unavailable", exc_info=True)
            return None
        return json.loads(raw) if raw else None

    async def put(self, key: str, body: bytes, content_type: str, model: str | None, cost_usd: float | None, ttl_s: int) -> bool:
        if len(body) > MAX_ENTRY_BYTES or time.monotonic() < self._down_until:
            return False
        entry = {"body": body.decode("utf-8"), "content_type": content_type, "model": model, "cost_usd": cost_usd}
        try:
            await self._redis.set(key, json.dumps(entry), ex=ttl_s)
        except RedisError:
            self._down_until = time.monotonic() + 5.0
            log.warning("could not store response in cache", exc_info=True)
            return False
        return True
