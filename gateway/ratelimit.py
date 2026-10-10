"""Per-key rate limits: requests per minute and tokens per minute, as Redis token buckets.

Each limit is a bucket holding up to `limit` units that refills continuously at
`limit / 60` units per second, so traffic is smoothed rather than reset at the
top of each minute.

Token counts are only known once a response finishes, so the token limit is
enforced in two halves: a request is admitted while the bucket is above zero,
and its actual tokens are subtracted afterwards, which may push the bucket
negative. A key that overspends then waits until the bucket refills.

All checks for a request run in one Lua script: one network round trip, and
atomic, so two gateway processes can't both take the last unit.
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

from gateway.keys import ApiKey

log = logging.getLogger(__name__)

WINDOW_MS = 60_000
REDIS_RETRY_AFTER_FAILURE_S = 5.0

# KEYS: one Redis key per bucket.
# ARGV: per bucket, (capacity, min_required, cost).
#   Admit a request:  min_required = 1, cost = 1 (requests) or 0 (tokens).
#   Charge tokens:    min_required = -1e18 (always passes), cost = tokens used.
# Returns {1, 0, 0} if every bucket passed (and costs were taken), or
# {0, retry_after_ms, index of the first failing bucket} (and nothing was taken).
# Uses the Redis server clock, so gateway processes with skewed clocks agree.
BUCKET_SCRIPT = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local window = tonumber(ARGV[1])
local levels = {}
local retry_ms = 0
local failed = 0

for i, key in ipairs(KEYS) do
    local base = 1 + (i - 1) * 3
    local capacity = tonumber(ARGV[base + 1])
    local min_required = tonumber(ARGV[base + 2])
    local rate = capacity / window
    local state = redis.call('HMGET', key, 'level', 'ts')
    local level = tonumber(state[1]) or capacity
    local ts = tonumber(state[2]) or now
    level = math.min(capacity, level + math.max(0, now - ts) * rate)
    levels[i] = level
    if level < min_required then
        retry_ms = math.max(retry_ms, math.ceil((min_required - level) / rate))
        if failed == 0 then failed = i end
    end
end

if failed > 0 then
    return {0, retry_ms, failed}
end

for i, key in ipairs(KEYS) do
    local base = 1 + (i - 1) * 3
    local capacity = tonumber(ARGV[base + 1])
    local level = levels[i] - tonumber(ARGV[base + 3])
    redis.call('HSET', key, 'level', tostring(level), 'ts', now)
    -- Keep the bucket until it would be full again; expiring sooner would
    -- forgive an overdrawn key's debt.
    local refill_ms = math.ceil((capacity - level) / (capacity / window))
    redis.call('PEXPIRE', key, math.max(1000, refill_ms + 1000))
end
return {1, 0, 0}
"""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    retry_after_s: int = 0
    limit: str | None = None  # Which limit was hit: "requests" or "tokens".


ALLOW = Decision(True)
ALWAYS_PASS = -1e18


def _bucket_key(key_id: str, kind: str) -> str:
    return f"gw:rl:{key_id}:{kind}"


class RateLimiter:
    def __init__(self, redis: Redis):
        self._redis = redis
        self._script = redis.register_script(BUCKET_SCRIPT)
        self._background: set[asyncio.Task] = set()
        # After a Redis failure, skip Redis for a few seconds so that only one
        # request pays for the timeout, not every request while it's down.
        self._redis_down_until = 0.0

    def _redis_down(self) -> bool:
        return time.monotonic() < self._redis_down_until

    def _mark_redis_down(self) -> None:
        self._redis_down_until = time.monotonic() + REDIS_RETRY_AFTER_FAILURE_S

    async def _run(self, buckets: list[tuple[str, int, float, int]]) -> tuple[bool, int, int]:
        """Returns (allowed, retry_after_ms, 1-based index of the first failing bucket)."""
        keys = [b[0] for b in buckets]
        args: list = [WINDOW_MS]
        for _, capacity, min_required, cost in buckets:
            args += [capacity, min_required, cost]
        allowed, retry_ms, failed = await self._script(keys=keys, args=args)
        return bool(allowed), int(retry_ms), int(failed)

    async def admit(self, key: ApiKey) -> Decision:
        """Check (and take one request from) the key's limits before calling upstream."""
        buckets, names = [], []
        if key.rpm_limit is not None:
            buckets.append((_bucket_key(key.id, "rpm"), key.rpm_limit, 1, 1))
            names.append("requests")
        if key.tpm_limit is not None:
            buckets.append((_bucket_key(key.id, "tpm"), key.tpm_limit, 1, 0))
            names.append("tokens")
        if not buckets:
            return ALLOW  # Unlimited keys never touch Redis.
        if self._redis_down():
            return ALLOW

        try:
            allowed, retry_ms, failed = await self._run(buckets)
        except RedisError:
            # Fail open: a Redis outage shouldn't take the gateway down with it.
            self._mark_redis_down()
            log.warning("rate limiter unavailable, allowing requests for %.0fs", REDIS_RETRY_AFTER_FAILURE_S, exc_info=True)
            return ALLOW
        if allowed:
            return ALLOW
        return Decision(False, retry_after_s=max(1, math.ceil(retry_ms / 1000)), limit=names[failed - 1])

    def charge_tokens(self, key: ApiKey, tokens: int) -> None:
        """Subtract tokens used by a finished request. Runs in the background."""
        if key.tpm_limit is None or tokens <= 0 or self._redis_down():
            return
        task = asyncio.create_task(self._charge(key, tokens))
        self._background.add(task)  # Keep a reference so the task isn't garbage collected.
        task.add_done_callback(self._background.discard)

    async def _charge(self, key: ApiKey, tokens: int) -> None:
        try:
            await self._run([(_bucket_key(key.id, "tpm"), key.tpm_limit, ALWAYS_PASS, tokens)])
        except RedisError:
            self._mark_redis_down()
            log.warning("could not charge %d tokens to %s", tokens, key.id, exc_info=True)

    async def drain(self) -> None:
        """Wait for background charges to finish (on shutdown and in tests)."""
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)


def tokens_for_rate_limit(usage: dict) -> int:
    """Tokens that count toward the per-minute limit.

    Mirrors Anthropic's own limits: cache reads don't count, so caching a long
    prompt lets a key do more work within the same limit.
    """
    return sum(
        usage.get(field) or 0 for field in ("input_tokens", "cache_creation_input_tokens", "output_tokens")
    )
