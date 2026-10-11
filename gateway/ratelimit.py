"""Per-key limits in Redis: requests and tokens per minute, and a monthly dollar budget.

Rate limits
-----------
Each limit is a bucket holding up to `limit` units that refills continuously at
`limit / 60` units per second, so traffic is smoothed rather than reset at the
top of each minute.

Token counts are only known once a response finishes, so the token limit is
enforced in two halves: a request is admitted while the bucket is above zero,
and its actual tokens are subtracted afterwards, which may push the bucket
negative. A key that overspends then waits until the bucket refills.

All bucket checks for a request run in one Lua script: atomic, so two gateway
processes can't both take the last unit.

Monthly budgets
---------------
Spend is a Redis counter per key per calendar month (UTC), in micro-dollars so
float rounding never accumulates. It's checked in the same round trip as the
rate limits. If the counter is missing (first request of the month, or Redis
lost its data) it's rebuilt from the request log, so a Redis restart can't wipe
a key's spend.

The budget is a soft cap: requests already in flight when it runs out still
complete, so a key can end slightly over. A hard cap would mean reserving each
request's worst-case cost up front.
"""

import asyncio
import hashlib
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from redis.asyncio import Redis
from redis.exceptions import NoScriptError, RedisError

from gateway.keys import ApiKey
from gateway.models import format_usd

log = logging.getLogger(__name__)

WINDOW_MS = 60_000
MICRODOLLARS = 1_000_000
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

# Adds to a spend counter only if it exists. A missing counter is rebuilt from
# the request log on the next check; creating it here from zero would hide the
# month's earlier spend.
SPEND_CHARGE_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 1 then
    return redis.call('INCRBY', KEYS[1], ARGV[1])
end
return nil
"""

# (key id, start of month) -> dollars spent since then, from the request log.
SpendLoader = Callable[[str, datetime], Awaitable[float]]

SCRIPTS = {hashlib.sha1(script.encode()).hexdigest(): script for script in (BUCKET_SCRIPT, SPEND_CHARGE_SCRIPT)}
BUCKET_SHA, SPEND_CHARGE_SHA = SCRIPTS


@dataclass(frozen=True)
class Decision:
    allowed: bool
    retry_after_s: int = 0
    limit: str | None = None  # Which limit was hit: "requests", "tokens" or "monthly budget".
    message: str | None = None
    remaining_budget_usd: float | None = None  # Set when the key has a budget and the request is allowed.


ALLOW = Decision(True)
ALWAYS_PASS = -1e18


def _bucket_key(key_id: str, kind: str) -> str:
    return f"gw:rl:{key_id}:{kind}"


def month_start(now: datetime) -> datetime:
    return now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def next_month_start(now: datetime) -> datetime:
    start = month_start(now)
    return start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)


def _spend_key(key_id: str, now: datetime) -> str:
    return f"gw:spend:{key_id}:{month_start(now):%Y-%m}"


class RateLimiter:
    def __init__(
        self,
        redis: Redis,
        load_spend: SpendLoader | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self._redis = redis
        self._load_spend = load_spend
        self._now = clock or (lambda: datetime.now(UTC))
        self._background: set[asyncio.Task] = set()
        # After a Redis failure, skip Redis for a few seconds so that only one
        # request pays for the timeout, not every request while it's down.
        self._redis_down_until = 0.0

    def _redis_down(self) -> bool:
        return time.monotonic() < self._redis_down_until

    def _mark_redis_down(self) -> None:
        self._redis_down_until = time.monotonic() + REDIS_RETRY_AFTER_FAILURE_S

    @staticmethod
    def _bucket_call(buckets: list[tuple[str, int, float, int]]) -> tuple:
        args: list = [WINDOW_MS]
        for _, capacity, min_required, cost in buckets:
            args += [capacity, min_required, cost]
        return ("EVALSHA", BUCKET_SHA, len(buckets), *[b[0] for b in buckets], *args)

    async def _execute(self, calls: list[tuple]) -> list:
        """Run commands in one round trip.

        Scripts are sent by hash (EVALSHA) and only uploaded when Redis doesn't
        have them (first use, or after a Redis restart). redis-py's own script
        pipelining checks for them on every call, which costs an extra round trip.
        """
        pipe = self._redis.pipeline(transaction=False)
        for call in calls:
            pipe.execute_command(*call)
        results = await pipe.execute(raise_on_error=False)
        for i, result in enumerate(results):
            if isinstance(result, NoScriptError):
                for script in SCRIPTS.values():
                    await self._redis.script_load(script)
                results[i] = await self._redis.execute_command(*calls[i])
            elif isinstance(result, Exception):
                raise result
        return results

    async def admit(self, key: ApiKey) -> Decision:
        """Check the key's limits before calling upstream, taking one request from its rate limit."""
        buckets, names = [], []
        if key.rpm_limit is not None:
            buckets.append((_bucket_key(key.id, "rpm"), key.rpm_limit, 1, 1))
            names.append("requests")
        if key.tpm_limit is not None:
            buckets.append((_bucket_key(key.id, "tpm"), key.tpm_limit, 1, 0))
            names.append("tokens")
        has_budget = key.monthly_budget_usd is not None
        if not buckets and not has_budget:
            return ALLOW  # Unlimited keys never touch Redis.
        if self._redis_down():
            return ALLOW

        now = self._now()
        try:
            # One round trip for every check.
            calls = []
            if has_budget:
                calls.append(("GET", _spend_key(key.id, now)))
            if buckets:
                calls.append(self._bucket_call(buckets))
            results = await self._execute(calls)

            if has_budget:
                spent = results[0]
                spent = int(spent) if spent is not None else await self._rebuild_spend(key.id, now)
                if spent >= key.monthly_budget_usd * MICRODOLLARS:
                    reset = next_month_start(now)
                    return Decision(
                        False,
                        retry_after_s=max(1, math.ceil((reset - now).total_seconds())),
                        limit="monthly budget",
                        message=(
                            f"Gateway key {key.id} reached its monthly budget of {format_usd(key.monthly_budget_usd)} "
                            f"(spent {format_usd(spent / MICRODOLLARS)}). It resets on {reset:%Y-%m-%d} UTC."
                        ),
                    )
        except RedisError:
            # Fail open: a Redis outage shouldn't take the gateway down with it.
            self._mark_redis_down()
            log.warning("limiter unavailable, allowing requests for %.0fs", REDIS_RETRY_AFTER_FAILURE_S, exc_info=True)
            return ALLOW

        if buckets:
            allowed, retry_ms, failed = (int(v) for v in results[-1])
            if not allowed:
                limit = names[failed - 1]
                return Decision(
                    False,
                    retry_after_s=max(1, math.ceil(retry_ms / 1000)),
                    limit=limit,
                    message=f"Gateway key {key.id} exceeded its {limit} per minute limit.",
                )
        if has_budget:
            return Decision(True, remaining_budget_usd=key.monthly_budget_usd - spent / MICRODOLLARS)
        return ALLOW

    async def _rebuild_spend(self, key_id: str, now: datetime) -> int:
        """Reload this month's spend from the request log into Redis, and return it."""
        if self._load_spend is None:
            return 0
        spent = round(await self._load_spend(key_id, month_start(now)) * MICRODOLLARS)
        ttl_s = math.ceil((next_month_start(now) - now).total_seconds()) + 7 * 86_400
        # NX: if another gateway process rebuilt it first (and charges have landed since), keep theirs.
        if not await self._redis.set(_spend_key(key_id, now), spent, nx=True, ex=ttl_s):
            spent = int(await self._redis.get(_spend_key(key_id, now)) or spent)
        return spent

    def record_usage(self, key: ApiKey, tokens: int, cost_usd: float | None) -> None:
        """Charge a finished request's tokens and cost to the key. Runs in the background."""
        charge_tokens = key.tpm_limit is not None and tokens > 0
        charge_cost = key.monthly_budget_usd is not None and bool(cost_usd)
        if not (charge_tokens or charge_cost) or self._redis_down():
            return
        task = asyncio.create_task(
            self._record(key, tokens if charge_tokens else 0, cost_usd if charge_cost else 0.0)
        )
        self._background.add(task)  # Keep a reference so the task isn't garbage collected.
        task.add_done_callback(self._background.discard)

    async def _record(self, key: ApiKey, tokens: int, cost_usd: float) -> None:
        try:
            calls = []
            if tokens:
                calls.append(self._bucket_call([(_bucket_key(key.id, "tpm"), key.tpm_limit, ALWAYS_PASS, tokens)]))
            if cost_usd:
                micro = round(cost_usd * MICRODOLLARS)
                calls.append(("EVALSHA", SPEND_CHARGE_SHA, 1, _spend_key(key.id, self._now()), micro))
            await self._execute(calls)
        except RedisError:
            self._mark_redis_down()
            log.warning("could not record usage for %s", key.id, exc_info=True)

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
