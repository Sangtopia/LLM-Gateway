"""Sending a request upstream with timeouts and retries.

Retried:
  - 429, 529 and other 5xx responses
  - connections that fail or drop before any response arrives
Not retried:
  - other 4xx responses (the same request would fail again)
  - read timeouts (the model may still be generating; a retry could bill twice)

Waits use exponential backoff with full jitter (a random wait between zero and
the backoff ceiling), so clients that failed together don't retry together.
Anthropic's retry-after and x-should-retry headers take precedence.
"""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

# Failures where the request never reached the model, so retrying is safe.
RETRYABLE_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0


@dataclass
class Attempt:
    """One try at the upstream, for the request log and trace view."""

    model: str | None
    status: int | None = None
    error: str | None = None
    ms: float = 0.0
    retry_delay_ms: float | None = None  # How long we waited before the next try, if there was one.

    def as_dict(self) -> dict:
        return {k: round(v, 1) if isinstance(v, float) else v for k, v in self.__dict__.items() if v is not None}


def should_retry(response: httpx.Response) -> bool:
    hint = response.headers.get("x-should-retry")
    if hint == "true":
        return True
    if hint == "false":
        return False
    return response.status_code == 429 or response.status_code >= 500


def backoff_delay(retry_number: int, policy: RetryPolicy, response: httpx.Response | None, rand=random.random) -> float:
    """Seconds to wait before retry `retry_number` (1-based)."""
    if response is not None:
        try:
            retry_after = float(response.headers.get("retry-after", ""))
        except ValueError:
            retry_after = None
        if retry_after is not None and 0 <= retry_after <= policy.max_delay_s:
            return retry_after
    ceiling = min(policy.max_delay_s, policy.base_delay_s * 2 ** (retry_number - 1))
    return rand() * ceiling


async def send_with_retries(
    send: Callable[[], Awaitable[httpx.Response]],
    policy: RetryPolicy,
    attempts: list[Attempt],
    model: str | None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> httpx.Response:
    """Call `send` until it succeeds, fails for good, or retries run out.

    Appends one Attempt per try to `attempts`. Returns the last response (which
    may be an error response) or raises the last transport error.
    """
    for retry_number in range(policy.max_retries + 1):
        attempt = Attempt(model=model)
        attempts.append(attempt)
        started = time.perf_counter()
        response = None
        try:
            response = await send()
        except RETRYABLE_ERRORS as exc:
            attempt.error = type(exc).__name__
            if retry_number == policy.max_retries:
                attempt.ms = (time.perf_counter() - started) * 1000
                raise
        except httpx.TransportError as exc:
            attempt.error = type(exc).__name__
            attempt.ms = (time.perf_counter() - started) * 1000
            raise
        attempt.ms = (time.perf_counter() - started) * 1000

        if response is not None:
            attempt.status = response.status_code
            if response.status_code < 400 or not should_retry(response) or retry_number == policy.max_retries:
                return response
            await response.aclose()  # Release the connection before waiting.

        delay = backoff_delay(retry_number + 1, policy, response)
        attempt.retry_delay_ms = round(delay * 1000, 1)
        await sleep(delay)
    raise AssertionError("unreachable")
