import asyncio

import httpx
import pytest

from gateway.retry import Attempt, RetryPolicy, backoff_delay, send_with_retries, should_retry

POLICY = RetryPolicy(max_retries=2, base_delay_s=0.5, max_delay_s=8.0)


def respond(status: int, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json={}, headers=headers)


def run_sequence(outcomes: list, policy: RetryPolicy = POLICY):
    """Feed send_with_retries a scripted list of responses/exceptions; record sleeps."""
    calls, sleeps, attempts = [], [], []

    async def send():
        outcome = outcomes[len(calls)]
        calls.append(outcome)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def sleep(seconds):
        sleeps.append(seconds)

    async def go():
        return await send_with_retries(send, policy, attempts, "claude-haiku-5-5", sleep=sleep)

    try:
        result = asyncio.run(go())
    except Exception as exc:
        result = exc
    return result, calls, sleeps, attempts


@pytest.mark.parametrize("status", [429, 500, 502, 503, 529])
def test_retries_retryable_statuses_then_succeeds(status):
    result, calls, sleeps, attempts = run_sequence([respond(status), respond(status), respond(200)])
    assert result.status_code == 200
    assert len(calls) == 3 and len(sleeps) == 2
    assert [a.status for a in attempts] == [status, status, 200]
    assert attempts[0].retry_delay_ms is not None and attempts[-1].retry_delay_ms is None


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413])
def test_does_not_retry_client_errors(status):
    result, calls, sleeps, _ = run_sequence([respond(status)])
    assert result.status_code == status
    assert len(calls) == 1 and sleeps == []


def test_returns_last_error_response_when_retries_run_out():
    result, calls, _, attempts = run_sequence([respond(529), respond(529), respond(529)])
    assert result.status_code == 529
    assert len(calls) == 3 and len(attempts) == 3


def test_retries_connection_failures_and_reraises_the_last_one():
    errors = [httpx.ConnectError("refused"), httpx.ConnectError("refused"), httpx.ConnectError("refused")]
    result, calls, _, attempts = run_sequence(errors)
    assert isinstance(result, httpx.ConnectError)
    assert len(calls) == 3
    assert [a.error for a in attempts] == ["ConnectError"] * 3


def test_does_not_retry_read_timeouts():
    result, calls, _, attempts = run_sequence([httpx.ReadTimeout("slow")])
    assert isinstance(result, httpx.ReadTimeout)
    assert len(calls) == 1 and attempts[0].error == "ReadTimeout"


def test_x_should_retry_header_overrides_status():
    result, calls, _, _ = run_sequence([respond(529, {"x-should-retry": "false"})])
    assert result.status_code == 529 and len(calls) == 1
    result, calls, _, _ = run_sequence([respond(409, {"x-should-retry": "true"}), respond(200)])
    assert result.status_code == 200 and len(calls) == 2


def test_uses_retry_after_when_within_limit():
    _, _, sleeps, _ = run_sequence([respond(429, {"retry-after": "3"}), respond(200)])
    assert sleeps == [3.0]


def test_ignores_retry_after_longer_than_max_delay():
    # Waiting 60s inside a request is worse than returning the 429; use normal backoff.
    _, _, sleeps, _ = run_sequence([respond(429, {"retry-after": "60"}), respond(200)])
    assert 0 <= sleeps[0] <= POLICY.base_delay_s


def test_backoff_ceiling_doubles_and_is_capped():
    full = lambda: 1.0  # noqa: E731 - take the top of the jitter range
    assert [backoff_delay(n, POLICY, None, rand=full) for n in (1, 2, 3, 4, 5, 6)] == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]
    assert backoff_delay(3, POLICY, None, rand=lambda: 0.25) == 0.5  # Full jitter: anywhere from 0 to the ceiling.


def test_should_retry():
    assert should_retry(respond(529)) and should_retry(respond(500)) and should_retry(respond(429))
    assert not should_retry(respond(400)) and not should_retry(respond(404))


def test_attempt_as_dict_drops_empty_fields():
    assert Attempt(model="m", status=200, ms=1.5).as_dict() == {"model": "m", "status": 200, "ms": 1.5}
