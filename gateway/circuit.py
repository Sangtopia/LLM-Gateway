"""Per-model circuit breakers.

closed     Normal. Consecutive failed attempts are counted; any success resets the count.
open       After `failure_threshold` consecutive failures the model is skipped, so
           requests go straight to the next model in the fallback chain instead of
           each spending seconds on retries against a model that's down.
half-open  After the cooldown, one request is let through as a probe. Success closes
           the circuit; failure opens it again with the cooldown doubled (up to a cap).

State is kept in memory per gateway process, like Envoy's outlier detection or
resilience4j. Sharing it through Redis would add a network round trip to every
request; per-process, each instance notices an outage within a few requests.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _Circuit:
    failures: int = 0
    opened_at: float | None = None
    cooldown_s: float = 0.0
    probe_started_at: float | None = None


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_s: float = 30.0,
        max_cooldown_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self.max_cooldown_s = max_cooldown_s
        self._clock = clock
        self._circuits: dict[str, _Circuit] = {}

    def _get(self, model: str) -> _Circuit:
        return self._circuits.setdefault(model, _Circuit())

    def state(self, model: str) -> str:
        c = self._circuits.get(model)
        if c is None or c.opened_at is None:
            return "closed"
        return "open" if self._clock() - c.opened_at < c.cooldown_s else "half_open"

    def allow(self, model: str) -> bool:
        """Whether a request may go to `model` now. In half-open, only the probe may."""
        c = self._get(model)
        state = self.state(model)
        if state == "closed":
            return True
        if state == "open":
            return False
        now = self._clock()
        # One probe at a time. If a probe never reports back (e.g. its client
        # disconnected), allow another once the cooldown has passed again.
        if c.probe_started_at is not None and now - c.probe_started_at < c.cooldown_s:
            return False
        c.probe_started_at = now
        return True

    def record(self, model: str, success: bool) -> None:
        c = self._get(model)
        if success:
            self._circuits[model] = _Circuit()
            return
        c.failures += 1
        if c.opened_at is not None:
            # A failed probe (or a straggler from before the circuit opened): back off longer.
            if c.probe_started_at is not None:
                c.cooldown_s = min(self.max_cooldown_s, c.cooldown_s * 2)
                c.opened_at = self._clock()
                c.probe_started_at = None
        elif c.failures >= self.failure_threshold:
            c.opened_at = self._clock()
            c.cooldown_s = self.cooldown_s

    def retry_after_s(self, models: list[str]) -> float:
        """Seconds until the first of `models` lets a probe through."""
        now = self._clock()
        waits = [
            max(0.0, c.opened_at + c.cooldown_s - now)
            for m in models
            if (c := self._circuits.get(m)) and c.opened_at is not None
        ]
        return min(waits, default=0.0)

    def snapshot(self) -> dict[str, str]:
        """State of every model that isn't closed, for /health."""
        return {m: s for m in self._circuits if (s := self.state(m)) != "closed"}


def attempt_succeeded(status: int | None) -> bool:
    """Whether an upstream attempt shows the model is healthy.

    A 4xx other than 429 is the request's fault, not the model's, so it counts
    as healthy. 429, 5xx and transport errors (status None) count as failures.
    """
    return status is not None and status < 500 and status != 429
