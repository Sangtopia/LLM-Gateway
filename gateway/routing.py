"""Model aliases and routing strategies.

Apps can ask for an alias like "fast" or "cheap" instead of a model ID; the
gateway ranks the alias's candidate models with its strategy and sends the
request to the best one. The rest of the ranking becomes the fallback chain, so
the circuit breaker and fallbacks work exactly as for a named model.

Strategies:
  fixed     candidates in the configured order
  cheapest  lowest estimated cost for this request, from the price registry
  fastest   lowest recent p50 time to first token, measured by this process;
            near-ties go to the earlier (cheaper) configured model
"""

import statistics
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from gateway.models import cost_usd

CHARS_PER_TOKEN = 4  # Rough estimate; good enough to rank models by price.
FASTEST_TOLERANCE = 0.15  # p50s within 15% of the fastest count as a tie.


@dataclass(frozen=True)
class Alias:
    strategy: str  # "fixed", "cheapest" or "fastest"
    models: list[str]


# All candidates share the Claude 5 request surface (adaptive thinking, effort),
# so any of them can take a request written for another.
DEFAULT_ALIASES: dict[str, Alias] = {
    "smart": Alias("fixed", ["claude-opus-5-5", "claude-sonnet-5-5"]),
    "balanced": Alias("fixed", ["claude-sonnet-5-5", "claude-sonnet-5", "claude-haiku-5-5"]),
    "cheap": Alias("cheapest", ["claude-haiku-5-5", "claude-sonnet-5", "claude-sonnet-5-5"]),
    "fast": Alias("fastest", ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-sonnet-5"]),
}


@dataclass(frozen=True)
class Route:
    alias: str
    strategy: str
    models: list[str]  # Best first; the rest are fallbacks.


class LatencyTracker:
    """Recent time-to-first-token samples per model, in memory."""

    def __init__(
        self,
        window: int = 50,
        max_age_s: float = 600.0,
        min_samples: int = 5,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.window = window
        self.max_age_s = max_age_s
        self.min_samples = min_samples
        self._clock = clock
        self._samples: dict[str, deque[tuple[float, float]]] = {}

    def record(self, model: str, ttft_ms: float) -> None:
        self._samples.setdefault(model, deque(maxlen=self.window)).append((self._clock(), ttft_ms))

    def p50(self, model: str) -> float | None:
        """Median recent time to first token, or None if there aren't enough fresh samples."""
        cutoff = self._clock() - self.max_age_s
        fresh = [ms for at, ms in self._samples.get(model, ()) if at >= cutoff]
        return statistics.median(fresh) if len(fresh) >= self.min_samples else None

    def snapshot(self) -> dict[str, float]:
        return {m: round(p, 1) for m in self._samples if (p := self.p50(m)) is not None}


def estimate_cost(model: str, body_bytes: int, max_tokens: int | None) -> float:
    usage = {"input_tokens": body_bytes // CHARS_PER_TOKEN, "output_tokens": max_tokens or 1_000}
    cost = cost_usd(model, usage)
    return cost if cost is not None else float("inf")


class Router:
    def __init__(self, aliases: dict[str, Alias], latency: LatencyTracker):
        self.aliases = aliases
        self.latency = latency

    def resolve(self, requested: str | None, request: dict, body_bytes: int) -> Route | None:
        """The ranked models for an alias, or None if `requested` isn't an alias."""
        alias = self.aliases.get(requested or "")
        if alias is None:
            return None
        models = list(alias.models)
        if alias.strategy == "cheapest":
            max_tokens = request.get("max_tokens") if isinstance(request.get("max_tokens"), int) else None
            models.sort(key=lambda m: estimate_cost(m, body_bytes, max_tokens))  # Stable: ties keep config order.
        elif alias.strategy == "fastest":
            # Models without enough recent samples go first (in configured order) so
            # they get measured; otherwise a model ranked last would never get
            # traffic and never get a p50. Then measured models, fastest first.
            p50 = {m: p for m in models if (p := self.latency.p50(m)) is not None}
            unmeasured = [m for m in models if m not in p50]
            # Models within FASTEST_TOLERANCE of the best p50 count as tied and keep
            # configured order (cheapest first), so a model that's a few percent
            # faster by noise doesn't win traffic at many times the price.
            cutoff = min(p50.values(), default=0.0) * (1 + FASTEST_TOLERANCE)
            tied = [m for m in models if m in p50 and p50[m] <= cutoff]
            slower = sorted((m for m in p50 if m not in tied), key=p50.__getitem__)
            models = unmeasured + tied + slower
        return Route(alias=requested, strategy=alias.strategy, models=models)
