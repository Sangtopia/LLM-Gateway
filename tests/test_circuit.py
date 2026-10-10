from gateway.circuit import CircuitBreaker, attempt_succeeded

M = "claude-sonnet-5-5"


class Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def make(threshold=3, cooldown=30.0):
    clock = Clock()
    return CircuitBreaker(failure_threshold=threshold, cooldown_s=cooldown, max_cooldown_s=120.0, clock=clock), clock


def fail_n(breaker, n, model=M):
    for _ in range(n):
        breaker.record(model, False)


def test_opens_after_consecutive_failures():
    breaker, _ = make(threshold=3)
    fail_n(breaker, 2)
    assert breaker.allow(M) and breaker.state(M) == "closed"
    fail_n(breaker, 1)
    assert breaker.state(M) == "open"
    assert not breaker.allow(M)


def test_success_resets_the_count():
    breaker, _ = make(threshold=3)
    fail_n(breaker, 2)
    breaker.record(M, True)
    fail_n(breaker, 2)
    assert breaker.state(M) == "closed"


def test_models_are_independent():
    breaker, _ = make(threshold=1)
    fail_n(breaker, 1)
    assert not breaker.allow(M)
    assert breaker.allow("claude-sonnet-5")


def test_half_open_lets_exactly_one_probe_through():
    breaker, clock = make(threshold=1, cooldown=30)
    fail_n(breaker, 1)
    clock.now += 29
    assert not breaker.allow(M)
    clock.now += 1
    assert breaker.state(M) == "half_open"
    assert breaker.allow(M)  # The probe.
    assert not breaker.allow(M)  # Everyone else waits for its result.


def test_successful_probe_closes_the_circuit():
    breaker, clock = make(threshold=1)
    fail_n(breaker, 1)
    clock.now += 30
    assert breaker.allow(M)
    breaker.record(M, True)
    assert breaker.state(M) == "closed"
    assert breaker.allow(M) and breaker.allow(M)


def test_failed_probe_reopens_with_longer_cooldown_up_to_cap():
    breaker, clock = make(threshold=1, cooldown=30)
    fail_n(breaker, 1)
    for expected_cooldown in (60, 120, 120):
        clock.now += 1_000
        assert breaker.allow(M)  # Probe...
        breaker.record(M, False)  # ...fails.
        assert breaker.state(M) == "open"
        clock.now += expected_cooldown - 1
        assert not breaker.allow(M)
        clock.now += 1
        assert breaker.state(M) == "half_open"
        clock.now -= expected_cooldown  # Rewind; the loop moves time forward again.


def test_abandoned_probe_is_replaced_after_another_cooldown():
    breaker, clock = make(threshold=1, cooldown=30)
    fail_n(breaker, 1)
    clock.now += 30
    assert breaker.allow(M)  # Probe starts but never reports back.
    clock.now += 29
    assert not breaker.allow(M)
    clock.now += 1
    assert breaker.allow(M)  # A new probe.


def test_failures_while_open_without_a_probe_do_not_extend_cooldown():
    breaker, clock = make(threshold=1, cooldown=30)
    fail_n(breaker, 1)
    fail_n(breaker, 5)  # Stragglers that started before the circuit opened.
    clock.now += 30
    assert breaker.state(M) == "half_open"


def test_retry_after_and_snapshot():
    breaker, clock = make(threshold=1, cooldown=30)
    fail_n(breaker, 1, "a")
    clock.now += 10
    fail_n(breaker, 1, "b")
    assert breaker.retry_after_s(["a", "b"]) == 20
    assert breaker.snapshot() == {"a": "open", "b": "open"}
    clock.now += 20
    assert breaker.snapshot() == {"a": "half_open", "b": "open"}


def test_attempt_succeeded():
    assert attempt_succeeded(200) and attempt_succeeded(400) and attempt_succeeded(404)
    assert not attempt_succeeded(429) and not attempt_succeeded(529) and not attempt_succeeded(500)
    assert not attempt_succeeded(None)  # Transport error.
