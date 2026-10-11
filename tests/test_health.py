from fastapi.testclient import TestClient

from gateway.circuit import CircuitBreaker
from gateway.main import app
from gateway.routing import DEFAULT_ALIASES, LatencyTracker, Router
from gateway.upstreamlimits import UpstreamLimits


def test_health():
    app.state.breaker = CircuitBreaker()
    app.state.router = Router(DEFAULT_ALIASES, LatencyTracker())
    app.state.upstream_limits = UpstreamLimits()
    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
