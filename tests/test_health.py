from fastapi.testclient import TestClient

from gateway.circuit import CircuitBreaker
from gateway.main import app


def test_health():
    app.state.breaker = CircuitBreaker()
    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
