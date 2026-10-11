import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import create_async_engine

from gateway.config import settings
from gateway.db import api_keys_table, init_db, requests_table
from gateway.keys import KeyStore
from gateway.main import app

NOW = datetime.now(UTC)


def row(**overrides) -> dict:
    return {
        "id": str(uuid.uuid4()), "created_at": NOW - timedelta(minutes=5), "api_key_id": "key_a",
        "model": "claude-sonnet-5-5", "served_model": "claude-sonnet-5-5", "provider": "anthropic",
        "stream": False, "status_code": 200, "outcome": "ok", "latency_ms": 1_000.0, "cost_usd": 0.01,
        "cache_hit": False, "fallback_used": False, "attempts": [{"model": "claude-sonnet-5-5", "status": 200}],
        # A multi-row insert takes its columns from the first row, so every row lists them all.
        "error_type": None, "response_cache_saved_usd": None,
        **overrides,
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "admin_token", "")  # Local-only mode, whatever the developer's .env says.
    rows = [
        row(),
        row(latency_ms=3_000.0, served_model="claude-sonnet-5", fallback_used=True, cost_usd=0.02,
            attempts=[{"model": "claude-sonnet-5-5", "status": 529}, {"model": "claude-sonnet-5", "status": 200}]),
        row(outcome="error", status_code=529, error_type="overloaded_error", cost_usd=None, served_model=None),
        row(cache_hit=True, cost_usd=0.0, response_cache_saved_usd=0.01, attempts=[]),
        row(outcome="error", status_code=429, error_type="gateway_rate_limited", cost_usd=None, api_key_id="key_b"),
        row(created_at=NOW - timedelta(days=3), cost_usd=5.0),  # Outside 24h.
    ]

    async def setup():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'admin.db'}")
        await init_db(engine)
        async with engine.begin() as conn:
            await conn.execute(insert(requests_table), rows)
            await conn.execute(insert(api_keys_table).values(
                id="key_a", name="helpdesk", key_hash="h", key_prefix="gw-", created_at=NOW))
        return engine

    engine = asyncio.run(setup())
    app.state.engine = engine
    app.state.keys = KeyStore(engine)
    yield TestClient(app), rows
    asyncio.run(engine.dispose())


def test_stats_aggregates_the_window(client):
    c, _ = client
    stats = c.get("/admin/stats?hours=24").json()
    k = stats["kpis"]
    assert k["requests"] == 5 and k["rejected"] == 1
    assert k["spend_usd"] == pytest.approx(0.03)
    assert k["error_rate"] == pytest.approx(1 / 4)  # The gateway's own 429 isn't an error.
    assert k["fallback_rate"] == pytest.approx(1 / 3)  # Cache hits never reach upstream.
    assert k["cache_hit_rate"] == pytest.approx(1 / 4)
    assert k["response_cache_savings_usd"] == pytest.approx(0.01)
    assert stats["spend_by_day"]["models"] == ["claude-sonnet-5", "claude-sonnet-5-5"]
    assert stats["spend_by_key"][0] == {"key_id": "key_a", "name": "helpdesk", "spend_usd": pytest.approx(0.03)}
    assert stats["series"]["granularity"] == "hour"


def test_longer_window_includes_older_rows_by_day(client):
    c, _ = client
    stats = c.get("/admin/stats?hours=168").json()
    assert stats["kpis"]["requests"] == 6
    assert stats["series"]["granularity"] == "day"


def test_recent_requests_and_trace(client):
    c, rows = client
    recent = c.get("/admin/requests?limit=10").json()
    assert len(recent) == 6
    assert {r["tries"] for r in recent} == {0, 1, 2}
    trace = c.get(f"/admin/requests/{rows[1]['id']}").json()
    assert [a["status"] for a in trace["attempts"]] == [529, 200]
    assert c.get("/admin/requests/nope").status_code == 404


def test_admin_token_is_enforced_when_set(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(settings, "admin_token", "s3cret")
    assert c.get("/admin/stats").status_code == 401
    assert c.get("/admin/stats", headers={"x-admin-token": "wrong"}).status_code == 401
    assert c.get("/admin/stats", headers={"x-admin-token": "s3cret"}).status_code == 200


def test_dashboard_page_is_served(client):
    c, _ = client
    page = c.get("/dashboard")
    assert page.status_code == 200 and "Gateway Dashboard" in page.text


def test_create_list_and_revoke_keys_over_the_api(client):
    c, _ = client
    created = c.post("/admin/keys", json={"name": "helpdesk", "rpm": 60, "monthly_budget_usd": 20}).json()
    assert created["key"].startswith("gw-")
    listed = {k["id"]: k for k in c.get("/admin/keys").json()}
    assert listed[created["id"]]["rpm_limit"] == 60 and "key_hash" not in listed[created["id"]]
    assert asyncio.run(app.state.keys.authenticate(created["key"])) is not None
    assert c.delete(f"/admin/keys/{created['id']}").json() == {"revoked": created["id"]}
    assert c.delete(f"/admin/keys/{created['id']}").status_code == 404
    assert c.post("/admin/keys", json={"name": ""}).status_code == 422


def test_key_api_needs_the_admin_token_when_set(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(settings, "admin_token", "s3cret")
    assert c.post("/admin/keys", json={"name": "x"}).status_code == 401
    assert c.post("/admin/keys", json={"name": "x"}, headers={"x-admin-token": "s3cret"}).status_code == 201
