import asyncio
import json

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from gateway.db import batches_table, init_db
from gateway.main import app
from tests.test_proxy import OTHER_KEY, TEST_KEY, make_client

USAGE = {"input_tokens": 1_000, "output_tokens": 100, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}


def fake_batches_api():
    """Anthropic's batch endpoints, just enough to exercise the gateway."""
    calls = []
    batches = {}

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path))
        path = req.url.path
        if req.method == "POST" and path == "/v1/messages/batches":
            batch_id = f"msgbatch_{len(batches) + 1}"
            batches[batch_id] = {"id": batch_id, "type": "message_batch", "processing_status": "ended",
                                 "results_url": f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results"}
            return httpx.Response(200, json=batches[batch_id])
        if req.method == "GET" and path == "/v1/messages/batches":
            return httpx.Response(200, json={"data": list(batches.values()), "has_more": False})
        batch_id = path.split("/")[4]
        if path.endswith("/results"):
            lines = [
                {"custom_id": "a", "result": {"type": "succeeded", "message": {"model": "claude-sonnet-5-5", "usage": USAGE}}},
                {"custom_id": "b", "result": {"type": "succeeded", "message": {"model": "claude-sonnet-5-5", "usage": USAGE}}},
                {"custom_id": "c", "result": {"type": "errored", "error": {"type": "invalid_request_error"}}},
            ]
            return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode() + b"\n")
        if path.endswith("/cancel"):
            return httpx.Response(200, json={**batches[batch_id], "processing_status": "canceling"})
        if req.method == "DELETE":
            return httpx.Response(200, json={"id": batch_id, "type": "message_batch_deleted"})
        return httpx.Response(200, json=batches[batch_id])

    return handler, calls


@pytest.fixture
def batch_client(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'batches.db'}")
    asyncio.run(init_db(engine))
    handler, calls = fake_batches_api()
    client, log = make_client(handler)
    app.state.engine = engine
    yield client, log, calls, engine
    asyncio.run(engine.dispose())


BODY = {"requests": [{"custom_id": c, "params": {"model": "claude-sonnet-5-5", "max_tokens": 10,
                                                   "messages": [{"role": "user", "content": "hi"}]}} for c in "abc"]}


def test_create_records_owner_and_points_results_url_at_the_gateway(batch_client):
    client, log, _, engine = batch_client
    batch = client.post("/v1/messages/batches", json=BODY).json()

    assert batch["results_url"] == f"http://testserver/v1/messages/batches/{batch['id']}/results"

    async def owner():
        async with engine.connect() as conn:
            return (await conn.execute(select(batches_table))).mappings().one()

    row = asyncio.run(owner())
    assert row["api_key_id"] == "key_test" and row["request_count"] == 3 and row["charged"] is False
    assert log.records[0]["endpoint"] == "batch_create"


def test_other_keys_cannot_see_or_touch_the_batch(batch_client):
    client, _, calls, _ = batch_client
    batch_id = client.post("/v1/messages/batches", json=BODY).json()["id"]
    other = {"x-api-key": OTHER_KEY}

    for method, path in [("GET", f"/v1/messages/batches/{batch_id}"), ("GET", f"/v1/messages/batches/{batch_id}/results"),
                         ("POST", f"/v1/messages/batches/{batch_id}/cancel"), ("DELETE", f"/v1/messages/batches/{batch_id}")]:
        assert client.request(method, path, headers=other).status_code == 404
    assert client.get("/v1/messages/batches", headers=other).json()["data"] == []
    assert len(calls) == 2  # Only the create, and the list (filtered afterwards); nothing else reached Anthropic.


def test_results_stream_through_and_are_charged_once_at_the_batch_price(batch_client):
    client, log, _, engine = batch_client
    batch_id = client.post("/v1/messages/batches", json=BODY).json()["id"]

    first = client.get(f"/v1/messages/batches/{batch_id}/results")
    second = client.get(f"/v1/messages/batches/{batch_id}/results")

    assert first.text == second.text and len(first.text.strip().splitlines()) == 3
    full_price = 2 * (1_000 * 2.00 + 100 * 10.00) / 1e6  # Two succeeded Sonnet 5.5 requests.
    assert app.state.limiter.charged == [("key_test", 0, pytest.approx(full_price / 2))]  # Once, at 50% off.
    charge = [r for r in log.records if r["endpoint"] == "batch_results"]
    assert len(charge) == 1 and charge[0]["cost_usd"] == pytest.approx(full_price / 2)
    assert charge[0]["input_tokens"] == 2_000 and charge[0]["output_tokens"] == 200

    async def stored():
        async with engine.connect() as conn:
            return (await conn.execute(select(batches_table))).mappings().one()

    row = asyncio.run(stored())
    assert row["charged"] is True and row["succeeded"] == 2


def test_retrieve_cancel_and_delete_for_the_owner(batch_client):
    client, _, _, _ = batch_client
    batch_id = client.post("/v1/messages/batches", json=BODY).json()["id"]
    got = client.get(f"/v1/messages/batches/{batch_id}").json()
    assert got["results_url"].startswith("http://testserver/")
    assert client.post(f"/v1/messages/batches/{batch_id}/cancel").json()["processing_status"] == "canceling"
    assert client.delete(f"/v1/messages/batches/{batch_id}").json()["type"] == "message_batch_deleted"


def test_batches_need_a_gateway_key(batch_client):
    client, _, calls, _ = batch_client
    assert client.post("/v1/messages/batches", json=BODY, headers={"x-api-key": "gw-nope"}).status_code == 401
    assert client.get("/v1/messages/batches", headers={"x-api-key": ""}).status_code == 401
    assert calls == []


def test_the_anthropic_sdk_works_unchanged(batch_client):
    """The SDK follows results_url; it has to land back on the gateway."""
    import anthropic

    client, _, _, _ = batch_client
    sdk = anthropic.Anthropic(base_url="http://testserver", api_key=TEST_KEY, http_client=client, max_retries=0)
    batch = sdk.messages.batches.create(requests=BODY["requests"])
    results = list(sdk.messages.batches.results(batch.id))
    assert [r.custom_id for r in results] == ["a", "b", "c"]
    assert results[0].result.type == "succeeded"
