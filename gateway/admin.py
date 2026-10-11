"""Read-only admin API behind the dashboard: aggregates and traces from the request log.

Protected by ADMIN_TOKEN (sent as `x-admin-token`). With no token configured, the
admin API only answers requests from this machine.
"""

import hmac
import statistics
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from gateway.config import settings
from gateway.db import api_keys_table, requests_table

router = APIRouter()

DASHBOARD_HTML = Path(__file__).parent / "static" / "dashboard.html"

# Requests the gateway itself turned away. They count as traffic but not as failures:
# the error rate is about whether requests that should have worked did.
GATEWAY_REJECTIONS = {"authentication_error", "gateway_rate_limited", "gateway_budget_exceeded"}
MAX_ROWS = 50_000


def require_admin(request: Request) -> None:
    token = settings.admin_token
    if token:
        if not hmac.compare_digest(request.headers.get("x-admin-token", ""), token):
            raise HTTPException(401, "Admin token required.")
    elif request.client is None or request.client.host not in ("127.0.0.1", "::1", "testclient"):
        raise HTTPException(403, "Set ADMIN_TOKEN to use the admin API from another machine.")


def _engine(request: Request) -> AsyncEngine:
    return request.app.state.engine


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(DASHBOARD_HTML.read_text(encoding="utf-8"))


@router.get("/admin/stats")
async def stats(request: Request, hours: int = 24) -> dict:
    require_admin(request)
    hours = max(1, min(hours, 24 * 90))
    now = datetime.now(UTC)
    since = now - timedelta(hours=hours)
    t = requests_table
    query = select(t).where(t.c.created_at >= since).order_by(t.c.created_at).limit(MAX_ROWS)
    async with _engine(request).connect() as conn:
        rows = [dict(r) for r in (await conn.execute(query)).mappings()]
        key_names = {r.id: r.name for r in await conn.execute(select(api_keys_table.c.id, api_keys_table.c.name))}

    hourly = hours <= 48
    def bucket(at: datetime) -> str:
        at = _as_utc(at)
        return at.strftime("%Y-%m-%dT%H:00Z") if hourly else at.strftime("%Y-%m-%d")

    served = [r for r in rows if r["error_type"] not in GATEWAY_REJECTIONS]
    upstream = [r for r in served if not r["cache_hit"]]
    latencies = [r["latency_ms"] for r in served if r["outcome"] == "ok"]

    spend_by_day: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    spend_by_key: dict[str, float] = defaultdict(float)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["cost_usd"]:
            spend_by_day[_as_utc(r["created_at"]).strftime("%Y-%m-%d")][r["served_model"] or r["model"] or "unknown"] += r["cost_usd"]
            spend_by_key[r["api_key_id"] or "none"] += r["cost_usd"]
        by_bucket[bucket(r["created_at"])].append(r)

    days = sorted(spend_by_day)
    models = sorted({m for day in spend_by_day.values() for m in day})
    # Every bucket in the window, including empty ones, so the time axis is evenly spaced.
    step = timedelta(hours=1) if hourly else timedelta(days=1)
    buckets, at = [], since
    while at <= now:
        buckets.append(bucket(at))
        at += step
    buckets = sorted(set(buckets) | set(by_bucket))

    def bucket_series(fn) -> list:
        return [fn(by_bucket[b]) for b in buckets]

    def ok_latencies(group: list[dict]) -> list[float]:
        return [r["latency_ms"] for r in group if r["outcome"] == "ok" and r["error_type"] not in GATEWAY_REJECTIONS]

    def error_rate(group: list[dict]) -> float | None:
        counted = [r for r in group if r["error_type"] not in GATEWAY_REJECTIONS]
        return _rate(sum(r["outcome"] == "error" for r in counted), len(counted))

    def fallback_rate(group: list[dict]) -> float | None:
        counted = [r for r in group if r["error_type"] not in GATEWAY_REJECTIONS and not r["cache_hit"]]
        return _rate(sum(bool(r["fallback_used"]) for r in counted), len(counted))

    def cache_hit_rate(group: list[dict]) -> float | None:
        counted = [r for r in group if r["error_type"] not in GATEWAY_REJECTIONS]
        return _rate(sum(bool(r["cache_hit"]) for r in counted), len(counted))

    names = [key_names.get(k, k) for k in spend_by_key]
    def label(key_id: str) -> str:
        name = key_names.get(key_id, key_id)
        return f"{name} ({key_id})" if names.count(name) > 1 else name

    return {
        "hours": hours,
        "key_names": {k: label(k) for k in key_names},
        "generated_at": now.isoformat(timespec="seconds"),
        "truncated": len(rows) == MAX_ROWS,
        "kpis": {
            "requests": len(rows),
            "rejected": len(rows) - len(served),
            "spend_usd": sum(r["cost_usd"] or 0 for r in rows),
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            "error_rate": error_rate(rows),
            "fallback_rate": _rate(sum(bool(r["fallback_used"]) for r in upstream), len(upstream)),
            "cache_hit_rate": cache_hit_rate(rows),
            "prompt_cache_savings_usd": sum(r["prompt_cache_savings_usd"] or 0 for r in rows),
            "response_cache_savings_usd": sum(r["response_cache_saved_usd"] or 0 for r in rows),
        },
        "spend_by_day": {
            "days": days,
            "models": models,
            "values": {m: [round(spend_by_day[d].get(m, 0.0), 6) for d in days] for m in models},
        },
        "spend_by_key": sorted(
            ({"key_id": k, "name": label(k), "spend_usd": round(v, 6)} for k, v in spend_by_key.items()),
            key=lambda x: -x["spend_usd"],
        ),
        "series": {
            "buckets": buckets,
            "granularity": "hour" if hourly else "day",
            "requests": bucket_series(len),
            "p50_latency_ms": bucket_series(lambda g: _percentile(ok_latencies(g), 50)),
            "p95_latency_ms": bucket_series(lambda g: _percentile(ok_latencies(g), 95)),
            "error_rate": bucket_series(error_rate),
            "fallback_rate": bucket_series(fallback_rate),
            "cache_hit_rate": bucket_series(cache_hit_rate),
        },
    }


SUMMARY_COLUMNS = [
    "id", "created_at", "api_key_id", "model", "served_model", "route", "stream", "status_code", "outcome",
    "error_type", "input_tokens", "output_tokens", "cost_usd", "latency_ms", "ttft_ms", "cache_hit",
    "fallback_used", "attempts",
]


@router.get("/admin/requests")
async def recent_requests(request: Request, limit: int = 100) -> list[dict]:
    require_admin(request)
    t = requests_table
    query = select(*[t.c[c] for c in SUMMARY_COLUMNS]).order_by(t.c.created_at.desc()).limit(max(1, min(limit, 500)))
    async with _engine(request).connect() as conn:
        rows = [dict(r) for r in (await conn.execute(query)).mappings()]
    for r in rows:
        r["created_at"] = _as_utc(r["created_at"]).isoformat(timespec="seconds")
        r["tries"] = len(r.pop("attempts") or [])
    return rows


@router.get("/admin/requests/{request_id}")
async def request_trace(request: Request, request_id: str) -> dict:
    """Everything logged for one request, including each upstream attempt."""
    require_admin(request)
    async with _engine(request).connect() as conn:
        row = (await conn.execute(select(requests_table).where(requests_table.c.id == request_id))).mappings().first()
    if row is None:
        raise HTTPException(404, "No such request.")
    row = dict(row)
    row["created_at"] = _as_utc(row["created_at"]).isoformat(timespec="seconds")
    return row
