"""Measure the latency the gateway adds on top of the upstream.

Starts a fake upstream that answers instantly, starts the gateway pointed at it,
then times the same requests sent directly vs. through the gateway.

    python -m bench.overhead [-n 500]
"""

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

import httpx
from sqlalchemy.ext.asyncio import create_async_engine

from gateway.db import init_db
from gateway.keys import KeyStore

FAKE_PORT = 9100
GATEWAY_PORT = 9101
BODY = {"model": "claude-haiku-5-5", "max_tokens": 20, "messages": [{"role": "user", "content": "Say hi in 3 words"}]}


def start(app: str, port: int, env: dict | None = None) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", app, "--port", str(port), "--log-level", "warning"],
        env={**os.environ, **(env or {})},
    )
    for _ in range(50):
        try:
            httpx.get(f"http://127.0.0.1:{port}/docs", timeout=0.5)
            return proc
        except httpx.TransportError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError(f"{app} did not start on port {port}")


def time_full(client: httpx.Client, url: str) -> float:
    start = time.perf_counter()
    resp = client.post(url, json=BODY)
    resp.raise_for_status()
    return (time.perf_counter() - start) * 1000


def time_to_first_token(client: httpx.Client, url: str) -> float:
    # Reads the whole stream like a real client. Hanging up after the first token
    # would make the gateway drop its upstream connection (by design: it stops
    # generation), so every next request would pay for a fresh connection.
    start = time.perf_counter()
    first = None
    with client.stream("POST", url, json={**BODY, "stream": True}) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if first is None and line.startswith("data: ") and json.loads(line[6:])["type"] == "content_block_delta":
                first = (time.perf_counter() - start) * 1000
    if first is None:
        raise RuntimeError("stream ended without a content_block_delta")
    return first


def percentile(samples: list[float], p: float) -> float:
    return statistics.quantiles(samples, n=100)[int(p) - 1]


async def create_key(database_url: str) -> str:
    engine = create_async_engine(database_url)
    await init_db(engine)
    _, plaintext = await KeyStore(engine).create("benchmark")
    await engine.dispose()
    return plaintext


def run(measure, url: str, n: int, key: str) -> list[float]:
    with httpx.Client(timeout=10, headers={"x-api-key": key}) as client:
        for _ in range(min(50, n)):  # Warm up connections and code paths.
            measure(client, url)
        return [measure(client, url) for _ in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=500, help="requests per scenario")
    args = parser.parse_args()

    # Keep benchmark traffic out of the real request log.
    database_url = f"sqlite+aiosqlite:///{tempfile.mkdtemp()}/bench.db"
    key = asyncio.run(create_key(database_url))

    fake = start("bench.fake_upstream:app", FAKE_PORT)
    gateway = start(
        "gateway.main:app",
        GATEWAY_PORT,
        env={
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{FAKE_PORT}",
            "ANTHROPIC_API_KEY": "fake",
            "DATABASE_URL": database_url,
        },
    )
    direct = f"http://127.0.0.1:{FAKE_PORT}/v1/messages"
    via_gateway = f"http://127.0.0.1:{GATEWAY_PORT}/v1/messages"

    try:
        print(f"{args.n} sequential requests per scenario, fake upstream answers instantly\n")
        print(f"{'scenario':<28}{'direct p50':>12}{'gateway p50':>13}{'added p50':>11}{'added p95':>11}")
        for name, measure in [("non-streaming (total)", time_full), ("streaming (time to 1st token)", time_to_first_token)]:
            d = run(measure, direct, args.n, key)
            g = run(measure, via_gateway, args.n, key)
            d50, g50 = statistics.median(d), statistics.median(g)
            added95 = percentile(g, 95) - percentile(d, 95)
            print(f"{name:<28}{d50:>10.2f}ms{g50:>11.2f}ms{g50 - d50:>9.2f}ms{added95:>9.2f}ms")
    finally:
        gateway.kill()
        fake.kill()


if __name__ == "__main__":
    main()
