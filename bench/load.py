"""Load test: throughput and the latency the gateway adds, at several concurrency levels.

    python -m bench.load [--concurrency 1,10,50,100] [--duration 15] [--upstream-delay-ms 300] [--stream]

Starts the fake upstream (answering after --upstream-delay-ms, to stand in for model
latency) and a gateway pointed at it, then for each concurrency level runs that many
clients in a closed loop for --duration seconds: once straight at the fake upstream and
once through the gateway. The difference is what the gateway adds under load.

The gateway key has no limits, so the test never touches Redis: this measures the
gateway itself, not a remote Redis's round trip or its free-tier ops limit.

The load generator uses aiohttp: a single httpx client stalls above ~50 concurrent
connections (bench/client_check.py), which would measure the client, not the gateway.
Run high levels on Linux, e.g. in Docker; Windows loopback stalls on its own around 50-100.
"""

import argparse
import asyncio
import json
import statistics
import tempfile
import time

import aiohttp
import httpx

from bench.overhead import FAKE_PORT, GATEWAY_PORT, create_key, start, stop

BODY = {"model": "claude-haiku-5-5", "max_tokens": 20, "messages": [{"role": "user", "content": "Say hi in 3 words"}]}


async def one(session: aiohttp.ClientSession, url: str, stream: bool) -> float:
    started = time.perf_counter()
    async with session.post(url, json={**BODY, "stream": True} if stream else BODY) as resp:
        resp.raise_for_status()
        async for _ in resp.content.iter_any():
            pass
    return (time.perf_counter() - started) * 1000


async def run_level(url: str, key: str, concurrency: int, duration: float, stream: bool) -> dict:
    latencies: list[float] = []
    errors = 0
    deadline = time.perf_counter() + duration
    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(headers={"x-api-key": key}, connector=connector, timeout=timeout) as client:
        await asyncio.gather(*(one(client, url, stream) for _ in range(min(concurrency, 20))))  # Warm up.

        async def worker():
            nonlocal errors
            while time.perf_counter() < deadline:
                try:
                    latencies.append(await one(client, url, stream))
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                    errors += 1

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        elapsed = time.perf_counter() - started
    latencies.sort()
    q = statistics.quantiles(latencies, n=100) if len(latencies) > 1 else latencies * 99
    return {"rps": len(latencies) / elapsed, "p50": q[49], "p95": q[94], "p99": q[98],
            "errors": errors, "requests": len(latencies) + errors}


async def main_async(args) -> None:
    db_url = f"sqlite+aiosqlite:///{tempfile.mkdtemp()}/load.db"
    key = await create_key(db_url)
    # One process: uvicorn's multi-worker mode doesn't work reliably on Windows (workers fail to
    # listen), and each worker would keep its own _control settings.
    fake = start("bench.fake_upstream:app", FAKE_PORT)
    gateway = start("gateway.main:app", GATEWAY_PORT, env={
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{FAKE_PORT}", "ANTHROPIC_API_KEY": "fake", "DATABASE_URL": db_url,
    })
    try:
        httpx.post(f"http://127.0.0.1:{FAKE_PORT}/_control", json={"delay_ms": args.upstream_delay_ms})
        direct = f"http://127.0.0.1:{FAKE_PORT}/v1/messages"
        via = f"http://127.0.0.1:{GATEWAY_PORT}/v1/messages"
        print(f"Upstream answers after {args.upstream_delay_ms} ms; {args.duration:.0f}s per level; "
              f"{'streaming' if args.stream else 'non-streaming'}.\n")
        print("| Clients | Gateway req/s | p50 | p95 | p99 | Errors | Added p50 | Added p95 |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        results = []
        for c in args.concurrency:
            d = await run_level(direct, key, c, args.duration, args.stream)
            g = await run_level(via, key, c, args.duration, args.stream)
            results.append({"clients": c, "direct": d, "gateway": g})
            print(f"| {c} | {g['rps']:.0f} | {g['p50']:.0f} ms | {g['p95']:.0f} ms | {g['p99']:.0f} ms | "
                  f"{g['errors']}/{g['requests']} | {g['p50'] - d['p50']:+.1f} ms | {g['p95'] - d['p95']:+.1f} ms |")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
    finally:
        stop(gateway)
        stop(fake)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", default="1,10,50,100", type=lambda s: [int(x) for x in s.split(",")])
    parser.add_argument("--duration", type=float, default=15)
    parser.add_argument("--upstream-delay-ms", type=int, default=300)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--out", help="also write results as JSON")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
