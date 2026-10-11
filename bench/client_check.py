"""Is the HTTP client the bottleneck? Drive the fake upstream directly with httpx and aiohttp.

    python -m bench.client_check [--concurrency 50,100,200] [--duration 10]

The gateway calls Anthropic with httpx, so httpx's limits here are the gateway's limits too.
"""

import argparse
import asyncio
import statistics
import time

import aiohttp
import httpx
import httpx_aiohttp

from bench.overhead import FAKE_PORT, start, stop

URL = f"http://127.0.0.1:{FAKE_PORT}/v1/messages"
BODY = {"model": "claude-haiku-5-5", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]}


async def drive(send, concurrency: int, duration: float) -> tuple[float, float]:
    latencies = []
    deadline = time.perf_counter() + duration

    async def worker():
        while time.perf_counter() < deadline:
            t = time.perf_counter()
            await send()
            latencies.append((time.perf_counter() - t) * 1000)

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return len(latencies) / (time.perf_counter() - started), statistics.quantiles(latencies, n=20)[18]


async def with_httpx(concurrency: int, duration: float, keepalive: int) -> tuple[float, float]:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=keepalive)
    async with httpx.AsyncClient(limits=limits, timeout=30) as client:
        async def send():
            (await client.post(URL, json=BODY)).raise_for_status()
        return await drive(send, concurrency, duration)


async def with_sharded_httpx(concurrency: int, duration: float, shard_size: int = 25) -> tuple[float, float]:
    """Several small httpx clients used in rotation, instead of one large pool."""
    shards = max(1, concurrency // shard_size)
    limits = httpx.Limits(max_connections=shard_size, max_keepalive_connections=shard_size)
    clients = [httpx.AsyncClient(limits=limits, timeout=30) for _ in range(shards)]
    turn = 0
    try:
        async def send():
            nonlocal turn
            client = clients[turn % shards]
            turn += 1
            (await client.post(URL, json=BODY)).raise_for_status()
        return await drive(send, concurrency, duration)
    finally:
        for client in clients:
            await client.aclose()


async def with_httpx_on_aiohttp(concurrency: int, duration: float) -> tuple[float, float]:
    """httpx's API with aiohttp doing the networking (the httpx-aiohttp transport)."""
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    transport = httpx_aiohttp.AiohttpTransport(limits=limits)
    async with httpx.AsyncClient(transport=transport, timeout=30) as client:
        async def send():
            (await client.post(URL, json=BODY)).raise_for_status()
        return await drive(send, concurrency, duration)


async def with_aiohttp(concurrency: int, duration: float) -> tuple[float, float]:
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=concurrency)) as session:
        async def send():
            async with session.post(URL, json=BODY) as resp:
                resp.raise_for_status()
                await resp.read()
        return await drive(send, concurrency, duration)


async def main_async(args) -> None:
    fake = start("bench.fake_upstream:app", FAKE_PORT, extra_args=["--timeout-keep-alive", str(args.server_keepalive_s)])
    cell = lambda r: f"{r[0]:.0f} req/s, p95 {r[1]:.0f} ms"  # noqa: E731
    try:
        httpx.post(f"http://127.0.0.1:{FAKE_PORT}/_control", json={"delay_ms": 300})
        print("| Clients | httpx, keep-alive 20 (default) | httpx, keep-alive = clients | httpx, 25-connection shards | httpx on aiohttp | aiohttp |")
        print("| --- | --- | --- | --- | --- | --- |")
        for c in args.concurrency:
            default = await with_httpx(c, args.duration, keepalive=20)
            sized = await with_httpx(c, args.duration, keepalive=c)
            sharded = await with_sharded_httpx(c, args.duration)
            bridged = await with_httpx_on_aiohttp(c, args.duration)
            aio = await with_aiohttp(c, args.duration)
            print(f"| {c} | {cell(default)} | {cell(sized)} | {cell(sharded)} | {cell(bridged)} | {cell(aio)} |")
    finally:
        stop(fake)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", default="50,100,200", type=lambda s: [int(x) for x in s.split(",")])
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--server-keepalive-s", type=int, default=5, help="uvicorn's keep-alive timeout (its default is 5)")
    asyncio.run(main_async(parser.parse_args()))
