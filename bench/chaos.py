"""Failure test: take a model down mid-run and watch traffic move to the fallback and back.

    python -m bench.chaos [--clients 20] [--duration 60] [--down 15:40]

Steady load for claude-sonnet-5-5 goes through the gateway to the fake upstream. During
the --down window (seconds) the fake answers every claude-sonnet-5-5 request with 529
overloaded; other models keep working. The gateway runs with a 10s circuit cooldown so
the recovery is visible within the run. Prints, per 5-second window: requests, success
rate, which model served them, and p95 latency.
"""

import argparse
import asyncio
import json
import statistics
import tempfile
import time
from collections import Counter, defaultdict

import httpx

from bench.overhead import FAKE_PORT, GATEWAY_PORT, create_key, start, stop

MODEL = "claude-sonnet-5-5"
BODY = {"model": MODEL, "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]}
WINDOW_S = 5


async def main_async(args) -> None:
    db_url = f"sqlite+aiosqlite:///{tempfile.mkdtemp()}/chaos.db"
    key = await create_key(db_url)
    fake = start("bench.fake_upstream:app", FAKE_PORT)
    gateway = start("gateway.main:app", GATEWAY_PORT, env={
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{FAKE_PORT}", "ANTHROPIC_API_KEY": "fake", "DATABASE_URL": db_url,
        "CIRCUIT_COOLDOWN_S": "10",
    })
    control = f"http://127.0.0.1:{FAKE_PORT}/_control"
    url = f"http://127.0.0.1:{GATEWAY_PORT}/v1/messages"
    down_from, down_to = args.down
    samples: list[tuple[float, bool, str | None, float]] = []  # (t, ok, served model, latency ms)

    try:
        httpx.post(control, json={"delay_ms": args.upstream_delay_ms})
        started = time.perf_counter()

        async def chaos():
            await asyncio.sleep(down_from)
            httpx.post(control, json={"delay_ms": args.upstream_delay_ms, "fail_models": [MODEL], "status": 529})
            await asyncio.sleep(down_to - down_from)
            httpx.post(control, json={"delay_ms": args.upstream_delay_ms})

        async def client_loop(client: httpx.AsyncClient):
            while (now := time.perf_counter() - started) < args.duration:
                t0 = time.perf_counter()
                try:
                    resp = await client.post(url, json=BODY)
                    ok, served = resp.status_code == 200, resp.headers.get("x-gateway-model")
                except httpx.HTTPError:
                    ok, served = False, None
                samples.append((now, ok, served, (time.perf_counter() - t0) * 1000))

        limits = httpx.Limits(max_connections=args.clients)
        async with httpx.AsyncClient(headers={"x-api-key": key}, timeout=30, limits=limits) as client:
            await asyncio.gather(chaos(), *(client_loop(client) for _ in range(args.clients)))
    finally:
        stop(gateway)
        stop(fake)

    windows = defaultdict(list)
    for s in samples:
        windows[int(s[0] // WINDOW_S)].append(s)
    print(f"{args.clients} clients for {args.duration}s; {MODEL} returns 529 from {down_from}s to {down_to}s.\n")
    print("| Seconds | Requests | Succeeded | Served by | p95 |")
    print("| --- | --- | --- | --- | --- |")
    rows = []
    for w in sorted(windows):
        group = windows[w]
        ok = sum(s[1] for s in group)
        served = Counter((s[2] or "failed").replace("claude-", "") for s in group)
        p95 = statistics.quantiles([s[3] for s in group], n=20)[18] if len(group) > 1 else group[0][3]
        state = " (down)" if down_from <= w * WINDOW_S < down_to else ""
        print(f"| {w * WINDOW_S}-{(w + 1) * WINDOW_S}{state} | {len(group)} | {ok / len(group):.1%} | "
              f"{', '.join(f'{m} {n}' for m, n in served.most_common())} | {p95:.0f} ms |")
        rows.append({"start_s": w * WINDOW_S, "requests": len(group), "succeeded": ok, "served": dict(served), "p95_ms": p95})
    total_ok = sum(s[1] for s in samples)
    during = [s for s in samples if down_from <= s[0] < down_to]
    print(f"\nOverall: {total_ok}/{len(samples)} succeeded ({total_ok / len(samples):.2%}); while {MODEL} was down: "
          f"{sum(s[1] for s in during)}/{len(during)} ({sum(s[1] for s in during) / max(1, len(during)):.2%}).")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"windows": rows, "succeeded": total_ok, "requests": len(samples)}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--down", default="15:40", type=lambda s: tuple(float(x) for x in s.split(":")))
    parser.add_argument("--upstream-delay-ms", type=int, default=300)
    parser.add_argument("--out", help="also write results as JSON")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
