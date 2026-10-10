"""A stand-in for the Anthropic API that answers instantly with a canned reply.

Benchmarks point the gateway at this so the numbers measure only the gateway.
It can also misbehave on command, for testing retries and fallbacks:

    POST /_control {"fail_next": 2, "status": 529}   next 2 requests fail with 529
    POST /_control {"fail_rate": 0.3, "status": 500} 30% of requests fail
    POST /_control {"delay_ms": 2000}                 wait before answering
    POST /_control {"fail_models": ["claude-sonnet-5-5"], "status": 529}
                                                      that model always fails (others work)
    POST /_control {"stream_fail": "error_before_content", "stream_fail_next": 1}
                                                      next stream sends an overloaded error before any text
    POST /_control {"stream_fail": "cut_after_content", "stream_fail_next": 1}
                                                      next stream drops the connection after some text
    POST /_control {}                                 back to normal
    GET  /_control                                    current settings and request count
"""

import asyncio
import json
import random

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

app = FastAPI()

MESSAGE = {
    "id": "msg_fake",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "Hello there, friend!"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 17, "output_tokens": 10},
}

ERROR_TYPES = {429: "rate_limit_error", 500: "api_error", 503: "api_error", 529: "overloaded_error"}

DEFAULTS = {
    "fail_next": 0,
    "fail_rate": 0.0,
    "status": 529,
    "delay_ms": 0,
    "fail_models": [],
    "stream_fail": None,
    "stream_fail_next": 0,
}
state = {**DEFAULTS, "requests": 0}


def sse(event: dict) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def stream_events(model: str) -> list[bytes]:
    return [
        sse({"type": "message_start", "message": {**MESSAGE, "model": model, "content": [], "usage": {"input_tokens": 17, "output_tokens": 0}}}),
        sse({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        *[sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": w}}) for w in ["Hello ", "there, ", "friend!"]],
        sse({"type": "content_block_stop", "index": 0}),
        sse({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 10}}),
        sse({"type": "message_stop"}),
    ]


@app.post("/_control")
async def control(request: Request) -> dict:
    settings = await request.json()
    state.update({**DEFAULTS, **settings, "requests": 0})
    return state


@app.get("/_control")
async def get_control() -> dict:
    return state


def should_fail(model: str | None) -> bool:
    if model in state["fail_models"]:
        return True
    if state["fail_next"] > 0:
        state["fail_next"] -= 1
        return True
    return random.random() < state["fail_rate"]


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    body = await request.json()
    state["requests"] += 1
    if state["delay_ms"]:
        await asyncio.sleep(state["delay_ms"] / 1000)

    if should_fail(body.get("model")):
        status = state["status"]
        error = {"type": "error", "error": {"type": ERROR_TYPES.get(status, "api_error"), "message": "Injected failure"}}
        return Response(json.dumps(error), status_code=status, media_type="application/json")

    model = body.get("model", MESSAGE["model"])
    if body.get("stream"):
        stream_fail = None
        if state["stream_fail"] and state["stream_fail_next"] > 0:
            state["stream_fail_next"] -= 1
            stream_fail = state["stream_fail"]

        async def events():
            all_events = stream_events(model)
            if stream_fail == "error_before_content":
                yield all_events[0]
                yield sse({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
                return
            for i, event in enumerate(all_events):
                if stream_fail == "cut_after_content" and i == 4:  # After two text deltas.
                    raise ConnectionResetError("injected disconnect")
                yield event
                await asyncio.sleep(0)

        return StreamingResponse(events(), media_type="text/event-stream")
    return Response(json.dumps({**MESSAGE, "model": model}), media_type="application/json", headers={"request-id": "req_fake"})
