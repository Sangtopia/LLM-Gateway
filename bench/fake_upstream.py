"""A stand-in for the Anthropic API that answers instantly with a canned reply.

Benchmarks point the gateway at this so the numbers measure only the gateway.
"""

import json

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


def sse(event: dict) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


STREAM_EVENTS = [
    sse({"type": "message_start", "message": {**MESSAGE, "content": [], "usage": {"input_tokens": 17, "output_tokens": 0}}}),
    sse({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    *[sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": w}}) for w in ["Hello ", "there, ", "friend!"]],
    sse({"type": "content_block_stop", "index": 0}),
    sse({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 10}}),
    sse({"type": "message_stop"}),
]


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    body = await request.json()
    if body.get("stream"):
        async def events():
            for event in STREAM_EVENTS:
                yield event
        return StreamingResponse(events(), media_type="text/event-stream")
    return Response(json.dumps(MESSAGE), media_type="application/json", headers={"request-id": "req_fake"})
