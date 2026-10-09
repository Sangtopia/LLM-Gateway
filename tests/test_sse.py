import json

from gateway.sse import StreamTracker


def event(data: dict) -> bytes:
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()


STREAM = b"".join(
    [
        event({"type": "message_start", "message": {"model": "claude-haiku-5-5", "usage": {"input_tokens": 17, "output_tokens": 1}}}),
        event({"type": "ping"}),
        event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
        event({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 10, "input_tokens": None}}),
        event({"type": "message_stop"}),
    ]
)


def test_collects_model_usage_and_stop_reason():
    tracker = StreamTracker()
    tracker.feed(STREAM)
    assert tracker.model == "claude-haiku-5-5"
    assert tracker.usage == {"input_tokens": 17, "output_tokens": 10}  # None in message_delta doesn't erase 17.
    assert tracker.stop_reason == "end_turn"
    assert tracker.saw_content and tracker.completed
    assert tracker.error_type is None


def test_handles_events_split_across_chunks_at_any_byte():
    tracker = StreamTracker()
    for i in range(len(STREAM)):
        tracker.feed(STREAM[i : i + 1])
    assert tracker.usage == {"input_tokens": 17, "output_tokens": 10}
    assert tracker.completed


def test_handles_crlf_line_endings():
    tracker = StreamTracker()
    tracker.feed(STREAM.replace(b"\n", b"\r\n"))
    assert tracker.completed


def test_records_error_events():
    tracker = StreamTracker()
    tracker.feed(event({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}))
    assert tracker.error_type == "overloaded_error"
    assert not tracker.completed
