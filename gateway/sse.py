"""Reads Anthropic's server-sent events as they stream past, without changing them."""

import json


class StreamTracker:
    """Collects model, usage, stop reason and errors from a Messages API stream.

    Feed it raw chunks in order. Chunks can split events anywhere, so it buffers
    until it has a complete event (terminated by a blank line).
    """

    def __init__(self) -> None:
        self._buffer = b""
        self.model: str | None = None
        self.usage: dict = {}
        self.stop_reason: str | None = None
        self.error_type: str | None = None
        self.saw_content = False  # True once the first content_block_delta arrives.
        self.completed = False  # True once message_stop arrives.

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk.replace(b"\r\n", b"\n")
        while b"\n\n" in self._buffer:
            event, self._buffer = self._buffer.split(b"\n\n", 1)
            self._handle(event)

    def _handle(self, event: bytes) -> None:
        data_lines = [line[5:].lstrip() for line in event.split(b"\n") if line.startswith(b"data:")]
        if not data_lines:
            return
        try:
            data = json.loads(b"\n".join(data_lines))
        except ValueError:
            return

        kind = data.get("type")
        if kind == "message_start":
            message = data.get("message") or {}
            self.model = message.get("model")
            self._merge_usage(message.get("usage"))
        elif kind == "content_block_delta":
            self.saw_content = True
        elif kind == "message_delta":
            self.stop_reason = (data.get("delta") or {}).get("stop_reason") or self.stop_reason
            self._merge_usage(data.get("usage"))  # Counts here are cumulative, so they replace earlier ones.
        elif kind == "message_stop":
            self.completed = True
        elif kind == "error":
            self.error_type = (data.get("error") or {}).get("type") or "unknown"

    def _merge_usage(self, usage: dict | None) -> None:
        for key, value in (usage or {}).items():
            if value is not None:
                self.usage[key] = value
