"""The gateway's real upstream transport (httpx API, aiohttp networking) against real sockets.

The other proxy tests use httpx.MockTransport. These check that network failures through
httpx-aiohttp still surface as the httpx exceptions the retry and stream logic expect.
"""

import json
import socket
import threading

import httpx
import httpx_aiohttp
import pytest

from gateway.main import app
from tests.test_proxy import SSE_EVENTS, make_client


class OneShotServer:
    """A TCP server on a free port that handles every connection with `behave(conn)`."""

    def __init__(self, behave):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        self.port = self.sock.getsockname()[1]
        self.behave = behave
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        with conn:
            conn.recv(65536)  # The request.
            self.behave(conn)

    def close(self):
        self.sock.close()


def real_client(base_url: str, read_timeout: float = 5.0):
    client, log = make_client(lambda req: httpx.Response(500))
    app.state.http = httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(read_timeout, connect=2.0),
        transport=httpx_aiohttp.AiohttpTransport(),
    )
    return client, log


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_connection_refused_is_retried_then_502():
    client, log = real_client(f"http://127.0.0.1:{free_port()}")  # Nothing listening.
    resp = client.post("/v1/messages", json={"model": "claude-haiku-5-5"})

    assert resp.status_code == 502
    [record] = log.records
    assert record["error_type"] == "upstream_unreachable"
    # Retried as a connect failure. Linux refuses at once (ConnectError); Windows retries the SYN
    # until the connect timeout (ConnectTimeout). The gateway retries both.
    assert len(record["attempts"]) == 3
    assert {a["error"] for a in record["attempts"]} <= {"ConnectError", "ConnectTimeout"}


def test_server_that_never_answers_is_a_504_not_retried():
    server = OneShotServer(lambda conn: threading.Event().wait(3))  # Accepts, then says nothing.
    try:
        client, log = real_client(f"http://127.0.0.1:{server.port}", read_timeout=0.5)
        resp = client.post("/v1/messages", json={"model": "claude-haiku-5-5"})
    finally:
        server.close()

    assert resp.status_code == 504
    assert [a["error"] for a in log.records[0]["attempts"]] == ["ReadTimeout"]


def test_stream_cut_after_content_ends_with_error_event():
    def send_partial_stream(conn):
        head = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
        body = b"".join(SSE_EVENTS[:2])  # message_start and one text delta, then the connection drops.
        conn.sendall(head + f"{len(body):x}\r\n".encode() + body + b"\r\n")

    server = OneShotServer(send_partial_stream)
    try:
        client, log = real_client(f"http://127.0.0.1:{server.port}")
        with client.stream("POST", "/v1/messages", json={"model": "claude-haiku-5-5", "stream": True}) as resp:
            body = resp.read()
    finally:
        server.close()

    events = [json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith("data: ")]
    assert [e["type"] for e in events] == ["message_start", "content_block_delta", "error"]
    assert log.records[0]["outcome"] == "error"
    assert log.records[0]["error_type"] in ("upstream_stream_interrupted", "upstream_timeout")


@pytest.fixture(autouse=True)
def _close_real_client():
    yield
    # Each test made its own aiohttp-backed client inside the TestClient's event loop; drop it.
    app.state.http = None
