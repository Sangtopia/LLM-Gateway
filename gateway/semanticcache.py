"""Semantic response cache: reuse an answer when a new request means the same thing.

Opt in per request with `x-gateway-cache: semantic` (it also uses the exact-match
cache first). Deliberately narrow, because a wrong cached answer is worse than a
miss:

  - only single-turn, text-only requests (one user message);
  - only the user's text is compared by meaning. Everything else - gateway key,
    model, system prompt, tools, parameters, API version, betas and the app's
    `x-gateway-cache-scope` - must match exactly, so it forms the scope a lookup
    searches within. Measured on 64 help desk tickets (bench/semantic_eval.py),
    paraphrases never matched the wrong ticket, but the same text from a different
    sender matched at 0.92-0.98: without a per-user scope, one user would get
    another's answer;
  - a hit needs cosine similarity >= the threshold (default 0.95, adjustable per
    request with `x-gateway-cache-threshold`, never below 0.80).

Embeddings come from a small local model (BAAI/bge-small-en-v1.5 via fastembed,
384 dimensions, a few ms on CPU): no extra API, account or per-call cost. Each
scope is a capped Redis list of (embedding, response) entries; a lookup reads the
list and compares in numpy, which is fast at this size and needs no vector index.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING

from redis.asyncio import Redis
from redis.exceptions import RedisError

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_THRESHOLD = 0.95
MIN_THRESHOLD = 0.80
MAX_ENTRIES_PER_SCOPE = 200
MAX_ENTRY_BYTES = 64 * 1024


class Embedder:
    """Loads the embedding model on first use (a one-time download of about 30 MB)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _embed_sync(self, text: str) -> np.ndarray:
        import numpy as np  # Optional dependencies (requirements-dev.txt): only the semantic cache needs them.

        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name)
        vector = np.asarray(next(iter(self._model.embed([text]))), dtype=np.float32)
        return vector / np.linalg.norm(vector)

    async def embed(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self._embed_sync, text)


def semantic_threshold(headers) -> float:
    try:
        value = float(headers.get("x-gateway-cache-threshold", DEFAULT_THRESHOLD))
    except ValueError:
        return DEFAULT_THRESHOLD
    return min(1.0, max(MIN_THRESHOLD, value))


def semantic_scope(request: dict, headers, key_id: str) -> tuple[str, str] | None:
    """(scope id, user text) for a request the semantic cache may serve, else None."""
    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "user":
        return None
    content = messages[0].get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list) and all(isinstance(b, dict) and b.get("type") == "text" for b in content):
        text = "\n".join(b.get("text", "") for b in content)
    else:
        return None  # Images, documents, tool results: never matched by meaning.
    if not text.strip():
        return None
    rest = {k: v for k, v in request.items() if k not in ("messages", "stream")}
    material = json.dumps(
        {"rest": rest, "version": headers.get("anthropic-version", ""), "beta": headers.get("anthropic-beta", ""),
         "key": key_id, "scope": headers.get("x-gateway-cache-scope", ""), "stream": request.get("stream") is True},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "gw:sem:" + hashlib.sha256(material.encode()).hexdigest(), text


class SemanticCache:
    def __init__(self, redis: Redis, embedder: Embedder):
        self._redis = redis
        self.embedder = embedder
        self._down_until = 0.0
        self._background: set[asyncio.Task] = set()

    async def lookup(self, scope: str, vector: np.ndarray, threshold: float) -> tuple[dict, float] | None:
        """The closest cached entry in `scope` if its similarity clears `threshold`."""
        if time.monotonic() < self._down_until:
            return None
        try:
            raw = await self._redis.lrange(scope, 0, -1)
        except RedisError:
            self._down_until = time.monotonic() + 5.0
            log.warning("semantic cache unavailable", exc_info=True)
            return None
        if not raw:
            return None
        import numpy as np

        entries = [json.loads(r) for r in raw]
        matrix = np.stack([np.frombuffer(base64.b64decode(e["v"]), dtype=np.float32) for e in entries])
        similarities = matrix @ vector
        best = int(np.argmax(similarities))
        similarity = float(similarities[best])
        return (entries[best], similarity) if similarity >= threshold else None

    def store_later(self, scope: str, vector: np.ndarray, body: bytes, content_type: str, model: str | None,
                    cost_usd: float | None, ttl_s: int) -> None:
        if len(body) > MAX_ENTRY_BYTES or time.monotonic() < self._down_until:
            return
        import numpy as np

        entry = {
            "v": base64.b64encode(vector.astype(np.float32).tobytes()).decode(),
            "body": body.decode("utf-8"), "content_type": content_type, "model": model, "cost_usd": cost_usd,
        }
        task = asyncio.create_task(self._store(scope, json.dumps(entry), ttl_s))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _store(self, scope: str, entry: str, ttl_s: int) -> None:
        try:
            pipe = self._redis.pipeline(transaction=False)
            pipe.lpush(scope, entry)
            pipe.ltrim(scope, 0, MAX_ENTRIES_PER_SCOPE - 1)
            pipe.expire(scope, ttl_s)
            await pipe.execute()
        except RedisError:
            self._down_until = time.monotonic() + 5.0
            log.warning("could not store semantic cache entry", exc_info=True)

    async def drain(self) -> None:
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)
