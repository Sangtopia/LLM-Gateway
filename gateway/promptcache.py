"""Automatic Anthropic prompt caching.

Many apps never set `cache_control`, so they pay full price to re-send the same
system prompt, tool definitions and conversation history on every call. For a
request with no `cache_control` of its own, the gateway adds:

  - an explicit breakpoint on the last system block (or, with no system prompt,
    the last tool), so the shared prefix every request starts with is cached;
  - top-level automatic caching when the request looks like a conversation that
    will continue (it has tools, or more than one message), so each turn reads
    the previous turn's prefix from cache.

Single-shot requests without tools get only the system breakpoint: an automatic
breakpoint after their unique content would be a cache write that is never read,
a pure 25% surcharge on those tokens.

Requests that already use `cache_control` anywhere are left alone; the app knows
its own prompt better than the gateway does. Apps can opt out per request with
`x-gateway-prompt-cache: off`.
"""

import copy

EPHEMERAL = {"type": "ephemeral"}


def _has_cache_control(value) -> bool:
    if isinstance(value, dict):
        return "cache_control" in value or any(_has_cache_control(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_cache_control(v) for v in value)
    return False


def apply_prompt_caching(request: dict) -> dict | None:
    """A copy of `request` with cache breakpoints added, or None if it should be left as is."""
    if _has_cache_control(request):
        return None
    out = copy.deepcopy(request)
    changed = False

    system = out.get("system")
    if isinstance(system, str) and system:
        out["system"] = [{"type": "text", "text": system, "cache_control": EPHEMERAL}]
        changed = True
    elif isinstance(system, list) and system and isinstance(system[-1], dict):
        system[-1]["cache_control"] = EPHEMERAL
        changed = True
    elif isinstance(out.get("tools"), list) and out["tools"] and isinstance(out["tools"][-1], dict):
        out["tools"][-1]["cache_control"] = EPHEMERAL  # Tools render first, so this caches all of them.
        changed = True

    messages = out.get("messages")
    if out.get("tools") or (isinstance(messages, list) and len(messages) > 1):
        out["cache_control"] = EPHEMERAL
        changed = True
    return out if changed else None

