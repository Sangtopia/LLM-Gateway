import pytest

from gateway.models import cache_savings_usd
from gateway.promptcache import apply_prompt_caching

EPHEMERAL = {"type": "ephemeral"}
TOOL = {"name": "lookup_user", "input_schema": {"type": "object"}}


def test_string_system_prompt_gets_a_breakpoint():
    out = apply_prompt_caching({"system": "You are helpful.", "messages": [{"role": "user", "content": "hi"}]})
    assert out["system"] == [{"type": "text", "text": "You are helpful.", "cache_control": EPHEMERAL}]
    assert "cache_control" not in out  # Single-shot, no tools: no automatic tail breakpoint.


def test_block_system_prompt_marks_last_block():
    request = {"system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "messages": []}
    out = apply_prompt_caching(request)
    assert "cache_control" not in out["system"][0]
    assert out["system"][1]["cache_control"] == EPHEMERAL


def test_agent_requests_also_get_automatic_tail_caching():
    with_tools = apply_prompt_caching({"system": "s", "tools": [TOOL], "messages": [{"role": "user", "content": "x"}]})
    multi_turn = apply_prompt_caching({"system": "s", "messages": [{}, {}, {}]})
    assert with_tools["cache_control"] == EPHEMERAL
    assert multi_turn["cache_control"] == EPHEMERAL


def test_tools_without_system_get_the_breakpoint():
    out = apply_prompt_caching({"tools": [TOOL, {**TOOL, "name": "b"}], "messages": [{"role": "user", "content": "x"}]})
    assert out["tools"][-1]["cache_control"] == EPHEMERAL
    assert "cache_control" not in out["tools"][0]


@pytest.mark.parametrize("request_", [
    {"system": "s", "messages": [], "cache_control": EPHEMERAL},
    {"system": [{"type": "text", "text": "s", "cache_control": EPHEMERAL}], "messages": []},
    {"system": "s", "messages": [{"role": "user", "content": [{"type": "text", "text": "x", "cache_control": EPHEMERAL}]}]},
])
def test_requests_that_already_cache_are_left_alone(request_):
    assert apply_prompt_caching(request_) is None


def test_nothing_to_cache():
    assert apply_prompt_caching({"messages": [{"role": "user", "content": "hi"}]}) is None


def test_original_request_is_not_modified():
    request = {"system": [{"type": "text", "text": "s"}], "messages": []}
    apply_prompt_caching(request)
    assert request == {"system": [{"type": "text", "text": "s"}], "messages": []}


def test_savings_on_opus_5_5_reads():
    # 10K tokens read from cache: $4/MTok uncached vs $0.20 cached.
    usage = {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 10_000}
    assert cache_savings_usd("claude-opus-5-5", usage) == pytest.approx(10_000 * (4.00 - 0.20) / 1e6)


def test_unread_writes_are_a_loss():
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 10_000}
    assert cache_savings_usd("claude-opus-5-5", usage) == pytest.approx(-10_000 * 4.00 * 0.25 / 1e6)


def test_no_caching_no_savings():
    assert cache_savings_usd("claude-opus-5-5", {"input_tokens": 10, "output_tokens": 10}) == 0.0
    assert cache_savings_usd("not-a-model", {"input_tokens": 10}) is None
