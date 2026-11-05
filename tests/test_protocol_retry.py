"""Exercise protocol rejection through the real invocation loop."""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from src.llm_client import LLMClient


@pytest.fixture
def client():
    """Build an invocation client with isolated model, transport, and storage."""
    conversations = SimpleNamespace(
        get_conversation=AsyncMock(return_value=SimpleNamespace(history="[]")),
        update_conversation=AsyncMock(),
    )
    plugin = SimpleNamespace(
        config={
            "log_llm_prompt": False,
            "history_as_role_contexts": False,
            "memory_importance_scoring_enabled": False,
            "impression_stream_enabled": False,
            "mood_enabled": False,
        },
        state=SimpleNamespace(
            check_interrupt=lambda *_args: False,
            get_self_id=lambda _umo: "audit-bot",
            get_self_name=lambda _umo: "Alice",
        ),
        context=SimpleNamespace(conversation_manager=conversations),
        db_mgr=SimpleNamespace(ensure_message_embedded=AsyncMock()),
    )
    instance = LLMClient(plugin)
    instance._prepare_llm_config = lambda *_args: {
        "provider": None,
        "use_custom_llm": True,
        "llm": {
            "base": "https://example.invalid",
            "key": "test",
            "model": "test",
            "temp": 0,
        },
    }
    instance._prepare_conversation_context = AsyncMock(
        return_value={
            "persona": None,
            "conversation": SimpleNamespace(cid="audit-protocol"),
        }
    )
    instance._detect_intents = lambda *_args: {}
    instance._setup_tools = lambda *_args: (None, False)
    instance._build_instructions = lambda *_args, **_kwargs: {"suffix": "JSON only"}
    instance._build_system_prompts = lambda *_args, **_kwargs: "system"
    instance._process_and_send_final_reply = AsyncMock(return_value=True)
    instance._execute_agent_tool = AsyncMock(return_value="tool result")
    return instance


def response(text, tool=False):
    """Create a provider-compatible completion for the invocation loop.

    Args:
        text: Completion body returned by the upstream model.
        tool: Whether this is a native tool call.

    Returns:
        Provider-compatible completion object.
    """
    return SimpleNamespace(
        completion_text=text,
        tools_call_name=["audit_tool"] if tool else [],
        tools_call_args=[{}] if tool else [],
        tools_call_ids=["audit-call"] if tool else [],
        raw_completion=None,
    )


GOOD = json.dumps(
    {"control": {"decision": "reply_only"}, "response": {"text": "Recovered"}}
)
BAD = "</｜｜DSML｜｜ calls>"


@pytest.mark.asyncio
@pytest.mark.parametrize("custom", [True, False])
@pytest.mark.parametrize(
    "bad",
    [
        BAD,
        'Hello\n<||DSML|| invoke name="audit_tool">',
        json.dumps(
            {
                "control": {"decision": "reply_only"},
                "response": {"segments": [{"text": BAD}]},
            }
        ),
    ],
)
async def test_protocol_leak_retries_once_before_sending_and_saving(
    client, custom, bad
):
    """Reject raw and escaped markers on both provider paths before side effects."""
    calls = []
    completions = iter([response(bad), response(GOOD)])

    async def generate(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return next(completions)

    if custom:
        client._call_custom_llm = generate
    else:
        provider = SimpleNamespace(text_chat=generate)
        client._prepare_llm_config = lambda *_args: {
            "provider": provider,
            "use_custom_llm": False,
        }
    result = await client.invoke_llm(
        "default:FriendMessage:audit-protocol", "hello", ""
    )
    await asyncio.sleep(0)
    assert result is True
    assert len(calls) == 2
    assert calls[0]["contexts"] == calls[1]["contexts"] == []
    assert "上一次返回包含了错误的 DSML" in calls[1]["prompt"]
    client._process_and_send_final_reply.assert_awaited_once()
    client._execute_agent_tool.assert_not_awaited()
    saved = (
        client.plugin.context.conversation_manager.update_conversation.await_args.args[
            2
        ]
    )
    assert saved == [
        {
            "role": "assistant",
            "content": "Recovered",
            "sender_name": "Alice",
            "sender_id": "audit-bot",
        }
    ]


@pytest.mark.asyncio
async def test_repeated_protocol_leak_fails_without_send_history_or_tool_execution(
    client,
):
    """The second malformed completion must terminate this invocation."""
    client._call_custom_llm = AsyncMock(return_value=response(BAD, tool=True))
    assert (
        await client.invoke_llm("default:FriendMessage:audit-protocol", "hello", "")
        is False
    )
    assert client._call_custom_llm.await_count == 2
    client._process_and_send_final_reply.assert_not_awaited()
    client.plugin.context.conversation_manager.update_conversation.assert_not_awaited()
    client.plugin.db_mgr.ensure_message_embedded.assert_not_awaited()
    client._execute_agent_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_after_tool_result_preserves_context_without_repeating_tool(client):
    """Retry only the malformed model step after an already completed tool call."""
    tool = SimpleNamespace(tools=[SimpleNamespace(name="audit_tool")])
    client._setup_tools = lambda *_args: (tool, True)
    calls = []
    completions = iter([response("", tool=True), response(BAD), response(GOOD)])

    async def generate(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return next(completions)

    client._call_custom_llm = generate
    assert (
        await client.invoke_llm("default:FriendMessage:audit-protocol", "hello", "")
        is True
    )
    await asyncio.sleep(0)
    assert len(calls) == 3
    assert calls[1]["contexts"] == calls[2]["contexts"]
    assert any(m.get("role") == "tool" for m in calls[2]["contexts"])
    client._execute_agent_tool.assert_awaited_once()
    client._process_and_send_final_reply.assert_awaited_once()


@pytest.mark.asyncio
async def test_normal_completion_does_not_retry(client):
    """A valid first completion still uses exactly one model call."""
    client._call_custom_llm = AsyncMock(return_value=response(GOOD))
    assert (
        await client.invoke_llm("default:FriendMessage:audit-protocol", "hello", "")
        is True
    )
    await asyncio.sleep(0)
    client._call_custom_llm.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin, persona_phase", [("name_call", "read_air"), ("platform_at", "at")]
)
async def test_name_candidate_uses_provisional_persona_and_preserves_confirmation(
    client, origin, persona_phase
):
    from src.core.scheduler import Scheduler
    from src.core.state import MemChatState

    state = MemChatState()
    client.plugin.state = state
    client.plugin.scheduler = Scheduler(None, None, {}, state, None, None)
    event = SimpleNamespace(
        get_group_id=lambda: "42", _mem_chat_attention_origin=origin
    )
    phases = []

    def build_system(*args, **kwargs):
        phases.append(kwargs["task_type"])
        return "system"

    client._build_system_prompts = build_system
    client._call_custom_llm = AsyncMock(
        return_value=response(
            json.dumps(
                {
                    "control": {"decision": "reply_only", "addressed": False},
                    "response": {"text": "understood"},
                }
            )
        )
    )
    try:
        assert await client.invoke_llm(
            "default:GroupMessage:42",
            "甲对乙说：爱丽丝是对的",
            "",
            event=event,
            task_type="at",
        )
        assert phases and set(phases) == {persona_phase}
        assert event._mem_chat_last_structured_payload["control"]["addressed"] is False
        sent_prompt = client._call_custom_llm.await_args.kwargs["prompt"]
        assert ("control.addressed" in sent_prompt) is (origin == "name_call")
    finally:
        client.plugin.scheduler.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("custom", [True, False])
@pytest.mark.parametrize(
    "model,expected",
    [
        ("deepseek-flash", True),
        ("deepseek-v4.1-flash", True),
        ("gemini-3.8-flash", False),
        ("deepseek-chat", False),
    ],
)
async def test_model_style_survives_tools_and_protocol_retry(
    client, custom, model, expected
):
    """Apply the exact model-specific style through continuation and retry."""
    from pathlib import Path

    style = (
        (Path(__file__).resolve().parents[1] / "personas" / "deepseek_flash_style.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    calls = []
    completions = iter([response("", tool=True), response(BAD), response(GOOD)])

    async def generate(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return next(completions)

    config = client._prepare_llm_config(None, None)
    config["llm"]["model"] = model
    config["provider"] = SimpleNamespace(text_chat=generate, get_model=lambda: model)
    config["use_custom_llm"] = custom
    client._prepare_llm_config = lambda *_args: config
    client._call_custom_llm = generate
    tool = SimpleNamespace(tools=[SimpleNamespace(name="audit_tool")])
    client._setup_tools = lambda *_args: (tool, True)
    client._split_history_into_contexts = lambda *_args: (
        "history stripped",
        [{"role": "user", "content": "earlier"}],
    )
    client.plugin.config["history_as_role_contexts"] = True
    assert await client.invoke_llm(
        "default:FriendMessage:audit-protocol", "hello", "history"
    )
    await asyncio.sleep(0)
    assert len(calls) == 3
    for call in calls:
        assert call["system_prompt"].count(style) == int(expected)
    client._execute_agent_tool.assert_awaited_once()


@pytest.mark.asyncio
async def test_optional_group_bubble_does_not_advertise_or_run_tools(client):
    """Keep spontaneous expression separate from actions requested by old context."""
    from src.core.scheduler import Scheduler
    from src.core.state import MemChatState

    client.plugin.state = MemChatState()
    client.plugin.state.set_self_id("default:GroupMessage:42", "audit-bot")
    client.plugin.scheduler = Scheduler(
        client.plugin,
        client.plugin.context,
        client.plugin.config,
        client.plugin.state,
        None,
        client,
    )
    offered = SimpleNamespace(tools=[SimpleNamespace(name="audit_tool")])
    client._setup_tools = lambda *_args: (offered, True)
    client._call_custom_llm = AsyncMock(return_value=response(GOOD))
    event = SimpleNamespace(
        _mem_chat_group_bubble_revision=1,
        get_group_id=lambda: "42",
        get_sender_id=lambda: "system",
    )
    assert (
        await client.invoke_llm(
            "default:GroupMessage:42",
            "Already-read background; optional bubble.",
            "",
            event=event,
            task_type="read_air",
            user_msgs_to_save=[],
            impression_mode="off",
        )
        is True
    )
    assert client._call_custom_llm.await_args.kwargs["tools"] is None
    client._execute_agent_tool.assert_not_awaited()
    saved = (
        client.plugin.context.conversation_manager.update_conversation.await_args.args[
            2
        ]
    )
    assert all(item["role"] == "assistant" for item in saved)
