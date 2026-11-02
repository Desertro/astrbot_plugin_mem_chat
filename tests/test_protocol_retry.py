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
