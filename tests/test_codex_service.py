import json
from types import SimpleNamespace

import pytest
from src.core.codex_service import CodexService, parse_codex_events
from src.llm_client import LLMClient

from astrbot.core.agent.tool import FunctionTool, ToolSet


class FakeEvent:
    unified_msg_origin = "aiocqhttp:FriendMessage:42"

    def __init__(self, *, sender_id="42", group_id=""):
        self._sender_id = sender_id
        self._group_id = group_id

    def get_sender_id(self):
        return self._sender_id

    def get_group_id(self):
        return self._group_id


def test_parse_codex_events_extracts_thread_and_final_message():
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "thread.started",
                    "thread_id": "01a187dd-06cd-7830-bac9-c7f23456d13f",
                }
            ),
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "first"},
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "final"},
                }
            ),
            json.dumps({"type": "turn.completed"}),
        ]
    )

    parsed = parse_codex_events(stdout)

    assert parsed["thread_id"] == "01a187dd-06cd-7830-bac9-c7f23456d13f"
    assert parsed["message"] == "final"
    assert parsed["messages"] == ["first", "final"]
    assert parsed["errors"] == []


@pytest.mark.asyncio
async def test_new_then_resume_uses_explicit_thread_and_sandbox(tmp_path):
    commands = []

    async def fake_run(command, timeout):
        commands.append(command)
        thread_id = "01a187dd-06cd-7830-bac9-c7f23456d13f"
        event = {
            "type": "thread.started",
            "thread_id": thread_id,
        }
        message = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "OK"},
        }
        return 0, json.dumps(event) + "\n" + json.dumps(message), ""

    service = CodexService(
        {
            "codex_enabled": True,
            "codex_cli_path": "codex",
            "codex_workdir": "/tmp",
            "codex_model": "deepseek-v4.1-flash",
            "codex_sandbox": "workspace-write",
            "codex_allowed_uids": ["42"],
        },
        store_path=tmp_path / "codex_sessions.json",
    )
    service._run_cli = fake_run

    created = await service.execute_tool(
        FakeEvent(), action="new", prompt="hello", session_name="demo"
    )
    resumed = await service.execute_tool(
        FakeEvent(), action="resume", prompt="follow up"
    )

    assert "thread_id: 01a187dd-06cd-7830-bac9-c7f23456d13f" in created
    assert "Codex 继续会话完成" in resumed
    assert commands[0][1] == "exec"
    assert commands[1][1:3] == ["exec", "resume"]
    assert commands[1][-2] == "01a187dd-06cd-7830-bac9-c7f23456d13f"
    assert commands[1][-1] == "follow up"
    assert "--last" not in commands[1]
    assert 'sandbox_mode="workspace-write"' in commands[1]

    stored = json.loads((tmp_path / "codex_sessions.json").read_text(encoding="utf-8"))
    scope = stored["scopes"]["aiocqhttp:FriendMessage:42"]
    assert scope["last_thread_id"] == "01a187dd-06cd-7830-bac9-c7f23456d13f"


@pytest.mark.asyncio
async def test_codex_rejects_group_call_by_default(tmp_path):
    service = CodexService(
        {"codex_enabled": True, "codex_allowed_uids": ["42"]},
        store_path=tmp_path / "codex_sessions.json",
    )

    result = await service.execute_tool(
        FakeEvent(group_id="100"), action="new", prompt="hello"
    )

    assert "owner" in result
    assert not (tmp_path / "codex_sessions.json").exists()


@pytest.mark.asyncio
async def test_codex_rejects_non_owner_private_call(tmp_path):
    service = CodexService(
        {"codex_enabled": True, "codex_allowed_uids": ["42"]},
        store_path=tmp_path / "codex_sessions.json",
    )

    result = await service.execute_tool(
        FakeEvent(sender_id="99"), action="new", prompt="hello"
    )

    assert "owner" in result


def test_codex_tool_survives_persona_and_global_tool_restrictions():
    codex_tool = FunctionTool(
        name="codex_agent",
        description="codex",
        parameters={"type": "object", "properties": {}},
    )
    other_tool = FunctionTool(
        name="other_tool",
        description="other",
        parameters={"type": "object", "properties": {}},
    )

    class ToolManager:
        def get_full_tool_set(self):
            return ToolSet(tools=[codex_tool, other_tool])

        def get_func(self, name):
            if name == "codex_agent":
                return codex_tool
            if name == "other_tool":
                return other_tool
            return None

    tm = ToolManager()
    plugin = SimpleNamespace(
        config={
            "enable_llm_tools": True,
            "codex_enabled": True,
            "llm_tools_restrict_enabled": True,
            "llm_tools_allowed_names": ["other_tool"],
        },
        context=SimpleNamespace(get_llm_tool_manager=lambda: tm),
    )
    client = LLMClient(plugin)

    tools, _ = client._setup_tools(
        "aiocqhttp:FriendMessage:42", {"allow_search": False}, {"tools": []}
    )

    assert tools is not None
    assert [tool.name for tool in tools.tools] == ["codex_agent"]


@pytest.mark.asyncio
async def test_registered_codex_tool_survives_loader_binding_and_continues_session(
    tmp_path,
):
    """Exercise registered new/resume/list calls after the loader binds the plugin."""
    import functools

    from main import MemChat

    tools = {}
    commands = []
    thread_id = "01a187dd-06cd-7830-bac9-c7f23456d13f"

    async def fake_run(command, timeout):
        commands.append(command)
        return (
            0,
            "\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": thread_id}),
                    json.dumps(
                        {
                            "type": "item.completed",
                            "item": {"type": "agent_message", "text": "OK"},
                        }
                    ),
                    json.dumps({"type": "turn.completed"}),
                ]
            ),
            "",
        )

    plugin = MemChat.__new__(MemChat)
    plugin.config = {"codex_enabled": True, "codex_allowed_uids": ["42"]}
    plugin.codex_service = CodexService(
        plugin.config, store_path=tmp_path / "sessions.json"
    )
    plugin.codex_service._run_cli = fake_run
    plugin.context = SimpleNamespace(
        add_llm_tools=lambda tool: tools.update({tool.name: tool}),
        get_llm_tool_manager=lambda: SimpleNamespace(get_func=tools.get),
    )
    plugin._register_codex_tool()
    tool = tools["codex_agent"]
    client = LLMClient(plugin)

    # Match the loader's initial binding and subsequent reload lifecycle.
    for _ in range(2):
        raw_handler = (
            tool.handler.func
            if isinstance(tool.handler, functools.partial)
            else tool.handler
        )
        tool.handler = functools.partial(raw_handler, plugin)

    created = await client._execute_agent_tool(
        "codex_agent",
        {"action": "new", "prompt": "hello"},
        FakeEvent(),
        {"codex_agent"},
    )
    resumed = await client._execute_agent_tool(
        "codex_agent",
        {"action": "resume", "prompt": "follow up"},
        FakeEvent(),
        {"codex_agent"},
    )
    listed = await client._execute_agent_tool(
        "codex_agent", {"action": "list"}, FakeEvent(), {"codex_agent"}
    )
    denied = await client._execute_agent_tool(
        "codex_agent",
        {"action": "new", "prompt": "hello"},
        FakeEvent(sender_id="99"),
        {"codex_agent"},
    )

    assert "Codex 新会话完成" in created
    assert "Codex 继续会话完成" in resumed
    assert thread_id in listed
    assert "owner" in denied
    assert len(commands) == 2
    assert commands[1][-2:] == [thread_id, "follow up"]
