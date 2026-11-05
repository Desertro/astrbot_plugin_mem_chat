from types import SimpleNamespace

from src.llm_client import LLMClient

from astrbot.core.agent.tool import FunctionTool, ToolSet


def test_plugin_agent_survives_tool_restrictions():
    plugin_agent = FunctionTool(
        name="plugin_agent",
        description="plugin manager",
        parameters={"type": "object", "properties": {}},
    )
    other_tool = FunctionTool(
        name="other_tool",
        description="other",
        parameters={"type": "object", "properties": {}},
    )

    class ToolManager:
        def get_full_tool_set(self):
            return ToolSet(tools=[other_tool])

        def get_func(self, name):
            if name == "plugin_agent":
                return plugin_agent
            if name == "other_tool":
                return other_tool
            return None

    plugin = SimpleNamespace(
        config={
            "enable_llm_tools": True,
            "codex_enabled": False,
            "llm_tools_restrict_enabled": True,
            "llm_tools_allowed_names": ["other_tool"],
        },
        context=SimpleNamespace(get_llm_tool_manager=lambda: ToolManager()),
    )
    client = LLMClient(plugin)

    tools, _ = client._setup_tools(
        "aiocqhttp:FriendMessage:42", {"allow_search": False}, {"tools": []}
    )

    assert tools is not None
    assert "plugin_agent" in [tool.name for tool in tools.tools]
    assert "other_tool" not in [tool.name for tool in tools.tools]


def test_plugin_agent_omitted_for_native_shoubanhua_command():
    plugin_agent = FunctionTool(
        name="plugin_agent",
        description="plugin manager",
        parameters={"type": "object", "properties": {}},
    )

    class ToolManager:
        def get_full_tool_set(self):
            # Start with plugin_agent present to cover personas/full tool sets,
            # not just the explicit re-add path.
            return ToolSet(tools=[plugin_agent])

        def get_func(self, name):
            return plugin_agent if name == "plugin_agent" else None

    plugin = SimpleNamespace(
        config={"enable_llm_tools": True, "codex_enabled": False},
        context=SimpleNamespace(get_llm_tool_manager=lambda: ToolManager()),
    )
    client = LLMClient(plugin)
    native_handler = SimpleNamespace(
        handler_module_path="data.plugins.astrbot_plugin_shoubanhua.main",
        handler_name="on_bnn",
    )
    event = SimpleNamespace(
        get_extra=lambda key, default=None: (
            [native_handler] if key == "activated_handlers" else default
        )
    )

    tools, _ = client._setup_tools(
        "aiocqhttp:FriendMessage:42",
        {"allow_search": False},
        {"tools": []},
        event=event,
    )

    assert tools is not None
    assert "plugin_agent" not in [tool.name for tool in tools.tools]
