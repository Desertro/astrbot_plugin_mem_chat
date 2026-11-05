"""Exercise dialogue failover with real local HTTP endpoints."""

import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import web
from src.llm_client import LLMClient
from src.llm_client_parts.custom_llm import _openai_supports_image_url


@pytest_asyncio.fixture
async def route():
    calls = []
    behavior = {"primary": "ok", "fallback": "ok", "vision": "ok"}

    async def handler(request):
        body = await request.json()
        target = request.path.split("/")[1]
        calls.append((target, request.headers.get("Authorization"), body))
        action = behavior[target]
        if action == "fail":
            return web.json_response({"error": "unavailable"}, status=503)
        if action == "slow":
            await asyncio.sleep(0.1)
        if action == "tool":
            message = {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-test",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"city":"Hangzhou"}',
                        },
                    }
                ],
            }
        else:
            message = {
                "content": "" if action == "empty" else '{"answer":"' + target + '"}'
            }
        return web.json_response({"choices": [{"message": message}]})

    app = web.Application()
    app.router.add_post("/{target}/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    config = {
        "custom_llm_format": "openai",
        "llm_base_url": base + "/primary/v1",
        "llm_api_key": "primary-key",
        "llm_model": "deepseek-v4.1-flash",
        "llm_format": "openai",
        "llm_fallback": {
            "enable": True,
            "base_url": base + "/fallback/v1",
            "api_key": "fallback-key",
            "model": "deepseek-flash",
            "format": "openai",
        },
        "vision_chat_groups": ["1042128452", "72208748", "775066874"],
        "vision_chat_model": "gemini-4-argon",
        "meme_vision": {
            "enable": True,
            "base_url": base + "/vision/v1",
            "api_key": "vision-key",
            "model": "gemini-3.8-flash",
        },
    }
    client = LLMClient(
        SimpleNamespace(
            config=config,
            context=SimpleNamespace(get_using_provider=lambda **_: None),
        )
    )
    request = {
        "base_url": config["llm_base_url"],
        "api_key": config["llm_api_key"],
        "model": config["llm_model"],
        "system_prompt": "JSON only",
        "prompt": "hello",
        "contexts": [{"role": "assistant", "content": "previous"}],
        "temperature": 0.7,
        "fmt": "openai",
    }
    try:
        yield client, request, behavior, calls
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_successful_primary_does_not_spend_on_fallback(route):
    client, request, _, calls = route
    response = await client._call_custom_llm(**request)
    assert response.completion_text == '{"answer":"primary"}'
    assert [item[0] for item in calls] == ["primary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["fail", "empty"])
async def test_failed_primary_uses_original_endpoint_and_context(route, failure):
    client, request, behavior, calls = route
    behavior["primary"] = failure
    response = await client._call_custom_llm(**request)
    assert response.completion_text == '{"answer":"fallback"}'
    assert [item[0] for item in calls] == ["primary", "fallback"]
    assert calls[0][1] == "Bearer primary-key"
    assert calls[1][1] == "Bearer fallback-key"
    assert calls[0][2]["model"] == "deepseek-v4.1-flash"
    assert calls[1][2]["model"] == "deepseek-flash"
    assert calls[0][2]["messages"] == calls[1][2]["messages"]
    assert calls[1][2]["temperature"] == 0.7
    assert request["contexts"] == [{"role": "assistant", "content": "previous"}]


@pytest.mark.asyncio
async def test_primary_is_retried_on_next_request_after_recovery(route):
    client, request, behavior, calls = route
    behavior["primary"] = "fail"
    assert (
        await client._call_custom_llm(**request)
    ).completion_text == '{"answer":"fallback"}'
    behavior["primary"] = "ok"
    assert (
        await client._call_custom_llm(**request)
    ).completion_text == '{"answer":"primary"}'
    assert [item[0] for item in calls] == ["primary", "fallback", "primary"]


@pytest.mark.asyncio
async def test_native_tool_call_with_empty_content_is_valid_primary_response(route):
    client, request, behavior, calls = route
    behavior["primary"] = "tool"
    response = await client._call_custom_llm(**request)
    assert response.tools_call_name == ["lookup"]
    assert response.tools_call_args == [{"city": "Hangzhou"}]
    assert [item[0] for item in calls] == ["primary"]


@pytest.mark.asyncio
async def test_timeout_reserves_time_for_fallback(route, monkeypatch):
    client, request, behavior, calls = route
    behavior["primary"] = "slow"
    real_wait_for = asyncio.wait_for
    deadlines = []

    async def accelerated_wait_for(awaitable, timeout):
        deadlines.append(timeout)
        return await real_wait_for(awaitable, 0.02 if timeout == 35.0 else timeout)

    monkeypatch.setattr(asyncio, "wait_for", accelerated_wait_for)
    response = await client._call_custom_llm(**request)
    assert response.completion_text == '{"answer":"fallback"}'
    assert deadlines[:2] == [35.0, 20.0]
    assert [item[0] for item in calls] == ["primary", "fallback"]


@pytest.mark.asyncio
async def test_cancelled_request_does_not_start_another_endpoint(route):
    client, request, behavior, calls = route
    behavior["primary"] = "slow"
    pending = asyncio.create_task(client._call_custom_llm(**request))
    for _ in range(50):
        if calls:
            break
        await asyncio.sleep(0.002)
    assert calls
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert [item[0] for item in calls] == ["primary"]


@pytest.mark.asyncio
async def test_both_endpoints_failing_returns_failure_without_recursion(route):
    client, request, behavior, calls = route
    behavior.update(primary="fail", fallback="fail")
    assert await client._call_custom_llm(**request) is None
    assert [item[0] for item in calls] == ["primary", "fallback"]


@pytest.mark.asyncio
@pytest.mark.parametrize("group_id", ["1042128452", "72208748", "775066874"])
async def test_selected_gemini_groups_never_use_deepseek_fallback(route, group_id):
    client, request, behavior, calls = route
    behavior["vision"] = "fail"
    config = client._prepare_llm_config(f"default:GroupMessage:{group_id}", None)["llm"]
    request.update(
        base_url=config["base"], api_key=config["key"], model=config["model"]
    )
    assert request["model"] == "gemini-4-argon"
    assert await client._call_custom_llm(**request) is None
    assert [item[0] for item in calls] == ["vision"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    ["disabled", "missing_key", "same_endpoint", "different_model", "different_key"],
)
async def test_fallback_requires_complete_enabled_and_matching_route(route, change):
    client, request, behavior, calls = route
    config = client.plugin.config
    behavior["primary"] = "fail"
    if change == "disabled":
        config["llm_fallback"]["enable"] = False
    elif change == "missing_key":
        config["llm_fallback"]["api_key"] = ""
    elif change == "same_endpoint":
        config["llm_fallback"].update(
            base_url=request["base_url"],
            api_key=request["api_key"],
            model=request["model"],
        )
    elif change == "different_model":
        request["model"] = "auxiliary-model"
    else:
        request["api_key"] = "other-key"
    assert await client._call_custom_llm(**request) is None
    assert [item[0] for item in calls] == ["primary"]


@pytest.mark.parametrize(
    "model",
    [
        "deepseek-flash",
        "deepseek-v4.1-flash",
        "deepseek-v4.1-flash-inhouse",
        "deepseek-v4.1-flash-1017",
    ],
)
def test_flash_aliases_keep_image_input(model):
    assert _openai_supports_image_url("http://agic.example/v1", model)
    assert not _openai_supports_image_url("http://agic.example/v1", "deepseek-chat")
