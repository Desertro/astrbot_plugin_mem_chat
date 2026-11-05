"""Exercise real HTTP routing for dedicated sticker recognition."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiohttp import web
from PIL import Image
from src.llm_client import LLMClient
from src.llm_client_parts.image_service import ImageService


@pytest_asyncio.fixture
async def endpoints():
    calls = []

    async def handler(request):
        body = await request.json()
        calls.append((request.path, request.headers.get("Authorization"), body))
        if request.path.startswith("/fail"):
            return web.json_response({"error": "unavailable"}, status=503)
        return web.json_response(
            {"choices": [{"message": {"content": "文字：我顶死你"}}]}
        )

    app = web.Application()
    app.router.add_post("/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    base = "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])
    try:
        yield base, calls
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_dedicated_model_receives_image_and_never_uses_embedding_key(endpoints):
    base, calls = endpoints
    conf = {
        "meme_vision": {
            "enable": True,
            "base_url": base + "/v1",
            "api_key": "vision-only",
            "model": "gemini-3.8-flash",
        },
        "embedding_openai_api_key": "embedding-only",
    }
    svc = ImageService(conf.get)
    assert (
        await svc.call_meme_vision(["data:image/png;base64,test"], "识图")
        == "文字：我顶死你"
    )
    assert len(calls) == 1
    assert calls[0][0] == "/v1/chat/completions"
    assert calls[0][1] == "Bearer vision-only"
    assert calls[0][2]["model"] == "gemini-3.8-flash"
    assert calls[0][2]["messages"][0]["content"][1]["image_url"]["url"].endswith("test")


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,expected", [(True, 2), (False, 1)])
async def test_backup_requires_explicit_enable_and_uses_its_own_key(
    endpoints, enabled, expected
):
    base, calls = endpoints
    conf = {
        "enable": True,
        "base_url": base + "/fail",
        "api_key": "primary",
        "model": "gemini-3.8-flash",
        "fallback_enabled": enabled,
        "fallback_base_url": base + "/backup",
        "fallback_api_key": "backup",
        "fallback_model": "backup-model",
    }
    svc = ImageService({"meme_vision": conf}.get)
    result = await svc.call_meme_vision(["image"], "识图")
    assert bool(result) is enabled
    assert len(calls) == expected
    if enabled:
        assert calls[-1][1] == "Bearer backup"
        assert calls[-1][2]["model"] == "backup-model"


@pytest.mark.asyncio
async def test_context_caption_uses_dedicated_model_without_qwen():
    svc = ImageService({"meme_vision": {"enable": True}}.get)
    svc.call_meme_vision = AsyncMock(return_value="文字：我顶死你")
    assert await svc._describe_vision_jpeg(b"jpeg", prompt="read") == "文字：我顶死你"
    assert svc.call_meme_vision.await_args.args[0] == [
        "data:image/jpeg;base64,anBlZw=="
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply,selected",
    [
        (None, False),
        (
            json.dumps({"choice": 1, "description": "笑脸", "ocr_text": "我顶死你"}),
            True,
        ),
    ],
)
async def test_sticker_selection_uses_dedicated_vision_and_does_not_fall_back_to_dialogue(
    tmp_path, reply, selected
):
    image = tmp_path / "sticker.png"
    Image.new("RGB", (40, 40), "blue").save(image)
    client = LLMClient(SimpleNamespace(config={"meme_vision": {"enable": True}}))
    client._prepare_llm_config = lambda *_: pytest.fail(
        "Dialogue model must not be called"
    )
    client._image_service.call_meme_vision = AsyncMock(return_value=reply)
    result = await client._select_meme_by_vision(
        "笑着回应", [{"path": str(image)}], "private"
    )
    assert bool(result) is selected
    if selected:
        assert result["ocr_text"] == "我顶死你"


@pytest.mark.asyncio
async def test_human_reviewed_caption_survives_model_misread(tmp_path):
    image = tmp_path / "angry.png"
    Image.new("RGB", (40, 40), "red").save(image)
    client = LLMClient(SimpleNamespace(config={"meme_vision": {"enable": True}}))
    client._image_service.call_meme_vision = AsyncMock(
        return_value=json.dumps(
            {"choice": 1, "description": "卖萌爱心", "ocr_text": "Frame 1"}
        )
    )
    result = await client._select_meme_by_vision(
        "表达不满",
        [
            {
                "path": str(image),
                "description": "压眉生气，旁边冒气团",
                "ocr_text": "",
                "review_model": "codex-direct-visual-review-20261030",
            }
        ],
        "private",
    )
    assert result["description"] == "压眉生气，旁边冒气团"
    assert result["ocr_text"] == ""
    prompt = json.loads(client._image_service.call_meme_vision.await_args.args[1])
    assert (
        prompt["candidates"][0]["human_reviewed_description"] == result["description"]
    )


@pytest.mark.parametrize(
    "umo,expected",
    [
        ("default:GroupMessage:1042128452", "gemini-3.8-flash"),
        ("aiocqhttp:GroupMessage:1042128452", "gemini-3.8-flash"),
        ("default:GroupMessage:72208748", "deepseek-flash"),
        ("default:GroupMessage:1080344345", "deepseek-flash"),
        ("default:FriendMessage:1042128452", "deepseek-flash"),
        ("default:GroupMessage:10421284520", "deepseek-flash"),
    ],
)
def test_vision_chat_route_isolated_to_exact_group(umo, expected):
    config = {
        "llm_base_url": "https://dialogue.example/v1",
        "llm_api_key": "dialogue-only",
        "llm_model": "deepseek-flash",
        "llm_format": "openai",
        "vision_chat_groups": ["1042128452"],
        "meme_vision": {
            "enable": True,
            "base_url": "https://vision.example/v1",
            "api_key": "vision-only",
            "model": "gemini-3.8-flash",
        },
    }
    context = SimpleNamespace(get_using_provider=lambda **_: None)
    client = LLMClient(SimpleNamespace(config=config, context=context))
    result = client._prepare_llm_config(umo, None)
    assert result["llm"]["model"] == expected
    assert result["llm"]["fmt"] == "openai"
    assert result["llm"]["key"] == (
        "vision-only" if expected.startswith("gemini") else "dialogue-only"
    )
    assert config["llm_model"] == "deepseek-flash"
    assert config["llm_api_key"] == "dialogue-only"


@pytest.mark.parametrize(
    "vision", [{"enable": False}, {"enable": True, "model": "gemini-3.8-flash"}]
)
def test_unavailable_vision_chat_keeps_complete_default_endpoint(vision):
    context = SimpleNamespace(get_using_provider=lambda **_: None)
    config = {
        "llm_base_url": "https://dialogue.example/v1",
        "llm_api_key": "dialogue-only",
        "llm_model": "deepseek-flash",
        "vision_chat_groups": ["1042128452"],
        "meme_vision": vision,
    }
    client = LLMClient(SimpleNamespace(config=config, context=context))
    result = client._prepare_llm_config("default:GroupMessage:1042128452", None)
    assert result["llm"]["base"] == "https://dialogue.example/v1"
    assert result["llm"]["key"] == "dialogue-only"
    assert result["llm"]["model"] == "deepseek-flash"


@pytest.mark.asyncio
async def test_vision_chat_transmits_real_image_to_selected_endpoint(
    endpoints, tmp_path
):
    base, calls = endpoints
    image = tmp_path / "native-input.png"
    Image.new("RGB", (80, 40), "blue").save(image)
    config = {
        "llm_base_url": base + "/wrong-endpoint",
        "llm_api_key": "dialogue-only",
        "llm_model": "deepseek-flash",
        "custom_llm_format": "google",
        "vision_chat_groups": ["1042128452"],
        "meme_vision": {
            "enable": True,
            "base_url": base,
            "api_key": "vision-only",
            "model": "gemini-3.8-flash",
        },
    }
    context = SimpleNamespace(get_using_provider=lambda **_: None)
    client = LLMClient(SimpleNamespace(config=config, context=context))
    cfg = client._prepare_llm_config("default:GroupMessage:1042128452", None)["llm"]
    response = await client._call_custom_llm(
        base_url=cfg["base"],
        api_key=cfg["key"],
        model=cfg["model"],
        system_prompt="Read the actual image.",
        prompt="What color is the image?",
        contexts=[{"role": "assistant", "content": "Prior conversation"}],
        image_urls=[str(image)],
        fmt=cfg["fmt"],
    )
    assert response.completion_text
    assert len(calls) == 1
    path, authorization, body = calls[0]
    assert path == "/v1/chat/completions"
    assert authorization == "Bearer vision-only"
    assert body["model"] == "gemini-3.8-flash"
    assert body["messages"][1] == {"role": "assistant", "content": "Prior conversation"}
    inputs = body["messages"][-1]["content"]
    image_inputs = [part for part in inputs if part.get("type") == "image_url"]
    assert len(image_inputs) == 1
    assert image_inputs[0]["image_url"]["url"].startswith("data:image/")


@pytest.mark.parametrize(
    "umo,expected",
    [
        ("default:GroupMessage:1042128452", "gemini-4-argon"),
        ("default:GroupMessage:72208748", "gemini-4-argon"),
        ("default:GroupMessage:775066874", "gemini-4-argon"),
        ("default:GroupMessage:1080344345", "deepseek-flash"),
        ("default:FriendMessage:775066874", "deepseek-flash"),
    ],
)
def test_vision_chat_model_override_keeps_caption_and_default_models(umo, expected):
    config = {
        "llm_base_url": "https://dialogue.example/v1",
        "llm_api_key": "dialogue-only",
        "llm_model": "deepseek-flash",
        "vision_chat_groups": ["1042128452", "72208748", "775066874"],
        "vision_chat_model": "gemini-4-argon",
        "meme_vision": {
            "enable": True,
            "base_url": "https://vision.example/v1",
            "api_key": "vision-only",
            "model": "gemini-3.8-flash",
        },
    }
    context = SimpleNamespace(get_using_provider=lambda **_: None)
    client = LLMClient(SimpleNamespace(config=config, context=context))
    result = client._prepare_llm_config(umo, None)
    assert result["llm"]["model"] == expected
    assert result["llm"]["key"] == (
        "vision-only" if expected == "gemini-4-argon" else "dialogue-only"
    )
    assert config["meme_vision"]["model"] == "gemini-3.8-flash"
    assert config["llm_model"] == "deepseek-flash"
