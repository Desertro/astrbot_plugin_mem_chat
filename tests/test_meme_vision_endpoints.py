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
