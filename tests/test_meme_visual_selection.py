"""Regression tests for matching the actual visible sticker to a reply intent."""

import io
import json
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from src.llm_client import LLMClient
from src.llm_client_parts.image_service import ImageService


@pytest.fixture
def selector(tmp_path):
    service = ImageService(lambda key, default: default, get_cwd=lambda: str(tmp_path))
    paths = []
    vectors = []
    for index, score in enumerate([0.88, 0.865, 0.81]):
        path = tmp_path / f"meme-{index}.png"
        Image.new("RGB", (32, 32), (index * 80, 100, 100)).save(path)
        paths.append(str(path))
        vectors.append((str(path), [score, math.sqrt(1 - score**2)]))
    service._load_meme_vectors = lambda _: vectors
    service._load_meme_tags = lambda: {
        paths[0]: {"description": "旧描述误标为开心", "ocr_text": "旧OCR"},
    }
    service._embed_query = AsyncMock(return_value=[1.0, 0.0])
    service._shrink_meme_for_send = lambda path: path
    return service, paths


@pytest.mark.asyncio
async def test_visual_choice_overrides_best_embedding_and_old_metadata(selector):
    service, paths = selector
    visual = AsyncMock(
        return_value={
            "path": paths[1],
            "description": "闭眼笑着挥手",
            "ocr_text": "",
        }
    )
    selected = {}
    result = await service.pick_meme_image_file(
        query="看懂笑话后笑出声",
        conversation_id="group-a",
        selected_meme=selected,
        visual_selector=visual,
    )
    assert result == paths[1]
    assert selected["description"] == "闭眼笑着挥手"
    assert selected["ocr_text"] == ""
    assert visual.await_args.args[0] == "看懂笑话后笑出声"
    assert len(visual.await_args.args[1]) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", [{}, {"path": "/unoffered.png"}])
async def test_visual_abstention_or_unoffered_path_never_sends_random_meme(
    selector, choice
):
    service, _paths = selector
    assert (
        await service.pick_meme_image_file(
            query="温柔安慰朋友",
            visual_selector=AsyncMock(return_value=choice),
        )
        is None
    )


@pytest.mark.asyncio
async def test_visual_error_does_not_fall_back_to_unchecked_image(selector):
    service, _paths = selector
    assert (
        await service.pick_meme_image_file(
            query="轻松接话",
            visual_selector=AsyncMock(side_effect=TimeoutError),
        )
        is None
    )


@pytest.mark.asyncio
async def test_without_visual_review_only_near_ties_rotate(selector):
    service, paths = selector
    results = [
        await service.pick_meme_image_file(query="轻松接话", conversation_id="group-a")
        for _ in range(5)
    ]
    assert results[0] == paths[0]
    assert set(results) == set(paths[:2])


def test_animated_preview_contains_multiple_frames_and_visible_background():
    frames = [Image.new("RGB", (40, 40), color) for color in ("red", "green", "blue")]
    raw = io.BytesIO()
    frames[0].save(
        raw, format="GIF", save_all=True, append_images=frames[1:], duration=100
    )
    jpeg = ImageService._prepare_vision_jpeg(raw.getvalue())
    assert jpeg
    with Image.open(io.BytesIO(jpeg)) as montage:
        pixels = list(montage.getdata())
        assert any(r > 200 and g < 30 and b < 30 for r, g, b in pixels)
        assert any(b > 200 and r < 30 and g < 30 for r, g, b in pixels)
    raw = io.BytesIO()
    Image.new("RGBA", (30, 30), (0, 0, 0, 0)).save(raw, format="PNG")
    with Image.open(
        io.BytesIO(ImageService._prepare_vision_jpeg(raw.getvalue()))
    ) as preview:
        assert min(preview.getpixel((0, 0))) > 240


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", [0, 99, True, "1"])
async def test_invalid_or_zero_model_choice_abstains(selector, choice):
    _service, paths = selector
    client = LLMClient(SimpleNamespace(config={}))
    client._prepare_llm_config = lambda *_args: {
        "use_custom_llm": True,
        "llm": {
            "base": "https://example.invalid",
            "key": "test",
            "model": "test",
            "fmt": "openai",
        },
    }
    client._call_custom_llm = AsyncMock(
        return_value=SimpleNamespace(completion_text=json.dumps({"choice": choice}))
    )
    assert (
        await client._select_meme_by_vision("轻松接话", [{"path": paths[0]}], "group-a")
        == {}
    )


@pytest.mark.asyncio
async def test_visual_model_reads_images_and_returns_fresh_ocr(selector):
    _service, paths = selector
    client = LLMClient(SimpleNamespace(config={}))
    client._prepare_llm_config = lambda *_args: {
        "use_custom_llm": True,
        "llm": {
            "base": "https://example.invalid",
            "key": "test",
            "model": "test",
            "fmt": "openai",
        },
    }
    client._call_custom_llm = AsyncMock(
        return_value=SimpleNamespace(
            completion_text='```json\n{"choice":1,"description":"皱眉抗议","ocr_text":"别闹了"}\n```'
        )
    )
    chosen = await client._select_meme_by_vision(
        "有点生气地抗议", [{"path": paths[0], "ocr_text": "旧OCR"}], "group-a"
    )
    assert chosen == {"path": paths[0], "description": "皱眉抗议", "ocr_text": "别闹了"}
    kwargs = client._call_custom_llm.await_args.kwargs
    assert kwargs["image_urls"][0].startswith("data:image/jpeg;base64,")
    assert kwargs["fmt"] == "openai"
    assert kwargs["tools"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "emotion", ["表情:害羞地躲开视线", "[表情:害羞地躲开视线]", "表情：害羞地躲开视线"]
)
async def test_intent_in_emotion_field_does_not_include_reply_body(emotion):
    client = LLMClient(SimpleNamespace(config={}, context=SimpleNamespace()))
    client._pick_meme_image_file = AsyncMock(return_value=None)
    event = SimpleNamespace(
        message_str="你好",
        message_obj=SimpleNamespace(message=[], raw_message={}),
        get_messages=lambda: [],
        get_self_id=lambda: "bot",
        get_sender_id=lambda: "user",
        get_group_id=lambda: "group-a",
        send=AsyncMock(),
    )
    await client._process_and_send_final_reply(
        json.dumps(
            {
                "control": {"decision": "reply_only"},
                "response": {
                    "text": "这句正文不应混进选图意图",
                    "emotion": emotion,
                },
            },
            ensure_ascii=False,
        ),
        event,
        "at",
        [],
        [],
        [],
        "group-a",
        [],
    )
    client._pick_meme_image_file.assert_awaited_once()
    assert client._pick_meme_image_file.await_args.args[:2] == ("", "害羞地躲开视线")
