"""Regression tests for saving multiple memes and sending named memes."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

PLUGIN = Path(__file__).resolve().parents[1]
if str(PLUGIN) not in sys.path:
    sys.path.insert(0, str(PLUGIN))

from src.llm_client import LLMClient  # noqa: E402
from src.llm_client_parts.image_service import ImageService  # noqa: E402

from astrbot.api.message_components import Image, Reply  # noqa: E402


@pytest.mark.asyncio
async def test_capture_decision_saves_every_image_in_current_message():
    urls = [
        "https://img.example.com/one.jpg",
        "https://img.example.com/two.jpg",
        "https://img.example.com/three.jpg",
    ]
    event = SimpleNamespace(
        message_obj=SimpleNamespace(message=[Image(file=url) for url in urls])
    )
    client = LLMClient.__new__(LLMClient)
    calls: list[tuple[str, str]] = []

    async def capture(source, category):
        calls.append((source, category))
        return f"/memes/{len(calls)}.jpg"

    client._image_service = SimpleNamespace(capture_meme_image=capture)
    stored, total = await client._capture_memes_from_event(event, "color")

    assert (stored, total) == (3, 3)
    assert calls == [(url, "color") for url in urls]


@pytest.mark.asyncio
async def test_capture_decision_resolves_images_from_quoted_message(monkeypatch):
    import astrbot.core.utils.quoted_message as quoted_message

    event = SimpleNamespace(
        message_obj=SimpleNamespace(message=[Reply(id="123456")]),
        bot=SimpleNamespace(),
    )
    monkeypatch.setattr(
        quoted_message,
        "extract_quoted_message_images",
        AsyncMock(return_value=["https://img.example.com/quoted.jpg"]),
    )
    client = LLMClient.__new__(LLMClient)
    client._image_service = SimpleNamespace(
        capture_meme_image=AsyncMock(return_value="/memes/quoted.jpg")
    )

    stored, total = await client._capture_memes_from_event(event, "color")

    assert (stored, total) == (1, 1)
    client._image_service.capture_meme_image.assert_awaited_once_with(
        "https://img.example.com/quoted.jpg", "color"
    )


def test_named_meme_can_be_selected_by_literal_ocr_text(tmp_path):
    meme = tmp_path / "data/memes_data/memes/like/trust.jpg"
    meme.parent.mkdir(parents=True)
    meme.write_bytes(b"image")
    db = tmp_path / "data/memes_data/meme_index.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE meme_index(path TEXT, emotion TEXT, description TEXT, "
        "vector BLOB, dim INTEGER, model TEXT, updated_at REAL)"
    )
    con.execute(
        "CREATE TABLE meme_tags(path TEXT, emotion TEXT, intensity TEXT, "
        "stance TEXT, target TEXT, ocr_text TEXT, motion TEXT, model TEXT, "
        "updated_at REAL, energy TEXT, reply_to TEXT)"
    )
    con.execute(
        "INSERT INTO meme_index VALUES(?,?,?,?,?,?,?)",
        (
            str(meme),
            "like",
            "图上文字：信じてます（我相信你）",
            b"",
            0,
            "test",
            0,
        ),
    )
    con.commit()
    con.close()

    svc = ImageService(lambda key, default: default, get_cwd=lambda: str(tmp_path))
    svc.is_blank_image = lambda path: False
    svc._shrink_meme_for_send = lambda path: path
    selected: dict[str, str] = {}

    assert svc._literal_meme_query("信じてます") == "信じてます"
    assert svc._pick_literal_meme("信じてます", "private-a", selected) == str(meme)
    assert selected["path"] == str(meme)
    assert "信じてます" in selected["description"]


def test_meme_prompt_declares_send_and_batch_capture_capabilities(
    tmp_path, monkeypatch
):
    (tmp_path / "data/memes_data/memes/happy").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    client = LLMClient.__new__(LLMClient)
    client.plugin = SimpleNamespace(
        config={
            "read_air_compact_prompt_enabled": True,
            "scene_engine_enabled": True,
            "scene_adult_mode": True,
        },
        context=SimpleNamespace(get_registered_star=lambda name: None),
    )

    prompt = client._get_instruction_prompts(task_type="at", is_private=True)

    assert "不要只复述图中文字" in prompt
    assert "收藏后的图仍然可以发送" in prompt
    assert "系统会把当前消息里的全部图片按这个分类保存" in prompt
    assert "系统会读取被引用消息里的图片" in prompt


def test_literal_query_extracts_caption_from_explicit_send_request():
    svc = ImageService(lambda key, default: default, get_cwd=lambda: "/tmp")
    assert (
        svc._literal_meme_query("发一下信じてます，再说一下是什么意思") == "信じてます"
    )
    assert svc._literal_meme_query("把信じてます发给我") == "信じてます"


def test_explicit_saved_meme_request_gets_fallback_tag():
    client = LLMClient.__new__(LLMClient)
    client._image_service = SimpleNamespace(
        _literal_meme_query=lambda text: "信じてます",
        _find_literal_meme_candidates=lambda query: [("/meme.jpg", "desc", "")],
    )
    resp = {"text": "信じてます。"}

    client._ensure_explicit_literal_meme(
        resp, "发一下信じてます，再说一下是什么意思", "private-a"
    )

    assert resp["text"] == "信じてます。 [表情:信じてます]"


def test_existing_meme_path_is_not_duplicated_by_fallback():
    client = LLMClient.__new__(LLMClient)
    client._image_service = SimpleNamespace(
        _literal_meme_query=lambda text: "信じてます",
        _find_literal_meme_candidates=lambda query: [("/meme.jpg", "desc", "")],
    )
    resp = {"segments": [{"text": "好哦 [表情:信じてます]"}]}

    client._ensure_explicit_literal_meme(resp, "发一下信じてます", "private-a")

    assert resp["segments"][0]["text"] == "好哦 [表情:信じてます]"
