"""Regression coverage for semantic meme selection and memory scoring."""

import asyncio
import importlib.util
import math
import sqlite3
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

PLUGIN = Path(__file__).resolve().parents[1]


def load_module(name, path):
    """Load a standalone plugin module without importing the plugin entry point.

    Args:
        name: Isolated module name.
        path: Source file to load.

    Returns:
        Imported module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


image_module = load_module(
    "review_image_service", PLUGIN / "src/llm_client_parts/image_service.py"
)
db_module = load_module("review_database", PLUGIN / "src/db/database.py")


@pytest.fixture
def service(tmp_path, monkeypatch):
    """Create a selector with a deterministic offline embedding provider.

    Args:
        tmp_path: Temporary library root.
        monkeypatch: Scoped patch manager.

    Returns:
        Selector, library builder, and tag dictionary.
    """
    svc = image_module.ImageService(
        lambda key, default: default, get_cwd=lambda: str(tmp_path)
    )
    svc._embed_query = AsyncMock(return_value=[1.0, 0.0])
    svc._shrink_meme_for_send = lambda path: path
    svc.is_blank_image = lambda path: False
    tags = {}
    svc._load_meme_tags = lambda: tags
    monkeypatch.setattr(image_module.random, "choice", lambda items: items[0])

    def library(scores):
        vectors = []
        for idx, score in enumerate(scores):
            path = tmp_path / f"{idx:02d}.png"
            path.write_bytes(b"image")
            vectors.append((str(path), [score, math.sqrt(1.0 - score * score)]))
        svc._load_meme_vectors = lambda *_args: vectors
        return [item[0] for item in vectors]

    return svc, library, tags


@pytest.mark.asyncio
async def test_repeated_intent_rotates_relevant_images(service):
    svc, library, _tags = service
    paths = library([0.8] * 20)
    selected = [
        await svc.pick_meme_image_file(query="同一意图", conversation_id="group-a")
        for _ in range(20)
    ]
    assert len(set(selected)) == 20
    assert set(selected) == set(paths)


@pytest.mark.asyncio
async def test_recent_images_are_scoped_to_conversation(service):
    svc, library, _tags = service
    paths = library([0.8] * 4)
    first = await svc.pick_meme_image_file(query="同一意图", conversation_id="group-a")
    second = await svc.pick_meme_image_file(query="同一意图", conversation_id="group-a")
    other = await svc.pick_meme_image_file(query="同一意图", conversation_id="group-b")
    assert first != second
    assert other == first
    assert first in paths


@pytest.mark.asyncio
async def test_small_pool_reuses_oldest_without_unrelated_images(service):
    svc, library, _tags = service
    paths = library([0.8, 0.79, 0.2, 0.1])
    selected = [
        await svc.pick_meme_image_file(query="想吐槽", conversation_id="group-a")
        for _ in range(8)
    ]
    assert set(selected) == set(paths[:2])
    assert all(a != b for a, b in zip(selected, selected[1:]))


@pytest.mark.asyncio
async def test_empty_reply_type_does_not_reward_missing_tags(service):
    svc, library, tags = service
    paths = library([0.8, 0.72])
    tags[paths[0]] = {"reply_to": "自言自语"}
    tags[paths[1]] = {"reply_to": ""}
    assert await svc.pick_meme_image_file(query="普通回应") == paths[0]


@pytest.mark.asyncio
async def test_noisy_multi_emotion_labels_do_not_override_relevance(service):
    svc, library, tags = service
    paths = library([0.8, 0.65])
    tags[paths[1]] = {"emotion": "喜欢、撒娇", "reply_to": ""}
    assert await svc.pick_meme_image_file(query="可爱地撒娇") == paths[0]


@pytest.mark.asyncio
async def test_blank_candidate_does_not_cancel_valid_meme(service):
    svc, library, _tags = service
    paths = library([0.8, 0.79])
    svc.is_blank_image = lambda path: path == paths[0]
    assert await svc.pick_meme_image_file(query="回应") == paths[1]


@pytest.mark.asyncio
async def test_explicit_image_text_is_not_rotated_away(service):
    svc, library, tags = service
    paths = library([0.9, 0.65])
    tags[paths[1]] = {"ocr_text": "明天再说吧"}
    selected = [
        await svc.pick_meme_image_file(query="发一张写着明天再说吧的表情")
        for _ in range(3)
    ]
    assert selected == [paths[1]] * 3


def test_vector_cache_refreshes_when_only_wal_changes(tmp_path):
    db = tmp_path / "data/memes_data/meme_index.db"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        "CREATE TABLE meme_index(path TEXT, vector BLOB, dim INTEGER, model TEXT)"
    )
    con.commit()
    svc = image_module.ImageService(
        lambda key, default: (
            "test-model" if key == "embedding_openai_model" else default
        ),
        get_cwd=lambda: str(tmp_path),
    )
    con.execute(
        "INSERT INTO meme_index VALUES(?,?,?,?)",
        ("a.png", struct.pack("<2f", 1, 0), 2, "test-model"),
    )
    con.commit()
    assert len(svc._load_meme_vectors()) == 1
    db_mtime = db.stat().st_mtime_ns
    con.execute(
        "INSERT INTO meme_index VALUES(?,?,?,?)",
        ("b.png", struct.pack("<2f", 0, 1), 2, "test-model"),
    )
    con.commit()
    assert db.stat().st_mtime_ns == db_mtime
    assert len(svc._load_meme_vectors()) == 2
    con.close()


@pytest.mark.asyncio
async def test_cached_user_memory_accepts_score_without_reembedding(tmp_path):
    db = db_module.DatabaseManager(
        str(tmp_path / "memory.db"), {"embedding_openai_model": "test-model"}
    )
    db.get_embedding = AsyncMock(return_value=[1.0, 0.0])
    try:
        await db.ensure_message_embedded("我爸住院了", "group-a", "用户/123")
        await db.ensure_message_embedded(
            "我爸住院了", "group-a", "用户/123", importance=0.9
        )
        await db.ensure_message_embedded("我爸住院了", "group-a", "用户/123")
        rows = db._db_conn.execute(
            "SELECT content, importance FROM memory_embeddings"
        ).fetchall()
        assert rows == [("用户/123: 我爸住院了", 0.9)]
        db.get_embedding.assert_awaited_once()
    finally:
        db._db_conn.close()


@pytest.mark.asyncio
async def test_reply_score_updates_pending_input_when_impressions_disabled(tmp_path):
    sys.path.insert(0, str(PLUGIN))
    try:
        from src.llm_client import LLMClient
    finally:
        sys.path.pop(0)
    db = db_module.DatabaseManager(
        str(tmp_path / "memory.db"), {"embedding_openai_model": "test-model"}
    )
    db.get_embedding = AsyncMock(return_value=[1.0, 0.0])
    ready = asyncio.Event()
    inbound = ("我爸住院了", "group-a", "用户/123")

    async def initial_write():
        await ready.wait()
        await db.ensure_message_embedded(*inbound)

    initial = asyncio.create_task(initial_write())
    event = SimpleNamespace(
        _mem_chat_memory_input=inbound, _mem_chat_memory_task=initial
    )
    client = LLMClient(
        SimpleNamespace(config={"impression_stream_enabled": False}, db_mgr=db)
    )
    try:
        await client._persist_impression_updates(
            umo="group-a",
            event=event,
            task_type="at",
            impression_mode="",
            payload={"memory_importance": 9},
        )
        assert not initial.done()
        ready.set()
        await event._mem_chat_importance_task
        assert (
            db._db_conn.execute("SELECT importance FROM memory_embeddings").fetchone()[
                0
            ]
            == 0.9
        )
        db.get_embedding.assert_awaited_once()
    finally:
        initial.cancel()
        db._db_conn.close()


def test_timestamped_history_preserves_authorship_and_current_prompt():
    sys.path.insert(0, str(PLUGIN))
    try:
        from src.llm_client import LLMClient
    finally:
        sys.path.pop(0)
    client = LLMClient(SimpleNamespace(config={}))
    memory = (
        client._HISTORY_BLOCK_HEADER
        + "\n[19:40] arisbot/1000000000（管理员·LV100）: 我刚刚答应过你 [表情:想吐槽又忍不住想笑]"
        + "\n[19:41] 群友/11000000000（LV20）: 你答应了什么？"
        + "\n【人物印象】\n不要丢掉这个区块"
    )
    remaining, messages = client._split_history_into_contexts(memory, "1000000000")
    assert "不要丢掉这个区块" in remaining
    assert [row["role"] for row in messages] == ["assistant", "user"]
    assert messages[0]["content"].startswith("我刚刚答应过你")
    assert "当时配图：想吐槽又忍不住想笑" in messages[0]["content"]
    assert "[表情:" not in messages[0]["content"]
    assert "19:41" in messages[1]["content"]


def test_legacy_history_without_id_is_kept_without_guessing_author():
    sys.path.insert(0, str(PLUGIN))
    try:
        from src.llm_client import LLMClient
    finally:
        sys.path.pop(0)
    client = LLMClient(SimpleNamespace(config={}))
    memory = client._HISTORY_BLOCK_HEADER + "\n[REF:3] [08:30] 旧昵称：还记得吗"
    _, messages = client._split_history_into_contexts(memory, "1000000000")
    assert messages == [{"role": "user", "content": "[REF:3] [08:30] 旧昵称：还记得吗"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("emotion", ["", "test"])
async def test_selected_meme_metadata_keeps_original_after_resize(service, emotion):
    """Keep actual OCR and description even when the send path is temporary.

    Args:
        service: Offline selector fixture.
        emotion: Global or directory selection mode.
    """
    svc, add, tags = service
    paths = add([1.0])
    original = paths[0]
    if emotion:
        path = Path(svc._get_cwd()) / "data/memes_data/memes" / emotion / "sticker.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"image")
        original = str(path)
        svc._load_meme_vectors = lambda _: [(original, [1.0, 0.0])]
    tags[original] = {"ocr_text": "送你我穿过的袜子", "description": "角色举起袜子"}
    svc._shrink_meme_for_send = lambda _: "/tmp/meme_send_resized.jpg"
    selected = {"ocr_text": "stale"}
    ready = await svc.pick_meme_image_file(
        emotion, "想吐槽对方", selected_meme=selected
    )
    assert ready == "/tmp/meme_send_resized.jpg"
    assert selected == {"path": original, **tags[original]}
    svc._load_meme_vectors = lambda _: []
    assert (
        await svc.pick_meme_image_file(query="no match", selected_meme=selected) is None
    )
    assert selected == {}
