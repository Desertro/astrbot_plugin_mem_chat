"""Focused tests for the private scene coordinator and persistence."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

PLUGIN = Path(__file__).resolve().parents[1]
if str(PLUGIN) not in sys.path:
    sys.path.insert(0, str(PLUGIN))

from src.db.database import DatabaseManager
from src.llm.response_schemas import StructuredResponse
from src.scene.engine import SceneEngine


def make_engine(config=None, db=None):
    """Build a SceneEngine with only the collaborators needed by unit tests."""
    cfg = {
        "scene_engine_enabled": True,
        "scene_adult_mode": True,
        "scene_min_interval_sec": 4,
        "scene_max_interval_sec": 4,
        "scene_after_reply_delay_sec": 2,
        "scene_continuous_interval_sec": 4,
        "scene_stop_keywords": ["停下", "结束", "不要了"],
    }
    cfg.update(config or {})
    plugin = SimpleNamespace(
        state=SimpleNamespace(
            set_interrupt=lambda _umo: None,
            set_processing=lambda _umo, _value: None,
            get_self_id=lambda _umo: "42",
        ),
        context=SimpleNamespace(),
    )
    return SceneEngine(plugin, cfg, db or SimpleNamespace(), None, None)


def test_adult_mode_is_content_ceiling_not_engine_switch():
    assert make_engine({"scene_adult_mode": False}).is_enabled()
    assert not make_engine({"scene_engine_enabled": False}).is_enabled()


def test_scene_payload_stop_and_private_detection():
    engine = make_engine()
    assert engine.is_private_umo("aiocqhttp:friend:123")
    assert not engine.is_private_umo("aiocqhttp:group:123")
    assert engine.is_stop_text("停下")
    assert engine.is_stop_text("结束吧")
    assert not engine.is_stop_text("别停")
    assert not engine.is_stop_text("不要停，继续")
    assert not engine.is_stop_text("继续")


def test_continuous_request_is_narrow_and_parses_duration():
    engine = make_engine()
    assert engine._continuous_request("爱丽丝来连着发五分钟") == 300
    assert engine._continuous_request("连续发 10 分钟，不要停") == engine._max_duration_sec()
    assert engine._continuous_request("一直发") == engine._max_duration_sec()
    assert engine._continuous_request("今天天气不错") is None
    session = {"continuous": True, "continuous_until": time.time() + 60}
    assert engine._max_messages_for(session) >= 160


def test_scene_payload_is_normalized_and_preserved():
    engine = make_engine()
    scene = engine.normalize_scene_payload(
        {
            "active": True,
            "start": True,
            "phase": "unknown",
            "intensity": 99,
            "next_tick_sec": 999,
        }
    )
    assert scene is not None
    assert scene["active"] is True
    assert scene["start"] is True
    assert scene["phase"] == "start"
    assert scene["intensity"] == 10.0
    self_play = engine.normalize_scene_payload(
        {"active": True, "phase": "self_play", "intensity": 8}
    )
    assert self_play is not None
    assert self_play["phase"] == "self_play"

    payload = StructuredResponse.normalize_response_dict(
        {
            "control": {"decision": "reply_only"},
            "response": {"text": "ok"},
            "scene": {"active": True, "start": True},
        }
    )
    assert payload["scene"]["active"] is True


def test_self_play_does_not_treat_silence_as_stop():
    engine = make_engine()
    now = time.time()
    prompt = engine._build_autonomous_prompt(
        {
            "phase": "self_play",
            "intensity": 7,
            "started_at": now - 300,
            "deadline_at": now + 600,
            "tick_count": 3,
            "sent_count": 8,
            "last_user_at": now - 120,
            "last_user_text": "一直发，我不回复",
        }
    )
    assert "静默不是停止信号" in prompt
    assert "不要仅因对方暂时没说话就机械收束" in prompt


@pytest.mark.asyncio
async def test_scene_session_db_roundtrip(tmp_path):
    db = DatabaseManager(str(tmp_path / "scene.db"), {})
    try:
        session = {
            "umo": "aiocqhttp:friend:123",
            "status": "active",
            "phase": "rising",
            "intensity": 4.5,
            "started_at": time.time(),
            "updated_at": time.time(),
            "deadline_at": time.time() + 300,
            "next_tick_at": time.time() + 20,
            "tick_count": 1,
            "sent_count": 2,
        }
        assert await db.save_scene_session(session)
        got = await db.get_scene_session(session["umo"])
        assert got is not None
        assert got["phase"] == "rising"
        assert got["sent_count"] == 2

        rows = await db.list_scene_sessions(status="active")
        assert len(rows) == 1
        assert rows[0]["umo"] == session["umo"]

        session["status"] = "done"
        session["updated_at"] = time.time()
        assert await db.save_scene_session(session)
        assert await db.list_scene_sessions(status="active") == []
    finally:
        db._db_conn.close()


@pytest.mark.asyncio
async def test_initial_scene_payload_starts_a_persistent_session():
    db = SimpleNamespace(
        save_scene_session=AsyncMock(return_value=True),
        get_scene_session=AsyncMock(return_value=None),
        list_scene_sessions=AsyncMock(return_value=[]),
    )
    engine = make_engine(db=db)
    event = SimpleNamespace(
        get_sender_id=lambda: "123",
        message_str="继续描述，别停",
        message_obj=SimpleNamespace(message_id="m1"),
    )
    scene = {
        "active": True,
        "start": True,
        "phase": "start",
        "intensity": 2,
        "next_tick_sec": 4,
    }
    try:
        session = await engine.start_from_scene_payload(
            "aiocqhttp:friend:123", event, scene, user_text=event.message_str
        )
        assert session is not None
        assert session["status"] == "active"
        assert session["deadline_at"] > time.time()
        assert db.save_scene_session.await_count >= 1
    finally:
        engine.shutdown()


@pytest.mark.asyncio
async def test_scene_payload_after_normal_reply_starts_session():
    engine = make_engine()
    engine._load_active_session = AsyncMock(return_value=None)
    engine.start_from_scene_payload = AsyncMock(return_value={"status": "active"})
    event = SimpleNamespace(
        _mem_chat_scene_payload={
            "active": True,
            "start": True,
            "phase": "start",
            "intensity": 1,
        },
        _mem_chat_last_send_ok=True,
        _mem_chat_last_structured_payload={
            "response": {"segments": [{"text": "一"}, {"text": "二"}, {"text": "三"}]}
        },
        message_str="连着发五分钟",
        get_sender_id=lambda: "123",
        message_obj=SimpleNamespace(message_id="m2"),
    )
    await engine.on_scheduler_reply_completed(
        "aiocqhttp:friend:123", event, "at", sent=True, replied=True
    )
    engine.start_from_scene_payload.assert_awaited_once()
    kwargs = engine.start_from_scene_payload.await_args.kwargs
    assert kwargs["initial_sent_count"] == 3
    assert kwargs["user_text"] == "连着发五分钟"

def test_scheduler_notifies_scene_after_private_at_turn():
    """AT turns must hand their scene payload to the autonomous coordinator."""
    source = (PLUGIN / "src/core/scheduler.py").read_text(encoding="utf-8")
    at_start = source.index('if task.task_type == "at":')
    focus_start = source.index(
        'elif task.task_type == "focus" or task.task_type == "read_air":',
        at_start,
    )
    at_block = source[at_start:focus_start]
    assert 'self._notify_scene_after_turn(' in at_block
    assert '"at"' in at_block


@pytest.mark.asyncio
async def test_scene_tick_sends_and_honors_end_flag():
    db = SimpleNamespace(save_scene_session=AsyncMock(return_value=True))
    engine = make_engine(db=db)
    engine.scheduler = SimpleNamespace(
        build_memory_context=AsyncMock(return_value=("", ""))
    )

    async def fake_invoke(**kwargs):
        event = kwargs["event"]
        event._mem_chat_last_send_ok = True
        event._mem_chat_scene_payload = {
            "active": True,
            "start": False,
            "phase": "finish",
            "intensity": 7,
            "end": True,
        }
        return True

    engine.llm_client = SimpleNamespace(invoke_llm=AsyncMock(side_effect=fake_invoke))
    umo = "aiocqhttp:friend:123"
    now = time.time()
    engine._sessions[umo] = {
        "umo": umo,
        "status": "active",
        "phase": "plateau",
        "intensity": 6,
        "started_at": now,
        "updated_at": now,
        "deadline_at": now + 60,
        "next_tick_at": now,
        "tick_count": 0,
        "sent_count": 0,
        "last_user_at": now,
        "last_bot_at": 0,
        "scope_id": "123",
        "last_user_text": "继续",
    }
    engine._generations[umo] = 0
    await engine._run_tick(umo, 0)
    saved = db.save_scene_session.await_args_list[-1].args[0]
    assert saved["status"] == "done"
    assert saved["close_reason"] == "scene_end"


@pytest.mark.asyncio
async def test_scene_tick_schedules_successor_without_self_cancel():
    """An autonomous tick must survive scheduling its own successor."""
    db = SimpleNamespace(save_scene_session=AsyncMock(return_value=True))
    engine = make_engine(db=db)
    engine.scheduler = SimpleNamespace(
        build_memory_context=AsyncMock(return_value=("", ""))
    )

    async def fake_invoke(**kwargs):
        event = kwargs["event"]
        event._mem_chat_last_send_ok = True
        event._mem_chat_last_structured_payload = {
            "response": {"segments": [{"text": "嗯……"}, {"text": "继续。"}]}
        }
        event._mem_chat_scene_payload = {
            "active": True,
            "start": False,
            "phase": "rising",
            "intensity": 5,
            "end": False,
            "next_tick_sec": 4,
        }
        return True

    engine.llm_client = SimpleNamespace(invoke_llm=AsyncMock(side_effect=fake_invoke))
    umo = "aiocqhttp:friend:123"
    now = time.time()
    engine._sessions[umo] = {
        "umo": umo,
        "status": "active",
        "phase": "start",
        "intensity": 3,
        "started_at": now,
        "updated_at": now,
        "deadline_at": now + 300,
        "next_tick_at": now,
        "tick_count": 0,
        "sent_count": 0,
        "last_user_at": now,
        "last_bot_at": 0,
        "scope_id": "123",
        "last_user_text": "继续",
    }
    engine._generations[umo] = 0

    tick_task = asyncio.create_task(engine._run_tick(umo, 0))
    engine._tasks[umo] = tick_task
    try:
        await tick_task
        assert not tick_task.cancelled()
        session = engine._sessions[umo]
        assert session["status"] == "active"
        assert session["tick_count"] == 1
        assert session["sent_count"] == 2
        successor = engine._tasks.get(umo)
        assert successor is not None
        assert successor is not tick_task
        assert not successor.done()
    finally:
        engine.shutdown()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_initial_continuous_scene_counts_first_reply_and_uses_short_cadence():
    db = SimpleNamespace(save_scene_session=AsyncMock(return_value=True))
    engine = make_engine(db=db)
    event = SimpleNamespace(
        _mem_chat_last_send_ok=True,
        _mem_chat_last_structured_payload={
            "response": {"segments": [{"text": "一"}, {"text": "二"}, {"text": "三"}]}
        },
        get_sender_id=lambda: "123",
        message_str="连着发五分钟",
        message_obj=SimpleNamespace(message_id="m1"),
    )
    scene = {
        "active": True,
        "start": True,
        "phase": "start",
        "intensity": 2,
        "next_tick_sec": 90,
    }
    try:
        session = await engine.start_from_scene_payload(
            "aiocqhttp:friend:123",
            event,
            scene,
            user_text="连着发五分钟",
            initial_sent_count=3,
        )
        assert session is not None
        assert session["phase"] == "self_play"
        assert session["sent_count"] == 3
        assert 295 <= session["deadline_at"] - time.time() <= 301
        assert session["next_tick_at"] - time.time() <= 6
        assert engine._continuous_active(session)
    finally:
        engine.shutdown()


@pytest.mark.asyncio
async def test_continuous_tick_ignores_early_end_and_keeps_going():
    db = SimpleNamespace(save_scene_session=AsyncMock(return_value=True))
    engine = make_engine(db=db)
    engine.scheduler = SimpleNamespace(
        build_memory_context=AsyncMock(return_value=("", ""))
    )

    async def fake_invoke(**kwargs):
        event = kwargs["event"]
        event._mem_chat_last_send_ok = True
        event._mem_chat_last_structured_payload = {
            "response": {"segments": [{"text": "一"}, {"text": "二"}]}
        }
        event._mem_chat_scene_payload = {
            "active": True,
            "start": False,
            "phase": "finish",
            "intensity": 5,
            "end": True,
            "next_tick_sec": 90,
        }
        return True

    engine.llm_client = SimpleNamespace(invoke_llm=AsyncMock(side_effect=fake_invoke))
    umo = "aiocqhttp:friend:123"
    now = time.time()
    engine._sessions[umo] = {
        "umo": umo,
        "status": "active",
        "phase": "self_play",
        "intensity": 5,
        "started_at": now,
        "updated_at": now,
        "deadline_at": now + 300,
        "next_tick_at": now,
        "tick_count": 0,
        "sent_count": 3,
        "last_user_at": now,
        "last_bot_at": now,
        "scope_id": "123",
        "last_user_text": "连着发五分钟",
        "continuous": True,
        "continuous_until": now + 300,
    }
    engine._generations[umo] = 0

    tick_task = asyncio.create_task(engine._run_tick(umo, 0))
    engine._tasks[umo] = tick_task
    try:
        await tick_task
        session = engine._sessions[umo]
        assert session["status"] == "active"
        assert session["phase"] == "self_play"
        assert session["sent_count"] == 5
        assert session["last_scene_payload"]["end"] is False
        assert session["next_tick_at"] - time.time() <= 6
        successor = engine._tasks.get(umo)
        assert successor is not None
        assert not successor.done()
    finally:
        engine.shutdown()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_explicit_continuous_request_starts_scene_even_without_llm_scene_block():
    engine = make_engine()
    engine._load_active_session = AsyncMock(return_value=None)
    engine.start_from_scene_payload = AsyncMock(return_value={"status": "active"})
    event = SimpleNamespace(
        _mem_chat_last_send_ok=True,
        _mem_chat_last_structured_payload={
            "response": {"segments": [{"text": "一"}, {"text": "二"}]}
        },
        message_str="连着发五分钟",
        get_sender_id=lambda: "123",
        message_obj=SimpleNamespace(message_id="m3"),
    )
    await engine.on_scheduler_reply_completed(
        "aiocqhttp:friend:123", event, "at", sent=True, replied=True
    )
    engine.start_from_scene_payload.assert_awaited_once()
    scene = engine.start_from_scene_payload.await_args.args[2]
    assert scene["start"] is True
    assert scene["phase"] == "self_play"
    kwargs = engine.start_from_scene_payload.await_args.kwargs
    assert kwargs["initial_sent_count"] == 2
