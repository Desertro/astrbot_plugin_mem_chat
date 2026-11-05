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
    "src.llm_client_parts.review_image_service",
    PLUGIN / "src/llm_client_parts/image_service.py",
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


def test_private_prompt_uses_private_scope_and_teasing_boundary():
    sys.path.insert(0, str(PLUGIN))
    try:
        from src.llm_client import LLMClient
    finally:
        sys.path.pop(0)
    state = SimpleNamespace(
        get_last_mood_state=lambda _umo: "中性",
        get_mood_state_ttl_remaining=lambda _umo: 0,
        get_self_id=lambda _umo: "",
        get_self_name=lambda _umo: "",
    )
    client = LLMClient(SimpleNamespace(config={}, state=state))
    instructions = client._get_instruction_prompts(is_private=True)
    assert "【私聊技能总模板" in instructions
    assert "群聊技能总模板" not in instructions
    assert "私聊里对方的话默认是在对你说" in instructions
    assert "当前只能擦边" in instructions
    assert "不升级为露骨成人内容" in instructions
    assert "R2." in instructions

    private_system = client._build_system_prompts(
        None, "", "default:FriendMessage:10001"
    )
    assert "要不要开口｜私聊" in private_system
    assert "不要当群聊旁白" in private_system
    assert "当前只能擦边" in private_system
    assert "不进入露骨成人场景" in private_system
    assert "当前会话尺度｜擦边模式" in private_system

    group_system = client._build_system_prompts(None, "", "default:GroupMessage:20002")
    assert "先做出情绪反应" in group_system
    assert "不要当旁观解说员" in group_system
    assert "表情可以单独作为一次回应" in group_system
    assert "少用“爱丽丝觉得" in group_system
    for prompt in (instructions, private_system, group_system):
        assert "自称统一使用“爱丽丝”" in prompt
        assert "不用“我”或“人家”自称" in prompt
        assert "凯伊" in prompt
        assert "引用他人原话时保留原有称谓" in prompt
    assert "普通群聊保持原设语气" in group_system

def test_persona_is_friendly_by_default_and_proactive_in_erotic_mode():
    """Guard the persona direction and its whitelist-only adult scope."""
    sys.path.insert(0, str(PLUGIN))
    try:
        from src.llm_client import LLMClient
    finally:
        sys.path.pop(0)
    state = SimpleNamespace(
        get_last_mood_state=lambda _umo: "中性",
        get_mood_state_ttl_remaining=lambda _umo: 0,
        get_self_id=lambda _umo: "",
        get_self_name=lambda _umo: "",
    )
    client = LLMClient(
        SimpleNamespace(
            config={
                "scene_adult_mode": True,
                "adult_group_whitelist": ["775066874"],
            },
            state=state,
        )
    )
    rules = client._load_state_rules()
    assert "默认对普通群友友善" in rules["read_air"]
    assert "先对消息本身有反应" in rules["read_air"]
    assert "不要每次都写成旁观短评" in rules["read_air"]
    assert "先有情绪反应" in rules["focus:中性"]
    assert "主动带节奏" in rules["focus:色色"]
    assert "每轮至少主动推进" in rules["focus:色色"]

    persona = (PLUGIN / "personas/personas.txt").read_text(encoding="utf-8")
    assert "默认对普通人是善意、友好" in persona
    assert "熟人玩笑可以轻松吐槽" in persona
    assert "不要只当被动应答器" in persona
    assert "人味不等于完整短评" in persona
    assert "自称统一使用“爱丽丝”" in persona
    assert "不用“我”或“人家”自称" in persona
    assert "明确切换为凯伊（Kei）时" in persona
    assert "凯伊保留独立自称" in persona
    assert "不要每句都重复" in persona
    assert "原设日常语气" in persona
    assert "表情包是日常语气的一部分" in persona
    assert "爱丽丝，好像明白了！" in persona
    assert "不能拿来装傻逃避擦边" in persona
    assert "对方在难受" in persona
    assert "认真纠正爱丽丝时可以坦然承认" in persona

    private_system = client._build_system_prompts(
        None, "", "default:FriendMessage:10001"
    )
    assert "普通正常互动默认善意" in private_system
    assert "当前只能擦边" in private_system
    assert "对方认真纠正时坦然承认" in private_system

    safe_group_system = client._build_system_prompts(
        None, "", "default:GroupMessage:20002"
    )
    assert "默认对普通群友友善" in safe_group_system
    assert "熟人玩笑轻松接住" in safe_group_system
    assert "讨论 AI、设定或数据库时平静说明身份" in safe_group_system
    assert "当前会话尺度｜擦边模式" in safe_group_system
    assert "主动带节奏" not in safe_group_system

    adult_group_system = client._build_system_prompts(
        None, "", "default:GroupMessage:775066874"
    )
    assert "当前会话尺度｜成人模式" in adult_group_system
    assert "本会话是私密的成人向角色扮演" in adult_group_system

def test_adult_persona_scope_is_limited_to_whitelisted_groups():
    """Only configured group chats may receive explicit adult instructions."""
    sys.path.insert(0, str(PLUGIN))
    try:
        from src.llm_client import LLMClient
    finally:
        sys.path.pop(0)
    state = SimpleNamespace(
        get_last_mood_state=lambda _umo: "中性",
        get_mood_state_ttl_remaining=lambda _umo: 0,
        get_self_id=lambda _umo: "",
        get_self_name=lambda _umo: "",
    )
    client = LLMClient(
        SimpleNamespace(
            config={
                "scene_adult_mode": True,
                "adult_group_whitelist": ["72208748", "775066874", "1009481409"],
            },
            state=state,
        )
    )

    assert client._is_adult_allowed("default:GroupMessage:72208748")
    assert client._is_adult_allowed("default:GroupMessage:775066874")
    assert client._is_adult_allowed("default:GroupMessage:1009481409")
    assert not client._is_adult_allowed("default:GroupMessage:20002")
    assert not client._is_adult_allowed("default:FriendMessage:10001")
    assert not LLMClient(
        SimpleNamespace(
            config={
                "scene_adult_mode": False,
                "adult_group_whitelist": ["72208748"],
            },
            state=state,
        )
    )._is_adult_allowed("default:GroupMessage:72208748")

    safe_system = client._build_system_prompts(
        None, "", "default:GroupMessage:20002", task_type="at"
    )
    safe_instructions = client._get_instruction_prompts(
        task_type="at", is_private=False, adult_allowed=False
    )
    for forbidden in (
        "主动解扣",
        "详细成人画面",
        "可以直白说骚话",
        "描写性行为画面",
        "每轮至少主动推进",
        "主动说出想对对方做什么",
    ):
        assert forbidden not in safe_system
        assert forbidden not in safe_instructions
    assert "当前会话尺度｜擦边模式" in safe_system
    assert "当前只能擦边" in safe_instructions

    adult_system = client._build_system_prompts(
        None, "", "default:GroupMessage:775066874", task_type="at"
    )
    assert "当前会话尺度｜成人模式" in adult_system
    assert "主动解扣" in adult_system

def test_at_prompt_uses_proactive_flirty_state_not_previous_sarcasm():
    """An @ reply must not inherit the stale sarcastic mood or lose the persona."""
    sys.path.insert(0, str(PLUGIN))
    try:
        from src.llm_client import LLMClient
    finally:
        sys.path.pop(0)
    state = SimpleNamespace(
        get_last_mood_state=lambda _umo: "讥讽",
        get_mood_state_ttl_remaining=lambda _umo: 0,
        get_self_id=lambda _umo: "",
        get_self_name=lambda _umo: "",
    )
    client = LLMClient(
        SimpleNamespace(
            config={
                "scene_adult_mode": True,
                "adult_group_whitelist": ["775066874"],
            },
            state=state,
        )
    )
    rules = client._load_state_rules()
    assert "at" in rules
    at_rule = rules["at"]
    assert "闷骚色小鬼" in at_rule
    assert "明确邀请“涩涩”" in at_rule
    assert "第一轮就接住" in at_rule
    assert "上一轮心境只作背景" in at_rule
    assert "对方明确拒绝、喊停" in at_rule

    system_prompt = client._build_system_prompts(
        None,
        "",
        "default:GroupMessage:775066874",
        task_type="at",
    )
    assert "被 @ 或点名的直接对话" in system_prompt
    assert "明确邀请“涩涩”" in system_prompt
    assert "第一轮就接住" in system_prompt
    assert "只在对方讨嫌、冒犯或双方明确玩嘴硬梗时启用" not in system_prompt


def test_group_prompt_allows_reaction_fragments_and_standalone_memes(monkeypatch):
    """Keep group replies from collapsing into narrated one-liners."""
    sys.path.insert(0, str(PLUGIN))
    try:
        from src.llm_client import LLMClient
    finally:
        sys.path.pop(0)
    state = SimpleNamespace(
        get_last_mood_state=lambda _umo: "中性",
        get_mood_state_ttl_remaining=lambda _umo: 0,
        get_self_id=lambda _umo: "",
        get_self_name=lambda _umo: "",
    )
    client = LLMClient(
        SimpleNamespace(config={}, state=state, context=SimpleNamespace())
    )
    monkeypatch.chdir(PLUGIN.parents[2])
    instructions = client._get_instruction_prompts(is_private=False)
    assert "表情可以作为单独一条消息" in instructions
    assert "表情包是日常语气的一部分" in instructions
    assert "纯文字能说清就不用加" not in instructions
    assert "6.2a 表情可以单独作为一条消息" in instructions
    assert "6.2b 历史消息末尾的" in instructions


@pytest.mark.asyncio
async def test_light_reaction_avoids_bawling_meme(service):
    """Do not use a tearful sticker for a playful speechless reaction."""
    svc, library, tags = service
    paths = library([0.85, 0.80])
    tags[paths[0]] = {"emotion": "委屈、大哭"}
    selected = await svc.pick_meme_image_file(
        query="被对方这套自我招供逗到无语又想笑，只想摊手", conversation_id="group-a"
    )
    assert selected == paths[1]


def test_parse_validated_keeps_main_reply_before_tagged_metadata():
    """Accept the model's JSON reply plus XML-like sidecar metadata blocks."""
    sys.path.insert(0, str(PLUGIN))
    try:
        from src.llm.response_schemas import StructuredResponse
    finally:
        sys.path.pop(0)

    raw = """{"control":{"decision":"reply_only"},"response":{"text":"上线收垃圾，听着像每天日常任务刷新了——那今天打算捡满几口袋？ [表情:听到日常任务刷新，觉得好笑又想调侃一句]"}}

<mood>
{"text":"听到熟悉的捡垃圾日常，忍不住想笑着接一句","state":"中性","ttl_sec":120,"intensity":0.3}
</mood>

<affinity>
{"delta":1,"reason":"轻松接日常玩笑","target_qq":"1456941987"}
</affinity>

<memory_importance>2</memory_importance>

<impression_delta>
[]
</impression_delta>"""
    parsed = StructuredResponse.parse_validated(raw)
    assert parsed is not None
    assert parsed["control"]["decision"] == "reply_only"
    assert "捡满几口袋" in parsed["response"]["text"]
    assert parsed["mood"]["state"] == "中性"
    assert parsed["affinity"]["delta"] == 1
    assert parsed["memory_importance"] == 2


def test_parse_validated_accepts_expression_only_reply():
    """An emotion-only message is a reply, not structural metadata."""
    sys.path.insert(0, str(PLUGIN))
    try:
        from src.llm.response_schemas import StructuredResponse
    finally:
        sys.path.pop(0)

    raw = "  [表情:被这种理想工作逗到，羡慕又无奈地笑]"
    parsed = StructuredResponse.parse_validated(raw)
    assert parsed is not None
    assert parsed["control"]["decision"] == "reply_only"
    assert parsed["response"]["text"].startswith("[表情:")


def test_internal_meme_history_note_never_reaches_chat():
    """Model-copied sticker metadata is stripped from visible replies."""
    sys.path.insert(0, str(PLUGIN))
    try:
        from src.llm_client import LLMClient
    finally:
        sys.path.pop(0)

    client = LLMClient(SimpleNamespace(config={}))
    raw = (
        "鹿大师你这追问的架势，跟催更一样……鲸鲸都摆明不说了。"
        "（实际发送的表情包：画面描述：表达“我不说、别问了”的嘴硬躲闪情绪，"
        "可带点小得意，无文字。）"
    )
    assert (
        client._sanitize_final_reply(raw)
        == "鹿大师你这追问的架势，跟催更一样……鲸鲸都摆明不说了。"
    )
    assert (
        client._sanitize_final_reply(
            "（系统记录：本轮实际发送了表情包；画面描述：开心地笑；图中文字：无）"
        )
        == ""
    )
    assert (
        client._strip_internal_meme_history_notes(
            "正文（实际发送的表情包：画面描述：未闭合"
        )
        == "正文"
    )


@pytest.mark.parametrize(
    "text",
    [
        "我会保护王女。",
        "凯伊：我会保护王女。",
        "爱丽丝明白了，我去看看。",
        "人家才不要……我有自己的想法。",
        "你说「人家好累」，我们晚点再聊。[表情:人家害羞]",
        "自我怀疑也不会让我退开。",
    ],
)
def test_final_reply_preserves_character_self_references(text):
    """Preserve each speaker's first person through outgoing reply cleanup."""
    sys.path.insert(0, str(PLUGIN))
    try:
        from src.llm_client import LLMClient
    finally:
        sys.path.pop(0)

    client = LLMClient(SimpleNamespace(config={}))
    assert client._sanitize_final_reply(text) == text


@pytest.mark.asyncio
async def test_empty_reply_only_payload_is_retriable_failure():
    """An empty generation must not consume the message as a silent reply."""
    sys.path.insert(0, str(PLUGIN))
    try:
        from src.llm_client import LLMClient
    finally:
        sys.path.pop(0)

    client = LLMClient.__new__(LLMClient)
    send_ok = await client._process_and_send_final_reply(
        "",
        None,
        "at",
        [],
        [],
        [],
        "default:FriendMessage:853058285",
        [],
        None,
    )

    assert send_ok is False
