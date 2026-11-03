"""Regression tests for repeated-image familiarity metadata."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

PLUGIN = Path(__file__).resolve().parents[1]
if str(PLUGIN) not in sys.path:
    sys.path.insert(0, str(PLUGIN))

from src.core.media_familiarity import (  # noqa: E402
    analyze_recent_media_occurrences,
    classify_media_kind,
    extract_media_alias_intent,
    fill_plain_image_captions,
    format_media_familiarity_block,
    media_reference_fingerprint,
    normalize_media_alias,
    pair_captions_with_images,
    parse_media_alias_actions,
)
from src.db.database import DatabaseManager  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402

from main import MemChat  # noqa: E402


def _media_row(message_id, image_path, caption, created_at, sender="阿猫"):
    return {
        "message_id": message_id,
        "sender_id": "10001",
        "sender_name": sender,
        "text": f"[图片内容: {caption}]",
        "image_urls_json": json.dumps([str(image_path)]),
        "created_at": created_at,
    }


def test_image_fingerprint_uses_bytes_not_temporary_filename(tmp_path):
    first = tmp_path / "media_image_001.jpg"
    second = tmp_path / "media_image_999.jpg"
    different = tmp_path / "media_image_other.jpg"
    first.write_bytes(b"same-sticker-bytes")
    second.write_bytes(b"same-sticker-bytes")
    different.write_bytes(b"another-sticker")

    assert media_reference_fingerprint(str(first)) == media_reference_fingerprint(
        str(second)
    )
    assert media_reference_fingerprint(str(first)) != media_reference_fingerprint(
        str(different)
    )


def test_generic_caption_does_not_merge_different_images(tmp_path):
    current = tmp_path / "current.jpg"
    repeated = tmp_path / "repeated.jpg"
    unrelated = tmp_path / "unrelated.jpg"
    current.write_bytes(b"meme-a")
    repeated.write_bytes(b"meme-a")
    unrelated.write_bytes(b"meme-b")
    rows = [
        _media_row("m1", repeated, "黄暴图片", 100.0, "甲"),
        _media_row("m2", unrelated, "黄暴图片", 200.0, "乙"),
    ]

    result = analyze_recent_media_occurrences([str(current)], ["黄暴图片"], rows)[0]

    assert result["prior_occurrences"] == 1
    assert result["last_seen_at"] == 100.0
    assert result["last_sender_name"] == "甲"


def test_historical_match_tolerates_minor_caption_drift(tmp_path):
    current = tmp_path / "current.jpg"
    repeated = tmp_path / "repeated.jpg"
    current.write_bytes(b"meme-c")
    repeated.write_bytes(b"meme-c")
    old_caption = "Q版蓝发女仆角色抱着饭碗委屈鼓脸，表达没吃到饭的难过和撒娇。"
    new_caption = "Q版蓝发女仆角色抱着饭碗委屈鼓脸，表达没吃到饭的难过、撒娇情绪。"
    rows = [_media_row("m1", repeated, old_caption, 321.0, "鲸鲸")]

    result = analyze_recent_media_occurrences([str(current)], [new_caption], rows)[0]

    assert result["prior_occurrences"] == 1
    assert result["last_sender_name"] == "鲸鲸"


def test_familiarity_block_only_marks_repeated_images_and_keeps_judgement():
    assert format_media_familiarity_block([{"caption": "新图", "seen_count": 1}]) == ""

    block = format_media_familiarity_block(
        [
            {
                "caption": "蓝发女仆抱饭碗委屈鼓脸",
                "seen_count": 6,
                "last_seen_at": 1_790_000_000.0,
                "last_sender_name": "鲸鲸",
            }
        ],
        now=1_790_000_100.0,
    )

    assert "是本会话里熟悉的表情包" in block
    assert "- 可以围绕这次的意思自然地熟人式吐槽。" in block.splitlines()
    assert "不表示今天刷屏" in block
    assert "共出现" not in block
    assert "鲸鲸" not in block
    assert "上次" not in block
    assert "又是这张" not in block


def test_media_kind_prefers_qq_sticker_flag_and_keeps_plain_images_plain():
    assert classify_media_kind(1, "") == "sticker"
    assert classify_media_kind(0, "QQ 表情包：蓝发女仆抱饭碗") == "sticker"
    assert classify_media_kind(0, "一个女孩露出委屈的表情") == "image"
    assert classify_media_kind(0, "现实照片，桌上有两杯茶") == "image"


@pytest.mark.parametrize("seen_count", [1, 2, 3, 6, 60])
def test_familiarity_prompt_labels_repeated_sticker_and_guides_meme_reading(seen_count):
    block = format_media_familiarity_block(
        [
            {
                "caption": "蓝发女仆抱饭碗委屈鼓脸",
                "media_kind": "sticker",
                "is_current": True,
                "seen_count": seen_count,
                "last_seen_at": 1_790_000_000.0,
                "last_sender_name": "鲸鲸",
            }
        ],
        now=1_790_000_100.0,
        include_current=True,
    )
    assert "当前表情包（画面：蓝发女仆抱饭碗委屈鼓脸）" in block
    assert "表达心情、意境和态度的日常用语" in block
    assert "自然接它的情绪、梗或话题" in block
    assert "共出现" not in block
    assert "鲸鲸" not in block


def test_image_caption_pairing_keeps_order():
    text = "[图片内容: 第一张] 说话 [图片内容: 第二张]"
    assert pair_captions_with_images(text, 2) == ["第一张", "第二张"]


@pytest.mark.asyncio
async def test_record_media_familiarity_is_idempotent_and_scoped(tmp_path):
    db = DatabaseManager(str(tmp_path / "memory.db"), {})
    try:
        first = await db.record_media_familiarity(
            scope_id="group-a",
            fingerprint="sha256:abc",
            caption="一张常用表情",
            message_id="m1",
            observed_at=1000.0,
            prior_occurrences=2,
            first_seen_at=500.0,
            last_seen_at=900.0,
            last_sender_name="甲",
        )
        assert first["seen_count"] == 3
        assert first["last_seen_at"] == 900.0

        duplicate = await db.record_media_familiarity(
            scope_id="group-a",
            fingerprint="sha256:abc",
            caption="一张常用表情",
            message_id="m1",
            observed_at=1001.0,
        )
        assert duplicate["seen_count"] == 3
        assert duplicate["duplicate_message"] is True

        next_seen = await db.record_media_familiarity(
            scope_id="group-a",
            fingerprint="sha256:abc",
            caption="一张常用表情",
            message_id="m2",
            observed_at=1002.0,
        )
        assert next_seen["seen_count"] == 4

        other_scope = await db.record_media_familiarity(
            scope_id="group-b",
            fingerprint="sha256:abc",
            caption="一张常用表情",
            message_id="m1",
            observed_at=1003.0,
        )
        assert other_scope["seen_count"] == 1
    finally:
        db.close()


@pytest.mark.parametrize(
    "seen_count, kind_label", [(1, "图片"), (2, "表情包"), (6, "表情包")]
)
def test_media_familiarity_is_injected_into_the_current_model_turn(
    seen_count, kind_label
):
    client = LLMClient.__new__(LLMClient)
    event = SimpleNamespace(
        _mem_chat_media_familiarity=[
            {
                "caption": "蓝发女仆抱饭碗",
                "seen_count": seen_count,
                "media_kind": "image",
                "is_current": True,
                "last_seen_at": 1_790_000_000.0,
                "last_sender_name": "鲸鲸",
            }
        ]
    )

    prompt = client._inject_media_familiarity("[发图]蓝发女仆抱饭碗", event)

    assert prompt.startswith("[发图]蓝发女仆抱饭碗")
    assert "【表情熟悉度" in prompt
    assert f"当前{kind_label}（画面：蓝发女仆抱饭碗）" in prompt
    if seen_count >= 2:
        assert "是本会话里熟悉的表情包" in prompt
        assert "自然接它的情绪、梗或话题" in prompt
    else:
        assert "共出现" not in prompt
        assert "自然接它的情绪、梗或话题" not in prompt
    assert "共出现" not in prompt
    assert "鲸鲸" not in prompt
    assert client._inject_media_familiarity("普通消息", SimpleNamespace()) == "普通消息"


def test_explicit_media_alias_intent_accepts_names_and_rejects_questions():
    assert extract_media_alias_intent("这个表情包是大肥鱼") == {
        "action": "set",
        "alias": "大肥鱼",
    }
    assert extract_media_alias_intent("这张图叫大肥鱼") == {
        "action": "set",
        "alias": "大肥鱼",
    }
    assert extract_media_alias_intent("以后叫它大肥鱼") == {
        "action": "replace",
        "alias": "大肥鱼",
    }
    assert extract_media_alias_intent("这个表情包的标签改成大肥鱼") == {
        "action": "replace",
        "alias": "大肥鱼",
    }
    assert extract_media_alias_intent("别人说这个表情包是大肥鱼") == {
        "action": "set",
        "alias": "大肥鱼",
    }
    assert extract_media_alias_intent("别叫它大肥鱼了") == {
        "action": "delete",
        "alias": "大肥鱼",
    }

    assert extract_media_alias_intent("这个表情包不是大肥鱼") is None
    assert extract_media_alias_intent("这个表情包是大肥鱼吗") is None
    assert extract_media_alias_intent("大肥鱼好可爱") is None
    assert extract_media_alias_intent("谁是大肥鱼") is None
    assert extract_media_alias_intent("这是大肥鱼发的") is None


def test_media_alias_actions_are_parsed_and_removed_from_visible_text():
    cleaned, actions = parse_media_alias_actions(
        "好耶 [改图名:2:咸鱼] [记图名:大肥鱼] [删图名:旧名]"
    )
    assert cleaned.strip() == "好耶"
    assert actions == [
        {"action": "replace", "alias": "咸鱼", "image_index": 2},
        {"action": "set", "alias": "大肥鱼", "image_index": None},
        {"action": "delete", "alias": "旧名", "image_index": None},
    ]
    assert normalize_media_alias("  “大肥鱼” ") == "大肥鱼"


@pytest.mark.asyncio
async def test_media_aliases_are_scoped_and_support_replace_delete(tmp_path):
    db = DatabaseManager(str(tmp_path / "aliases.db"), {})
    try:
        first = await db.set_media_alias(
            scope_id="group-a",
            fingerprint="sha256:abc",
            alias="大肥鱼",
            source_sender_name="甲",
            source_message_id="m1",
        )
        assert first["aliases"] == ["大肥鱼"]

        await db.set_media_alias(
            scope_id="group-a",
            fingerprint="sha256:abc",
            alias="咸鱼",
            source_sender_name="乙",
            source_message_id="m2",
        )
        aliases = await db.get_media_aliases(
            scope_id="group-a", fingerprint="sha256:abc"
        )
        assert {item["alias"] for item in aliases} == {"大肥鱼", "咸鱼"}

        batch = await db.get_media_aliases_batch(
            scope_id="group-a",
            fingerprints=["sha256:abc", "sha256:missing"],
        )
        assert {item["alias"] for item in batch["sha256:abc"]} == {"大肥鱼", "咸鱼"}
        assert batch["sha256:missing"] == []

        replaced = await db.set_media_alias(
            scope_id="group-a",
            fingerprint="sha256:abc",
            alias="大肥鱼二号",
            replace=True,
        )
        assert replaced["aliases"] == ["大肥鱼二号"]

        other_scope = await db.get_media_aliases(
            scope_id="group-b", fingerprint="sha256:abc"
        )
        assert other_scope == []

        deleted = await db.delete_media_alias(
            scope_id="group-a",
            fingerprint="sha256:abc",
            alias="大肥鱼二号",
        )
        assert deleted["deleted"] == 1
        assert deleted["aliases"] == []
    finally:
        db.close()


def test_current_image_alias_and_learning_guidance_are_in_prompt():
    block = format_media_familiarity_block(
        [
            {
                "caption": "蓝发女仆抱饭碗",
                "media_kind": "sticker",
                "seen_count": 1,
                "is_current": True,
                "aliases": ["大肥鱼"],
            }
        ],
        include_current=True,
        alias_learning=True,
    )
    assert "当前表情包（画面：蓝发女仆抱饭碗）" in block
    assert "本会话里叫它“大肥鱼”" in block
    assert "[记图名:大肥鱼]" in block
    assert "[改图名:大肥鱼]" in block
    assert "[记图名:2:大肥鱼]" in block
    assert "普通玩梗、疑问、否定" in block


@pytest.mark.asyncio
async def test_llm_alias_action_updates_the_bound_image(tmp_path):
    db = DatabaseManager(str(tmp_path / "alias-action.db"), {})
    plugin = SimpleNamespace(
        config={"media_alias_enabled": True},
        db_mgr=db,
    )
    client = LLMClient.__new__(LLMClient)
    client.plugin = plugin

    class _Event:
        message_obj = SimpleNamespace(message_id="m7")

        def get_group_id(self):
            return "group-a"

        def get_sender_id(self):
            return "10001"

        def get_sender_name(self):
            return "甲"

    event = _Event()
    event._mem_chat_media_alias_candidates = [
        {
            "fingerprint": "sha256:abc",
            "caption": "蓝发女仆抱饭碗",
            "media_kind": "sticker",
            "image_index": 1,
        }
    ]
    try:
        assert await client._apply_media_alias_action(
            event, "set", "大肥鱼", image_index=1
        )
        aliases = await db.get_media_aliases(
            scope_id="group-a", fingerprint="sha256:abc"
        )
        assert [item["alias"] for item in aliases] == ["大肥鱼"]

        assert await client._apply_media_alias_action(
            event, "replace", "咸鱼", image_index=1
        )
        aliases = await db.get_media_aliases(
            scope_id="group-a", fingerprint="sha256:abc"
        )
        assert [item["alias"] for item in aliases] == ["咸鱼"]

        assert await client._apply_media_alias_action(
            event, "delete", "咸鱼", image_index=1
        )
        assert (
            await db.get_media_aliases(scope_id="group-a", fingerprint="sha256:abc")
            == []
        )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_attach_learns_explicit_alias_for_current_image_turn(tmp_path):
    db = DatabaseManager(str(tmp_path / "attach-alias.db"), {})
    image = tmp_path / "current-sticker.png"
    image.write_bytes(b"stable-current-sticker-bytes")

    plugin = MemChat.__new__(MemChat)
    plugin.config = {
        "media_familiarity_enabled": True,
        "media_alias_enabled": True,
        "media_familiarity_scan_limit": 20,
    }
    plugin.db_mgr = db

    class _Event:
        message_str = "这个表情包是大肥鱼"
        message_obj = SimpleNamespace(message_id="m10", group_id="group-a", message=[])

        def get_sender_id(self):
            return "10001"

        def get_sender_name(self):
            return "甲"

    event = _Event()
    try:
        await plugin._attach_media_familiarity_for_event(
            event,
            scope_id="group-a",
            message_id="m10",
            text="[图片内容: 蓝发女仆抱饭碗委屈鼓脸]",
            image_urls=[str(image)],
        )
        aliases = await db.get_media_aliases(
            scope_id="group-a",
            fingerprint=event._mem_chat_media_alias_candidates[0]["fingerprint"],
        )
        assert [item["alias"] for item in aliases] == ["大肥鱼"]
        assert event._mem_chat_media_familiarity[0]["aliases"] == ["大肥鱼"]
        prompt_block = format_media_familiarity_block(
            event._mem_chat_media_familiarity,
            include_current=True,
            alias_learning=event._mem_chat_media_alias_learning,
        )
        assert "本会话里叫它“大肥鱼”" in prompt_block
        assert "当前图片（画面：蓝发女仆抱饭碗委屈鼓脸）" in prompt_block
    finally:
        db.close()


@pytest.mark.asyncio
async def test_alias_action_can_resolve_a_recent_image_without_attachment(tmp_path):
    db = DatabaseManager(str(tmp_path / "recent-alias.db"), {})
    image = tmp_path / "recent-image.png"
    image.write_bytes(b"recent-image-alias-bytes")
    await db.save_raw_group_message(
        group_id="group-a",
        message_id="m-previous",
        message_seq=None,
        sender_id="20002",
        sender_name="乙",
        text="[图片内容: 一张蓝发女仆表情包]",
        raw_text="[图片]",
        image_urls_json=json.dumps([str(image)]),
        created_at=time.time(),
    )
    plugin = SimpleNamespace(
        config={
            "media_alias_enabled": True,
            "media_alias_recent_image_max_age_sec": 180,
        },
        db_mgr=db,
    )
    client = LLMClient.__new__(LLMClient)
    client.plugin = plugin

    class _Event:
        message_str = "这个表情包是大肥鱼"
        message_obj = SimpleNamespace(
            message_id="m-current", group_id="group-a", message=[]
        )

        def get_group_id(self):
            return "group-a"

        def get_sender_id(self):
            return "10001"

        def get_sender_name(self):
            return "甲"

    event = _Event()
    try:
        assert await client._apply_media_alias_action(event, "set", "大肥鱼")
        fingerprint = media_reference_fingerprint(str(image))
        aliases = await db.get_media_aliases(
            scope_id="group-a", fingerprint=fingerprint
        )
        assert [item["alias"] for item in aliases] == ["大肥鱼"]
    finally:
        db.close()


def test_fill_plain_image_captions_keeps_order_and_rejects_alignment_mismatch():
    text = "[图片]这也是擦边表情，平时想骂人也可以用"
    enriched = fill_plain_image_captions(text, ["我顶死你！"])
    assert enriched == "[图片内容: 我顶死你！]这也是擦边表情，平时想骂人也可以用"
    assert fill_plain_image_captions("[图片][图片]", ["只有一张"]) == "[图片][图片]"


@pytest.mark.asyncio
async def test_image_caption_is_persisted_and_reused_for_followup(tmp_path):
    from unittest.mock import AsyncMock

    image = tmp_path / "sticker.jpg"
    image.write_bytes(b"same-sticker-bytes")
    db = DatabaseManager(str(tmp_path / "captions.db"), {})
    describe = AsyncMock(
        return_value="图上文字：我顶死你！银发猫耳少女握拳，传达不服气的挑衅情绪。"
    )
    host = SimpleNamespace(
        db_mgr=db,
        llm_client=SimpleNamespace(
            _image_service=SimpleNamespace(describe_image_for_context=describe)
        ),
    )
    try:
        text = "[图片]这也是擦边表情，平时想骂人也可以用"
        enriched = await MemChat._ensure_image_captions(
            host, SimpleNamespace(), text, [str(image)], "853058285"
        )
        assert "[图片内容: 图上文字：我顶死你！" in enriched
        describe.assert_awaited_once_with(str(image))

        fingerprint = media_reference_fingerprint(str(image))
        await db.record_media_familiarity(
            scope_id="853058285",
            fingerprint=fingerprint,
            caption="图上文字：我顶死你！银发猫耳少女握拳，传达不服气的挑衅情绪。",
            message_id="m1",
            observed_at=time.time(),
        )
        describe.reset_mock()
        describe.side_effect = AssertionError(
            "cached caption should avoid another vision call"
        )
        cached = await MemChat._ensure_image_captions(
            host, SimpleNamespace(), text, [str(image)], "853058285"
        )
        assert "我顶死你" in cached
        describe.assert_not_awaited()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_followup_self_heals_a_recent_blank_image_caption(tmp_path):
    from unittest.mock import AsyncMock

    image = tmp_path / "sticker.jpg"
    image.write_bytes(b"same-sticker-bytes")
    db = DatabaseManager(str(tmp_path / "followup-captions.db"), {})
    try:
        await db.save_raw_group_message(
            group_id="853058285",
            message_id="m1",
            message_seq=None,
            sender_id="10001",
            sender_name="Calamity",
            text="[图片]这也是擦边表情，平时想骂人也可以用",
            raw_text="[图片]这也是擦边表情，平时想骂人也可以用",
            image_urls_json=json.dumps([str(image)]),
            created_at=time.time(),
        )
        fingerprint = media_reference_fingerprint(str(image))
        await db.record_media_familiarity(
            scope_id="853058285",
            fingerprint=fingerprint,
            caption="",
            message_id="m1",
            observed_at=time.time(),
        )
        describe = AsyncMock(
            return_value="我顶死你！银发猫耳少女握拳，传达不服气的挑衅情绪。"
        )
        host = SimpleNamespace(
            db_mgr=db,
            llm_client=SimpleNamespace(
                _image_service=SimpleNamespace(describe_image_for_context=describe)
            ),
        )

        class _Event:
            message_str = "上面写的什么字"

            def get_messages(self):
                return []

        result = await MemChat._append_recent_image_context(
            host, _Event(), "上面写的什么字", "853058285"
        )
        assert result.startswith("上面写的什么字")
        assert "【最近图片的系统资料】" in result
        assert "我顶死你！" in result
        cached = await db.get_media_familiarity_captions("853058285", [fingerprint])
        assert "我顶死你！" in cached[fingerprint]
    finally:
        db.close()
