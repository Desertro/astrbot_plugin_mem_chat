"""Companion plugin observations must remain passive, scoped and factual."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image as PILImage
from src.core.plugin_activity import PluginActivityStore

from astrbot.api.message_components import Image, Node, Nodes, Plain
from astrbot.core.message.message_event_result import MessageEventResult
from data.plugins.astrbot_plugin_parser.main import ParserPlugin
from data.plugins.astrbot_plugin_shoubanhua.main import FigurineProPlugin
from main import MemChat


class Event:
    unified_msg_origin = "default:FriendMessage:test"
    message_str = "https://example.org/art/123"

    def __init__(self):
        self.message_obj = SimpleNamespace(message_id="request-1")
        self.extra = {}
        self.result = None
        self.send = AsyncMock()

    def get_extra(self, key, default=None):
        return self.extra.get(key, default)

    def set_extra(self, key, value):
        self.extra[key] = value

    def get_sender_id(self):
        return "user-1"

    def get_sender_name(self):
        return "tester"

    def get_group_id(self):
        return ""

    def get_platform_id(self):
        return "default"

    def get_result(self):
        return self.result

    def clear_result(self):
        self.result = None

    def _outline_chain(self, chain):
        return "test components"

    def chain_result(self, chain):
        return MessageEventResult(chain=chain)

    def plain_result(self, text):
        return MessageEventResult().message(text)


@pytest.fixture
def setup(tmp_path):
    owner = MemChat.__new__(MemChat)
    owner.config = {"private_chat_uids": ["user-1"]}
    owner.plugin_activities = PluginActivityStore(tmp_path / "activities")
    owner.scheduler = SimpleNamespace(trigger_cycle=AsyncMock())
    event = Event()
    image = tmp_path / "parser-cache.png"
    PILImage.new("RGB", (20, 20), "green").save(image)
    return owner, event, image


@pytest.mark.asyncio
async def test_persistent_observation_retains_image_without_scheduling(setup):
    owner, event, image = setup
    await owner.record_plugin_activity(
        event,
        plugin="parser",
        status="sent",
        summary="Art title",
        chain=[Image(file=str(image))],
    )
    await owner.record_plugin_activity(
        event,
        plugin="parser",
        status="sent",
        summary="Art title",
        chain=[Image(file=str(image))],
    )
    image.unlink()
    restarted = PluginActivityStore(owner.plugin_activities.directory)
    records = restarted.recent(event.unified_msg_origin, [])
    assert len(records) == 1
    assert records[0]["source_message_id"] == "request-1"
    assert records[0]["trigger_user_id"] == "user-1"
    assert Path(records[0]["images"][0]).is_file()
    assert restarted.recent("default:FriendMessage:other", []) == []
    owner.scheduler.trigger_cycle.assert_not_awaited()


@pytest.mark.asyncio
async def test_merged_forward_images_captions_cached_and_missing_caption_honest(setup):
    owner, event, image = setup
    chain = [
        Nodes(nodes=[Node(uin="bot", name="解析器", content=[Image(file=str(image))])])
    ]
    await owner.record_plugin_activity(
        event,
        plugin="parser",
        status="sent",
        summary="A title is not a caption",
        chain=chain,
    )
    vision = SimpleNamespace(
        describe_image_for_context=AsyncMock(return_value="绿色方块，文字：无")
    )
    block, images = await owner.plugin_activities.context(
        event.unified_msg_origin, ["request-1"], vision
    )
    assert "绿色方块" in block and "系统记录" in block
    assert len(images) == 1
    await owner.plugin_activities.context(event.unified_msg_origin, [], vision)
    vision.describe_image_for_context.assert_awaited_once()
    with owner.plugin_activities.connect() as con:
        con.execute("DELETE FROM captions")
    vision.describe_image_for_context = AsyncMock(side_effect=RuntimeError("offline"))
    block, images = await owner.plugin_activities.context(
        event.unified_msg_origin, [], vision
    )
    assert "暂未识别" in block and len(images) == 1
    assert '"attached_image_numbers": [1]' in block
    assert "并非用户又发送了新图片" in block
    assert "文字描述暂未识别不代表所附原图不可见" in block


@pytest.mark.asyncio
async def test_unknown_source_and_missing_image_do_not_invent_picture(setup):
    owner, event, image = setup
    await owner.record_plugin_activity(
        event, plugin="unknown", status="sent", summary="ignore"
    )
    assert owner.plugin_activities.recent(event.unified_msg_origin, []) == []
    await owner.record_plugin_activity(
        event, plugin="shoubanhua", status="failed", summary="生成失败"
    )
    block, images = await owner.plugin_activities.context(
        event.unified_msg_origin, [], SimpleNamespace()
    )
    assert '"status": "failed"' in block and images == []


def test_current_message_references_take_priority_over_newer_activity(tmp_path):
    store = PluginActivityStore(tmp_path)
    store.record("scope", {"source_message_id": "needed", "summary": "matching"})
    for i in range(9):
        store.record(
            "scope", {"source_message_id": f"unrelated-{i}", "summary": "other"}
        )
    rows = store.recent("scope", ["needed"])
    assert len(rows) == 6
    assert any(row["source_message_id"] == "needed" for row in rows)
    assert not store.recent("another-scope", ["needed"])


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_send", [False, True])
async def test_parser_direct_send_reports_actual_delivery(setup, fail_send):
    owner, event, image = setup
    plugin = ParserPlugin.__new__(ParserPlugin)
    plugin.config = {}
    plugin.context = SimpleNamespace(
        get_registered_star=lambda _: SimpleNamespace(activated=True, star_cls=owner)
    )
    plugin._image_key_cache = {}
    plugin._recent_image_keys = {}
    plugin._build_send_plan = lambda _: {
        "preview_card": True,
        "render_card": False,
        "force_merge": False,
        "light": [],
        "heavy": [],
    }
    plugin.renderer = SimpleNamespace(render_card=AsyncMock(return_value=image))
    if fail_send:
        event.send.side_effect = RuntimeError("send failed")
        with pytest.raises(RuntimeError):
            await plugin._send_parse_result(
                event,
                SimpleNamespace(
                    title="Parsed artwork", url="https://example.org/art/123"
                ),
            )
    else:
        await plugin._send_parse_result(
            event,
            SimpleNamespace(title="Parsed artwork", url="https://example.org/art/123"),
        )
    records = owner.plugin_activities.recent(event.unified_msg_origin, [])
    assert len(records) == 1
    assert records[0]["status"] == ("send_failed" if fail_send else "sent")
    assert bool(records[0]["images"]) is (not fail_send)
    owner.scheduler.trigger_cycle.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("fail_image", [False, True])
async def test_figurine_report_uses_actual_respond_stage_receipt(
    setup, monkeypatch, segmented, fail_image
):
    import astrbot.core.pipeline.respond.stage as respond
    from astrbot.core.pipeline.respond.stage import RespondStage

    owner, event, image = setup
    plugin = FigurineProPlugin.__new__(FigurineProPlugin)
    plugin.context = SimpleNamespace(
        get_registered_star=lambda _: SimpleNamespace(activated=True, star_cls=owner)
    )
    event.result = event.chain_result([Image(file=str(image)), Plain("生成成功")])
    event.set_extra(
        "_mem_chat_shoubanhua_activity", {"status": "sent", "summary": "生成图片成功"}
    )
    event.set_extra("_respond_delivered_components", [Image(file=str(image))])
    stage = RespondStage()
    stage.platform_settings = {}
    stage.is_seg_reply_required = lambda _: segmented
    stage._calc_comp_interval = AsyncMock(return_value=0)
    monkeypatch.setattr(respond.asyncio, "sleep", AsyncMock())

    async def send(chain):
        if fail_image and any(isinstance(part, Image) for part in chain.chain):
            raise RuntimeError("image send failed")

    event.send = AsyncMock(side_effect=send)

    async def hook(current_event, _type):
        await plugin.report_companion_output(current_event)
        return False

    monkeypatch.setattr(respond, "call_event_hook", hook)
    await stage.process(event)
    records = owner.plugin_activities.recent(event.unified_msg_origin, [])
    assert len(records) == 1
    assert records[0]["status"] == ("send_failed" if fail_image else "sent")
    assert bool(records[0]["images"]) is (not fail_image)
    assert event.get_extra("_mem_chat_shoubanhua_activity") is None
    owner.scheduler.trigger_cycle.assert_not_awaited()


@pytest.mark.asyncio
async def test_observations_reach_real_llm_invocation_with_images(setup):
    owner, event, image = setup
    await owner.record_plugin_activity(
        event,
        plugin="parser",
        status="sent",
        summary="圣园未花插画",
        chain=[Image(file=str(image))],
    )
    fixture = runpy.run_path(str(Path(__file__).with_name("test_protocol_retry.py")))[
        "client"
    ].__wrapped__
    client = fixture()
    client.plugin.plugin_activities = owner.plugin_activities
    client._image_service.describe_image_for_context = AsyncMock(
        return_value="粉发少女，文字：早上好"
    )
    seen = []

    async def generate(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(
            completion_text=json.dumps(
                {
                    "control": {"decision": "reply_only"},
                    "response": {"text": "可以找找类似的未花插画。"},
                },
                ensure_ascii=False,
            ),
            tools_call_name=[],
            tools_call_args=[],
            tools_call_ids=[],
            raw_completion=None,
        )

    client._call_custom_llm = generate
    assert await client.invoke_llm(
        event.unified_msg_origin,
        "还有没有类似的",
        "",
        event=event,
        quote_candidates=[("request-1", "link")],
    )
    assert "粉发少女" in seen[0]["prompt"]
    assert "插件活动" in seen[0]["prompt"]
    assert seen[0]["image_urls"]
    assert len(seen) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("generated", [True, False])
async def test_real_image_generator_yields_metadata_consumed_after_delivery(
    setup, generated
):
    owner, event, image = setup
    plugin = FigurineProPlugin.__new__(FigurineProPlugin)
    plugin.conf = {"enable_user_limit": False, "enable_group_limit": False}
    plugin.context = SimpleNamespace(
        get_config=lambda: {"admins_id": []},
        get_registered_star=lambda _: SimpleNamespace(activated=True, star_cls=owner),
    )
    plugin.count_manager = SimpleNamespace(
        check_and_decrease_quota=AsyncMock(return_value=(True, None))
    )
    plugin.iwf = SimpleNamespace(
        _image_source_kind=lambda _event: (True, [], False),
        _has_history_image_reference=lambda _prompt: False,
        get_images=AsyncMock(return_value=[image.read_bytes()]),
    )
    plugin._call_api = AsyncMock(
        return_value=(image.read_bytes(), "test-model")
        if generated
        else "temporary failure"
    )
    statuses = []
    async for result in plugin._process_figurine_request(event, "bnn", "a cat"):
        activity = event.get_extra("_mem_chat_shoubanhua_activity")
        assert activity
        event.set_extra("_respond_delivered_components", list(result.chain))
        await plugin.report_companion_output(event)
        statuses.append(
            owner.plugin_activities.recent(event.unified_msg_origin, [])[-1]["status"]
        )
    assert statuses == ["running", "sent" if generated else "failed"]
    latest = owner.plugin_activities.recent(event.unified_msg_origin, [])[-1]
    assert latest["input_image_count"] == 1
    assert len(latest["input_images"]) == 1

    block, attached = await owner.plugin_activities.context(
        event.unified_msg_origin,
        ["request-1"],
        SimpleNamespace(describe_image_for_context=AsyncMock(return_value=None)),
    )
    entries = [json.loads(line) for line in block.splitlines() if line.startswith("{")]
    final_entry = entries[-1]
    if generated:
        assert final_entry["attached_image_roles"] == ["plugin_output"]
    else:
        assert final_entry["attached_image_roles"] == ["request_reference"]
    assert len(attached) == 1


def test_expired_records_and_unreferenced_assets_are_removed(tmp_path):
    import os
    import time

    store = PluginActivityStore(tmp_path)
    store.assets.mkdir(parents=True)
    old = store.assets / "expired.png"
    old.write_bytes(b"expired")
    os.utime(old, (time.time() - 8 * 86400,) * 2)
    keep = store.assets / "active.png"
    keep.write_bytes(b"active")
    with store.connect() as con:
        con.execute(
            "INSERT INTO activities VALUES(?,?,?,?)",
            (
                "expired",
                "scope",
                time.time() - 8 * 86400,
                json.dumps({"images": [str(old)]}),
            ),
        )
    input_keep = store.assets / "input-active.png"
    input_keep.write_bytes(b"input-active")
    store.record(
        "scope",
        {
            "source_message_id": "current",
            "images": [str(keep)],
            "input_images": [str(input_keep)],
        },
    )
    assert not old.exists() and keep.exists() and input_keep.exists()
    assert len(store.recent("scope", [])) == 1


@pytest.mark.asyncio
async def test_companion_capture_respects_existing_conversation_allowlists(setup):
    owner, event, image = setup
    owner.config = {
        "private_chat_uids": ["another-user"],
        "group_whitelist": ["allowed-group"],
    }
    await owner.record_plugin_activity(
        event,
        plugin="parser",
        status="sent",
        summary="private",
        chain=[Image(file=str(image))],
    )
    assert owner.plugin_activities.recent(event.unified_msg_origin, []) == []
    event.get_group_id = lambda: "other-group"
    event.unified_msg_origin = "default:GroupMessage:other-group"
    await owner.record_plugin_activity(
        event,
        plugin="parser",
        status="sent",
        summary="group",
        chain=[Image(file=str(image))],
    )
    assert owner.plugin_activities.recent(event.unified_msg_origin, []) == []


@pytest.mark.asyncio
async def test_attached_image_numbers_follow_explicit_current_source(setup, tmp_path):
    """A quoted source wins, while an unrelated newer activity stays text-only."""
    owner, event, image = setup
    await owner.record_plugin_activity(
        event,
        plugin="parser",
        status="sent",
        summary="older parser image",
        chain=[Image(file=str(image))],
    )
    second = tmp_path / "generated.png"
    PILImage.new("RGB", (30, 30), "blue").save(second)
    event.message_obj.message_id = "request-2"
    await owner.record_plugin_activity(
        event,
        plugin="shoubanhua",
        status="sent",
        summary="newer generated image",
        chain=[Image(file=str(second))],
    )
    block, images = await owner.plugin_activities.context(
        event.unified_msg_origin,
        ["request-1"],
        SimpleNamespace(describe_image_for_context=AsyncMock(return_value=None)),
    )
    entries = [json.loads(line) for line in block.splitlines() if line.startswith("{")]
    assert len(images) == 1
    parser_entry = next(item for item in entries if item["plugin"] == "parser")
    newer_entry = next(item for item in entries if item["plugin"] == "shoubanhua")
    assert parser_entry["attached_image_numbers"] == [1]
    assert parser_entry["attached_image_roles"] == ["plugin_output"]
    assert newer_entry["attached_image_numbers"] == []


@pytest.mark.asyncio
async def test_failed_bnn_uses_its_reference_not_older_parser_output(setup, tmp_path):
    """A failed request may attach its input reference, never older output images."""
    owner, event, _image = setup
    parser_images = []
    parser_chain = []
    for index in range(4):
        path = tmp_path / f"xiaohongshu-{index}.png"
        PILImage.new("RGB", (20 + index, 20), "green").save(path)
        parser_images.append(path)
        parser_chain.append(Image(file=str(path)))
    await owner.record_plugin_activity(
        event,
        plugin="parser",
        status="sent",
        summary="练习 大肥鱼 鲸鲸",
        chain=parser_chain,
    )

    kay = tmp_path / "kay-reference.png"
    PILImage.new("RGB", (64, 64), "purple").save(kay)
    event.message_obj.message_id = "bnn-request"
    event.message_str = "bnn 手办化"
    await owner.record_plugin_activity(
        event,
        plugin="shoubanhua",
        status="failed",
        summary="bnn 图片生成失败，本次没有生成图片。",
        reference_images=[kay.read_bytes()],
    )

    block, attached = await owner.plugin_activities.context(
        event.unified_msg_origin,
        ["follow-up-question"],
        SimpleNamespace(
            describe_image_for_context=AsyncMock(return_value="紫发角色，凯伊")
        ),
    )
    assert len(attached) == 1
    assert Path(attached[0]).read_bytes() == kay.read_bytes()
    entries = [json.loads(line) for line in block.splitlines() if line.startswith("{")]
    parser_entry = next(item for item in entries if item["plugin"] == "parser")
    bnn_entry = next(item for item in entries if item["plugin"] == "shoubanhua")
    assert parser_entry["attached_image_numbers"] == []
    assert bnn_entry["image_count"] == 0
    assert bnn_entry["input_image_count"] == 1
    assert bnn_entry["attached_image_numbers"] == [1]
    assert bnn_entry["attached_image_roles"] == ["request_reference"]
    assert "紫发角色，凯伊" in block
