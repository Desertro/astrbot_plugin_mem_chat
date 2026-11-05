"""Regression tests for exact, conversation-aware sticker deletion."""

import functools
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image as PILImage
from src.llm_client import LLMClient
from src.llm_client_parts.image_service import (
    _MEME_TAG_CACHE,
    _MEME_VEC_CACHE,
    ImageService,
)
from src.llm_client_parts.meme_library import MemeLibrary

from astrbot.api.message_components import Image, Reply
from main import MemChat


@pytest.fixture
def setup(tmp_path):
    data = tmp_path / "data/memes_data"
    root = data / "memes/like"
    root.mkdir(parents=True)
    files = [root / "first.png", root / "second.png"]
    for i, path in enumerate(files):
        PILImage.new("RGB", (40, 40), (100 + i * 70, 80, 190)).save(path)
    with sqlite3.connect(data / "meme_index.db") as con:
        con.execute(
            "CREATE TABLE meme_index (path TEXT PRIMARY KEY, description TEXT, vector BLOB)"
        )
        con.execute("CREATE TABLE meme_tags (path TEXT PRIMARY KEY, ocr_text TEXT)")
        for i, path in enumerate(files):
            con.execute(
                "INSERT INTO meme_index VALUES(?,?,?)",
                (str(path), f"sticker {i}", b"embedding"),
            )
            con.execute(
                "INSERT INTO meme_tags VALUES(?,?)", (str(path), f"caption {i}")
            )
    plugin = MemChat.__new__(MemChat)
    plugin.config = {"codex_enabled": False}
    tools = {}
    plugin.context = SimpleNamespace(
        add_llm_tools=lambda tool: tools.update({tool.name: tool}),
        get_llm_tool_manager=lambda: SimpleNamespace(get_func=tools.get),
    )
    plugin.llm_client = client = LLMClient(plugin)
    client._image_service = ImageService(
        lambda _key, default: default, get_cwd=lambda: str(tmp_path)
    )
    plugin._register_meme_library_tool()
    tools["meme_library"].handler = functools.partial(
        tools["meme_library"].handler, plugin
    )
    event = SimpleNamespace(
        unified_msg_origin="default:GroupMessage:a",
        message_str="这张不合适，删掉",
        message_obj=SimpleNamespace(message_id="request-a", message=[], raw_message={}),
        get_messages=lambda: [],
        get_self_id=lambda: "bot",
        get_sender_id=lambda: "user",
        get_group_id=lambda: "a",
        send=AsyncMock(),
    )
    return client, event, files, tools


def test_delete_removes_only_exact_active_file_and_both_indexes(setup):
    client, event, files, _ = setup
    library = client._image_service.meme_library
    ref = library.remember(event.unified_msg_origin, str(files[0]), str(files[0]))
    assert library.delete(event.unified_msg_origin, ref)["ok"]
    assert not files[0].exists()
    assert files[1].exists()
    assert library.recent(event.unified_msg_origin) == []
    with sqlite3.connect(library.db_path) as con:
        for table in ["meme_index", "meme_tags"]:
            assert con.execute(f"SELECT path FROM {table}").fetchall() == [
                (str(files[1]),)
            ]
    archived = list((library.data_dir / "deleted").glob("*/like/first.png"))
    assert len(archived) == 1
    metadata = json.loads((archived[0].parents[1] / "metadata.json").read_text())
    assert metadata["records"]["meme_index"][0]["vector"] == {"hex": b"embedding".hex()}
    assert not library.delete(event.unified_msg_origin, ref)["ok"]


def test_references_persist_and_never_cross_conversations(setup):
    client, event, files, _ = setup
    library = client._image_service.meme_library
    ref = library.remember(event.unified_msg_origin, str(files[0]), str(files[0]))
    restarted = MemeLibrary(library.data_dir)
    assert restarted.recent(event.unified_msg_origin)[0]["meme_id"] == ref
    assert restarted.recent("default:GroupMessage:b") == []
    assert not restarted.delete("default:GroupMessage:b", ref)["ok"]
    assert files[0].exists()


def test_resized_delivery_matches_original_and_inspection_keeps_send_order(
    setup, tmp_path
):
    client, event, files, _ = setup
    library = client._image_service.meme_library
    delivered = tmp_path / "meme_send_small.png"
    PILImage.new("RGB", (12, 12), "red").save(delivered)
    first = library.remember(event.unified_msg_origin, str(files[0]), str(delivered))
    second = library.remember(event.unified_msg_origin, str(files[1]), str(files[1]))
    assert library.match(event.unified_msg_origin, delivered.read_bytes()) == [first]
    assert library.recent(event.unified_msg_origin)[0]["meme_id"] == second
    assert library.delete(event.unified_msg_origin, first)["ok"]
    assert files[1].exists() and delivered.exists()


@pytest.mark.parametrize("kind", ["outside", "symlink", "missing", "directory"])
def test_library_rejects_paths_outside_active_files(setup, tmp_path, kind):
    client, event, files, _ = setup
    library = client._image_service.meme_library
    outside = tmp_path / "other.png"
    outside.write_bytes(b"keep")
    if kind == "outside":
        target = outside
    elif kind == "symlink":
        target = files[0].parent / "escape.png"
        target.symlink_to(outside)
    elif kind == "missing":
        target = files[0].parent / "missing.png"
    else:
        target = files[0].parent
    assert library.remember(event.unified_msg_origin, str(target)) == ""
    assert outside.read_bytes() == b"keep"


def test_database_failure_restores_file_and_indexes(setup):
    client, event, files, _ = setup
    library = client._image_service.meme_library
    ref = library.remember(event.unified_msg_origin, str(files[0]), str(files[0]))
    with sqlite3.connect(library.db_path) as con:
        con.execute(
            "CREATE TRIGGER fail_delete BEFORE DELETE ON meme_tags BEGIN SELECT RAISE(ABORT, 'test failure'); END"
        )
    original = files[0].read_bytes()
    with pytest.raises(sqlite3.IntegrityError):
        library.delete(event.unified_msg_origin, ref)
    assert files[0].read_bytes() == original
    with sqlite3.connect(library.db_path) as con:
        assert con.execute("SELECT count(*) FROM meme_index").fetchone()[0] == 2
        assert con.execute("SELECT count(*) FROM meme_tags").fetchone()[0] == 2
    assert library.recent(event.unified_msg_origin)[0]["meme_id"] == ref


@pytest.mark.asyncio
async def test_native_tool_binding_lookup_delete_and_cache_refresh(setup):
    client, event, files, tools = setup
    library = client._image_service.meme_library
    library.remember(event.unified_msg_origin, str(files[0]), str(files[0]))
    result = json.loads(
        await client._execute_tool_handler(
            tools["meme_library"].handler, event, {"action": "recent"}
        )
    )
    ref = result["memes"][0]["meme_id"]
    _MEME_VEC_CACHE["test"] = None
    _MEME_TAG_CACHE["test"] = None
    deleted = json.loads(
        await client._execute_agent_tool(
            "meme_library",
            {"action": "delete", "meme_id": ref},
            event,
            {"meme_library"},
        )
    )
    assert deleted["ok"] and not files[0].exists()
    assert "test" not in _MEME_VEC_CACHE and "test" not in _MEME_TAG_CACHE
    assert not json.loads(
        await client._manage_meme_library(event, action="delete", meme_id=ref)
    )["ok"]


@pytest.mark.asyncio
async def test_tool_not_offered_cannot_delete(setup):
    client, event, files, _ = setup
    library = client._image_service.meme_library
    ref = library.remember(event.unified_msg_origin, str(files[0]), str(files[0]))
    result = await client._execute_agent_tool(
        "meme_library", {"action": "delete", "meme_id": ref}, event, set()
    )
    assert "not allowed" in result and files[0].exists()


@pytest.mark.asyncio
async def test_inspect_batch_resolves_chosen_requests_quote_not_anchor(
    setup, monkeypatch
):
    client, event, files, _ = setup
    import astrbot.core.utils.quoted_message as quoted

    client.plugin.state = SimpleNamespace(
        get_reply_target=lambda scope, mid: (
            "bot-sticker-b" if mid == "request-b" else None
        )
    )
    extract = AsyncMock(return_value=[str(files[1])])
    monkeypatch.setattr(quoted, "extract_quoted_message_images", extract)
    event.message_obj.message = [Image(file=str(files[0]))]
    args = {"action": "inspect", "quote_index": 2}
    result = json.loads(
        await client._execute_agent_tool(
            "meme_library",
            args,
            event,
            {"meme_library"},
            quote_candidates=[("request-a", "first"), ("request-b", "delete this")],
            umo=event.unified_msg_origin,
        )
    )
    assert result["ok"] and len(result["meme_ids"]) == 1
    assert extract.await_args.kwargs["reply_component"].id == "bot-sticker-b"
    assert json.loads(
        await client._manage_meme_library(
            event, action="delete", meme_id=result["meme_ids"][0]
        )
    )["ok"]
    assert files[0].exists() and not files[1].exists()


@pytest.mark.asyncio
async def test_quoted_image_without_old_send_history_and_unmatched_quote(
    setup, monkeypatch
):
    client, event, files, _ = setup
    import astrbot.core.utils.quoted_message as quoted

    event.message_obj.message = [Reply(id="old-sticker")]
    extract = AsyncMock(return_value=[str(files[0])])
    monkeypatch.setattr(quoted, "extract_quoted_message_images", extract)
    result = json.loads(await client._manage_meme_library(event, action="inspect"))
    assert result["ok"] and len(result["meme_ids"]) == 1
    assert client._image_service.meme_library.recent(event.unified_msg_origin) == []
    assert all(f.exists() for f in files)
    result = json.loads(
        await client._manage_meme_library(
            event, action="inspect", quote_index=3, quote_candidates=[]
        )
    )
    assert not result["ok"] and all(f.exists() for f in files)


@pytest.mark.asyncio
async def test_unmatched_image_does_not_select_latest_sticker(setup, tmp_path):
    client, event, files, _ = setup
    library = client._image_service.meme_library
    library.remember(event.unified_msg_origin, str(files[0]), str(files[0]))
    other = tmp_path / "unknown.png"
    PILImage.new("RGB", (15, 15), "yellow").save(other)
    event.message_obj.message = [Image(file=str(other))]
    result = json.loads(await client._manage_meme_library(event, action="inspect"))
    assert not result["ok"] and result["meme_ids"] == []
    assert all(f.exists() for f in files)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_send_records_only_successful_original_sticker_before_temp_cleanup(
    setup, tmp_path, failed
):
    client, event, files, _ = setup
    delivered = tmp_path / "meme_send_resized.png"
    PILImage.new("RGB", (12, 12), "red").save(delivered)

    async def pick(*args, selected_meme, **kwargs):
        selected_meme.update(path=str(files[0]), description="original sticker")
        return str(delivered)

    client._pick_meme_image_file = pick
    if failed:
        event.send.side_effect = RuntimeError("send failed")
    result = await client._process_and_send_final_reply(
        json.dumps(
            {"control": {"decision": "reply_only"}, "response": {"text": "[表情:开心]"}}
        ),
        event,
        "at",
        [],
        [],
        [],
        event.unified_msg_origin,
        [],
    )
    recent = client._image_service.meme_library.recent(event.unified_msg_origin)
    assert result is (not failed)
    assert bool(recent) is (not failed)
    assert not delivered.exists()
    assert files[0].exists()
    if not failed:
        assert client._image_service.meme_library.delete(
            event.unified_msg_origin, recent[0]["meme_id"]
        )["ok"]
        assert not files[0].exists()


def test_sticker_tool_available_with_persona_whitelist_and_other_tools_disabled(setup):
    client, event, _, _ = setup
    tools, _ = client._setup_tools(
        event.unified_msg_origin, {"allow_search": False}, {"tools": []}
    )
    assert [tool.name for tool in tools.tools] == ["meme_library"]


def test_original_capture_matches_after_jpeg_conversion_without_visual_guessing(setup):
    """Locate a saved PNG by its capture digest after its bytes became JPEG."""
    import hashlib
    import io

    client, event, files, _ = setup
    library = client._image_service.meme_library
    raw = files[0].read_bytes()
    target = files[0].parent / (hashlib.sha256(raw).hexdigest()[:32] + ".jpg")
    PILImage.open(io.BytesIO(raw)).convert("RGB").save(target, format="JPEG")
    assert target.read_bytes() != raw
    files[0].unlink()
    refs = library.match(event.unified_msg_origin, raw)
    assert len(refs) == 1
    assert library.recent(event.unified_msg_origin) == []
    assert library.match(event.unified_msg_origin, files[1].read_bytes()) != refs
    assert library.delete(event.unified_msg_origin, refs[0])["ok"]
    assert not target.exists() and files[1].exists()


@pytest.mark.asyncio
async def test_recent_receipt_does_not_claim_to_list_all_saved_memes(setup):
    """An empty sent list must not be reported as an empty saved library."""
    client, event, files, _ = setup
    result = json.loads(await client._manage_meme_library(event, action="recent"))
    assert result["ok"] and result["memes"] == []
    assert "不代表未收藏" in result["note"]
    assert "inspect" in result["note"]
    assert files[0].exists()
