import json
from pathlib import Path

from src.core.scheduler import Scheduler
from src.core.state import MemChatState

ROOT = Path(__file__).resolve().parents[1]


def make_scheduler(**overrides):
    config = {
        "heartbeat_cycle": 300,
        "decision_msg_count_min": 5,
        "decision_msg_count_max": 200,
        "focus_mode_cycle": 15,
        "focus_cycle_min_sec": 8,
        "focus_cycle_max_sec": 120,
        "focus_duration_sec": 100,
        "large_group_frequency_enabled": True,
        "large_group_member_threshold": 100,
        "large_group_heartbeat_multiplier": 1.8,
        "large_group_min_message_bonus": 5,
        "large_group_focus_duration_cap_sec": 45,
    }
    config.update(overrides)
    state = MemChatState()
    return Scheduler(None, None, config, state, None, None), state


def test_large_group_slows_read_air_and_raises_message_threshold():
    scheduler, state = make_scheduler()
    state.set_group_member_count("100", 150)
    umo = "aiocqhttp:GroupMessage:100"

    assert scheduler._read_air_wait_seconds(umo) == 540
    assert scheduler._read_air_min_messages(umo) == 10
    assert scheduler._cap_focus_duration(umo, 100) == 100
    assert scheduler._large_group_decision_note(umo)

    cycle_min, cycle_max, _step, base = scheduler._focus_cycle_bounds(umo)
    assert cycle_min == 8 * 1.8
    assert cycle_max == 120 * 1.8
    assert base == 15 * 1.8


def test_small_group_and_private_keep_existing_frequency():
    scheduler, state = make_scheduler()
    state.set_group_member_count("100", 50)

    assert scheduler._read_air_wait_seconds("aiocqhttp:GroupMessage:100") == 300
    assert scheduler._read_air_min_messages("aiocqhttp:GroupMessage:100") == 5
    assert scheduler._read_air_wait_seconds("aiocqhttp:FriendMessage:42") == 300
    assert scheduler._large_group_decision_note("aiocqhttp:GroupMessage:100") == ""


def test_large_group_settings_can_be_disabled():
    scheduler, state = make_scheduler(large_group_frequency_enabled=False)
    state.set_group_member_count("100", 500)
    umo = "aiocqhttp:GroupMessage:100"

    assert scheduler._read_air_wait_seconds(umo) == 300
    assert scheduler._read_air_min_messages(umo) == 5
    assert scheduler._cap_focus_duration(umo, 100) == 100


def test_schema_declares_codex_and_large_group_settings():
    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8-sig"))

    for key in (
        "codex_enabled",
        "codex_cli_path",
        "codex_workdir",
        "codex_model",
        "codex_sandbox",
        "codex_timeout_sec",
        "codex_allowed_uids",
        "codex_allow_groups",
        "large_group_frequency_enabled",
        "large_group_member_threshold",
        "large_group_heartbeat_multiplier",
        "large_group_min_message_bonus",
        "large_group_focus_duration_cap_sec",
    ):
        assert key in schema
