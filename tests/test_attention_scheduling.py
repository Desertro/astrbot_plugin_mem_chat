"""Attention leases, gradual cooling, and direct-call scheduling regressions."""

import asyncio
import copy
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from src.core.attention import detect_name_call, normalize_focus_control
from src.core.models import ReplyTask
from src.core.scheduler import Scheduler
from src.core.state import MemChatState

from astrbot.api.message_components import Image, Plain, Reply

UMO = "attention:GroupMessage:42"


class Event:
    def __init__(self, mid="a", text="hello"):
        self.message_obj = SimpleNamespace(message_id=mid, message=[Plain(text)])
        self.message_str = text
        self._mem_chat_wake_reason = "platform_at"

    def get_group_id(self):
        return "42"

    def get_messages(self):
        return self.message_obj.message


@pytest.fixture
def scheduler(monkeypatch):
    config = {
        "activity_scheduling_enabled": False,
        "heartbeat_cycle": 300,
        "decision_msg_count_min": 5,
        "focus_mode_cycle": 30,
        "focus_cycle_min_sec": 8,
        "focus_cycle_max_sec": 120,
        "focus_duration_sec": 100,
        "focus_duration_max_sec": 900,
        "focus_duration_step_max_sec": 120,
        "large_group_heartbeat_multiplier": 1.8,
    }
    state = MemChatState()
    instance = Scheduler(None, None, config, state, None, None)
    monkeypatch.setattr(instance, "_wait_focus_expiry", AsyncMock())
    yield instance
    instance.shutdown()


def add_pending(scheduler, mid, *, age=0, directed=False):
    scheduler.state.set_message_meta(UMO, mid, {"is_directed": directed})
    scheduler.state.update_last_message(UMO, mid, "hello", "user", "1")
    row = scheduler.state.pending_msgs[UMO][-1]
    scheduler.state.pending_msgs[UMO][-1] = (*row[:4], time.time() - age)


def turn(scheduler, *, lane="focus", action=None, seconds=90):
    event = Event()
    event._mem_chat_last_structured_payload = {"control": {"decision": "no_reply"}}
    if action:
        event._mem_chat_last_structured_payload["control"]["focus"] = {
            "action": action,
            "seconds": seconds,
        }
    return ReplyTask(
        UMO,
        0,
        lane,
        event=event,
        extra_args={"attention_version": scheduler._attention_versions[UMO]},
    )


@pytest.mark.parametrize(
    "text", ["爱丽丝", "爱丽丝你看看", "怎么办，爱丽丝？", "天童爱丽丝在吗"]
)
def test_name_call_from_current_plain_text(text):
    assert detect_name_call([Plain(text)], ["爱丽丝", "天童爱丽丝"])


@pytest.mark.parametrize(
    "messages",
    [
        [Plain("爱丽丝梦游仙境很好看")],
        [Plain("读《爱丽丝》这本书")],
        [Image(file="爱丽丝.png")],
        [Reply(id="1", message=[Plain("爱丽丝")])],
        [Plain("arisland")],
    ],
)
def test_media_quotes_and_word_fragments_are_not_name_calls(messages):
    assert not detect_name_call(messages, ["爱丽丝", "Aris"])


def test_custom_ascii_name_and_disabled_names():
    assert detect_name_call([Plain("Aris, help")], ["aris"])
    assert not detect_name_call([Plain("爱丽丝")], [])


@pytest.mark.parametrize("seconds", [True, "90", 0, -1, float("inf"), float("nan")])
def test_invalid_extension_is_ignored(seconds):
    assert (
        normalize_focus_control({"focus": {"action": "extend", "seconds": seconds}})
        == {}
    )


def test_extension_limits_and_legacy_control():
    assert normalize_focus_control({"decision": "no_reply"}) == {}
    assert normalize_focus_control({"focus": {"action": "extend"}})["seconds"] == 60
    assert (
        normalize_focus_control({"focus": {"action": "extend", "seconds": 999}})[
            "seconds"
        ]
        == 120
    )


@pytest.mark.parametrize(
    "large, expected, thresholds",
    [(False, [20, 160, 300], [1, 3, 5]), (True, [36, 288, 540], [1, 5, 10])],
)
def test_elapsed_linear_cooling_and_cold_start(
    scheduler, monkeypatch, large, expected, thresholds
):
    if large:
        scheduler.state.set_group_member_count("42", 150)
    assert scheduler._read_air_wait_seconds(UMO) == expected[-1]
    clock = [1000.0]
    monkeypatch.setattr("src.core.scheduler.time.monotonic", lambda: clock[0])
    scheduler._enter_read_air(UMO)
    for elapsed, wait, minimum in zip([0, 150, 300], expected, thresholds):
        clock[0] = 1000.0 + elapsed
        assert scheduler._read_air_wait_seconds(UMO) == wait
        assert scheduler._read_air_min_messages(UMO) == minimum
    clock[0] += 3600
    assert scheduler._read_air_wait_seconds(UMO) == expected[-1]


@pytest.mark.asyncio
async def test_no_reply_can_extend_without_marking_a_send(scheduler):
    scheduler.state.set_focus(UMO, 40)
    scheduler.state.set_focus_delay(UMO, 10)
    scheduler._finish_group_turn(
        turn(scheduler, action="extend"), replied=True, sent=False
    )
    assert 129 <= scheduler.state.get_focus_remaining_sec(UMO) <= 131
    assert UMO not in scheduler._last_send_ts
    assert UMO not in scheduler._read_air_started_at


@pytest.mark.asyncio
async def test_silent_read_air_extension_does_not_add_default_lease(scheduler):
    scheduler._enter_read_air(UMO)
    scheduler._finish_group_turn(
        turn(scheduler, lane="read_air", action="extend"), replied=True, sent=False
    )
    assert 89 <= scheduler.state.get_focus_remaining_sec(UMO) <= 91


@pytest.mark.asyncio
async def test_release_wins_over_auto_focus_and_stale_result(scheduler):
    scheduler.state.set_focus(UMO, 100)
    task = turn(scheduler, action="release")
    scheduler._finish_group_turn(task, replied=True, sent=True)
    assert not scheduler.state.is_in_focus(UMO)
    assert scheduler._read_air_wait_seconds(UMO) == pytest.approx(20, abs=0.01)
    task = turn(scheduler, action="release")
    scheduler.note_attention_call(UMO, Event())
    scheduler._finish_group_turn(task, replied=True, sent=True)
    assert scheduler.state.is_in_focus(UMO)


@pytest.mark.asyncio
async def test_focus_floor_and_maximum_cover_next_large_group_check(scheduler):
    scheduler.state.set_group_member_count("42", 150)
    scheduler.state.set_focus_delay(UMO, 216)
    scheduler._set_group_focus(UMO, 45, Event())
    assert scheduler.state.get_focus_remaining_sec(UMO) >= 230
    scheduler._set_group_focus(UMO, 9999, Event())
    assert 899 <= scheduler.state.get_focus_remaining_sec(UMO) <= 900


@pytest.mark.asyncio
async def test_ordinary_focus_send_cools_without_renewing_deadline(scheduler):
    scheduler.state.set_focus(UMO, 40)
    old_deadline = scheduler.state.focus_until[UMO]
    scheduler._finish_group_turn(turn(scheduler), replied=True, sent=True)
    assert scheduler.state.focus_until[UMO] == old_deadline
    assert scheduler.state.get_focus_delay(UMO) == 50


@pytest.mark.asyncio
async def test_call_rewards_once_and_completion_only_guarantees_default(scheduler):
    scheduler.note_attention_call(UMO, Event())
    scheduler.note_attention_call(UMO, Event())
    scheduler._finish_group_turn(turn(scheduler, lane="at"), replied=True, sent=False)
    assert 99 <= scheduler.state.get_focus_remaining_sec(UMO) <= 101
    assert scheduler.state.get_focus_delay(UMO) == 8


@pytest.mark.asyncio
async def test_cold_below_threshold_keeps_messages_and_rearms(scheduler):
    for n in range(4):
        add_pending(scheduler, str(n))
    scheduler._cycle_events[UMO] = Event()
    scheduler.trigger_view_cycle = AsyncMock()
    # Exercise the actual timer-finally ownership path.
    task = asyncio.create_task(
        scheduler._cycle_wait_and_submit(UMO, "read_air", 0, Event(), [])
    )
    scheduler.cycle_timers[UMO] = task
    await task
    assert scheduler.state.get_pending_count(UMO) == 4
    scheduler.trigger_view_cycle.assert_awaited_once_with(
        UMO, "read_air", scheduler._cycle_events[UMO], []
    )
    add_pending(scheduler, "fifth")
    scheduler.add_task = AsyncMock()
    await scheduler._cycle_wait_and_submit(UMO, "read_air", 0, Event(), [])
    scheduler.add_task.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_cooling_uses_elapsed_time_without_model_then_stops(scheduler):
    scheduler._enter_read_air(UMO)
    scheduler._cycle_events[UMO] = Event()
    scheduler.trigger_view_cycle = AsyncMock()
    scheduler.add_task = AsyncMock()
    await scheduler._cycle_wait_and_submit(UMO, "read_air", 0, Event(), [])
    assert scheduler._read_air_wait_seconds(UMO) == pytest.approx(20, abs=0.01)
    scheduler.add_task.assert_not_awaited()
    scheduler._read_air_started_at[UMO] -= 300
    await scheduler._rearm_group_cycle(UMO)
    scheduler.trigger_view_cycle.assert_not_awaited()


def test_pending_ttl_covers_two_large_group_intervals(scheduler):
    scheduler.state.set_group_member_count("42", 150)
    add_pending(scheduler, "first", age=540)
    add_pending(scheduler, "second", age=1000)
    add_pending(scheduler, "expired", age=1141)
    scheduler._prune_group_pending(UMO)
    assert [row[0] for row in scheduler.state.get_pending_messages(UMO)] == [
        "first",
        "second",
    ]


def test_directed_rows_survive_background_overflow(scheduler):
    add_pending(scheduler, "directed", directed=True)
    for n in range(200):
        add_pending(scheduler, str(n))
    rows = scheduler.state.get_pending_messages(UMO)
    assert rows[0][0] == "directed"
    assert len(rows) == 51
    for n in range(110):
        add_pending(scheduler, f"directed-{n}", directed=True)
    rows = scheduler.state.get_pending_messages(UMO)
    assert len(rows) <= 150
    assert rows[-1][0] == "directed-109"


@pytest.mark.asyncio
async def test_failed_batch_has_two_retries_and_new_input_unblocks(scheduler):
    add_pending(scheduler, "directed", directed=True)
    scheduler._cycle_events[UMO] = Event()
    scheduler.trigger_view_cycle = AsyncMock()
    for _ in range(3):
        scheduler._finish_group_turn(turn(scheduler), replied=False, sent=False)
    await scheduler._rearm_group_cycle(UMO)
    scheduler.trigger_view_cycle.assert_not_awaited()
    add_pending(scheduler, "new")
    await scheduler._rearm_group_cycle(UMO)
    assert scheduler.trigger_view_cycle.await_args.args[1] == "at"


@pytest.mark.asyncio
async def test_exhausted_stage_budget_does_not_become_unlimited(scheduler):
    task = turn(scheduler)
    task.created_at = time.time() - 1000
    awaited = AsyncMock()
    with pytest.raises(asyncio.TimeoutError):
        await scheduler._await_task_stage(task, "invoke_llm", awaited())
    awaited.assert_not_awaited()


@pytest.mark.asyncio
async def test_focus_expiry_runs_independently_of_busy_queue(scheduler, monkeypatch):
    event = Event()
    deadline = time.time() - 1
    scheduler.state.focus_until[UMO] = deadline
    scheduler.active_queues.add(UMO)
    scheduler.trigger_view_cycle = AsyncMock()
    await Scheduler._wait_focus_expiry(scheduler, UMO, deadline, event)
    assert UMO in scheduler._read_air_started_at
    scheduler.trigger_view_cycle.assert_awaited_once_with(UMO, "read_air", event, [])


@pytest.mark.asyncio
async def test_busy_read_air_is_deferred_and_rearmed(scheduler):
    scheduler.active_queues.add(UMO)
    await scheduler.trigger_view_cycle(UMO, "read_air", Event(), [])
    assert scheduler._deferred_cycles[UMO] == "read_air"
    scheduler.active_queues.clear()
    add_pending(scheduler, "new")
    scheduler.trigger_view_cycle = AsyncMock()
    await scheduler._rearm_group_cycle(UMO)
    assert scheduler.trigger_view_cycle.await_args.args[1] == "read_air"


def test_new_schema_and_control_do_not_mutate_input():
    original = {"decision": "no_reply", "focus": {"action": "extend", "seconds": 1000}}
    before = copy.deepcopy(original)
    assert normalize_focus_control(original)["seconds"] == 120
    assert original == before


def test_extremely_large_extension_is_bounded_without_float_overflow():
    result = normalize_focus_control(
        {"focus": {"action": "extend", "seconds": 10**1000}}
    )
    assert result["seconds"] == 120


def test_private_pending_keeps_existing_fifty_row_limit():
    state = MemChatState()
    scope = "attention:FriendMessage:42"
    for index in range(70):
        mid = str(index)
        state.set_message_meta(scope, mid, {"is_directed": True})
        state.update_last_message(scope, mid, "hello", "user", "1")
    assert len(state.get_pending_messages(scope)) == 50
    assert state.get_pending_messages(scope)[0][0] == "20"


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["read_air", "focus"])
async def test_ambient_send_does_not_reopen_focus_or_reset_cooling(scheduler, lane):
    scheduler._enter_read_air(UMO)
    scheduler._read_air_started_at[UMO] -= 150
    started = scheduler._read_air_started_at[UMO]
    scheduler._finish_group_turn(turn(scheduler, lane=lane), replied=True, sent=True)
    assert not scheduler.state.is_in_focus(UMO)
    assert scheduler._read_air_started_at[UMO] == started
    assert scheduler._read_air_wait_seconds(UMO) == pytest.approx(160, abs=0.01)


@pytest.mark.asyncio
async def test_release_while_already_cooling_does_not_restart_short_wait(scheduler):
    scheduler._enter_read_air(UMO)
    scheduler._read_air_started_at[UMO] -= 150
    started = scheduler._read_air_started_at[UMO]
    scheduler._finish_group_turn(
        turn(scheduler, lane="read_air", action="release"), replied=True, sent=True
    )
    assert scheduler._read_air_started_at[UMO] == started


@pytest.mark.asyncio
@pytest.mark.parametrize("already_focused", [False, True])
async def test_name_candidate_does_not_refresh_attention_on_arrival(
    scheduler, already_focused
):
    if already_focused:
        scheduler.state.set_focus(UMO, 40)
    else:
        scheduler._enter_read_air(UMO)
    deadline = scheduler.state.focus_until.get(UMO)
    started = scheduler._read_air_started_at.get(UMO)
    event = Event(text="爱丽丝是对的")
    event._mem_chat_wake_reason = "name_call"
    scheduler.note_attention_call(UMO, event)
    assert scheduler.state.focus_until.get(UMO) == deadline
    assert scheduler._read_air_started_at.get(UMO) == started
    scheduler.trigger_view_cycle = AsyncMock()
    add_pending(scheduler, "name-candidate", directed=True)
    scheduler._cycle_events[UMO] = event
    await scheduler._rearm_group_cycle(UMO)
    assert scheduler.trigger_view_cycle.await_args.args[1] == "at"


@pytest.mark.asyncio
@pytest.mark.parametrize("addressed", [False, None, "true", 1, True])
async def test_name_call_requires_model_confirmation_before_default_focus(
    scheduler, addressed
):
    scheduler._enter_read_air(UMO)
    started = scheduler._read_air_started_at[UMO]
    task = turn(scheduler, lane="at")
    task.extra_args["attention_origin"] = "name_call"
    task.event._mem_chat_last_structured_payload["control"]["addressed"] = addressed
    scheduler._finish_group_turn(task, replied=True, sent=True)
    assert scheduler.state.is_in_focus(UMO) is (addressed is True)
    if addressed is not True:
        assert scheduler._read_air_started_at[UMO] == started


@pytest.mark.asyncio
async def test_name_call_intentional_silence_does_not_start_focus_or_retry(scheduler):
    task = turn(scheduler, lane="at")
    task.extra_args["attention_origin"] = "name_call"
    task.event._mem_chat_last_structured_payload["control"]["addressed"] = True
    scheduler._finish_group_turn(task, replied=True, sent=False)
    assert not scheduler.state.is_in_focus(UMO)
    assert UMO not in scheduler._retry_windows


@pytest.mark.asyncio
async def test_ambient_reply_can_explicitly_extend_focus(scheduler):
    scheduler._enter_read_air(UMO)
    scheduler._finish_group_turn(
        turn(scheduler, lane="read_air", action="extend"), replied=True, sent=True
    )
    assert 89 <= scheduler.state.get_focus_remaining_sec(UMO) <= 91


def test_large_override_reaches_steady_wait_and_threshold_in_five_minutes(scheduler):
    scheduler.config["group_frequency_overrides"] = {
        "42": {"cycle_multiplier": 3.2, "min_message_bonus": 10}
    }
    scheduler._enter_read_air(UMO)
    scheduler._read_air_started_at[UMO] -= 300
    assert scheduler._read_air_wait_seconds(UMO) == 960
    assert scheduler._read_air_min_messages(UMO) == 15
