"""Verify adaptive group timing, priority, and bounded optional bubbles."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from src.core.group_activity import GroupActivity
from src.core.models import ReplyTask
from src.core.scheduler import Scheduler
from src.core.state import MemChatState
from src.llm_client import LLMClient

UMO = "activity:GroupMessage:42"


class Event:
    def __init__(self, sender="user"):
        self.sender = sender
        self.message_obj = SimpleNamespace(message_id="one", message=[])
        self._mem_chat_wake_reason = "ordinary"

    def get_group_id(self):
        return "42"

    def get_sender_id(self):
        return self.sender

    def get_sender_name(self):
        return "User"

    def get_self_id(self):
        return "bot"

    def get_messages(self):
        return self.message_obj.message


@pytest.fixture
def scheduler():
    config = {
        "activity_scheduling_enabled": True,
        "group_bubble_enabled": False,
        "heartbeat_cycle": 300,
        "decision_msg_count_min": 5,
        "focus_mode_cycle": 30,
        "focus_cycle_min_sec": 8,
        "focus_cycle_max_sec": 120,
        "decision_prompt_template": "Read all messages; choose reply_only or no_reply.",
        "group_frequency_overrides": {
            "42": {"cycle_multiplier": 3.2, "min_message_bonus": 10}
        },
    }
    context = SimpleNamespace(
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value=None),
            get_conversation=AsyncMock(return_value=None),
        )
    )
    state = MemChatState()
    state.set_self_id(UMO, "bot")
    instance = Scheduler(None, context, config, state, None, None)
    instance.get_recent_history = AsyncMock(
        return_value=["[recent] User: model progress"]
    )
    instance.build_memory_context = AsyncMock(return_value=("", ""))
    instance._notify_scene_after_turn = AsyncMock()
    instance._maybe_update_affinity_from_mood = lambda **_: None
    instance.llm_client = SimpleNamespace(invoke_llm=AsyncMock(return_value=True))
    yield instance
    instance.shutdown()


def test_same_traffic_has_same_schedule_regardless_of_membership(scheduler):
    other = "activity:GroupMessage:99"
    scheduler.state.set_group_member_count("42", 900)
    scheduler.state.set_group_member_count("99", 4)
    for scope in [UMO, other]:
        scheduler.group_activity.record(scope, "user", now=100)
    assert scheduler._read_air_wait_seconds(UMO) == scheduler._read_air_wait_seconds(
        other
    )
    assert scheduler._read_air_min_messages(UMO) == 1
    assert scheduler._focus_cycle_bounds(UMO) == scheduler._focus_cycle_bounds(other)
    assert "大群发言频率约束" not in scheduler._large_group_decision_note(UMO)
    assert "不需要每句话都提供新知识" in scheduler._large_group_decision_note(UMO)
    assert scheduler._read_air_wait_seconds("activity:FriendMessage:42") == 300


def test_hot_traffic_slows_observation_and_cools_gradually():
    activity = GroupActivity()
    for i in range(100):
        activity.record(UMO, str(i % 4), now=1000 + i)
    hot = activity.interval(UMO, now=1099)
    assert hot == 180
    after = activity.interval(UMO, now=1221)
    cold = activity.interval(UMO, now=2100)
    assert 45 <= cold < after <= hot


def test_adjacent_fragments_merge_without_losing_arrival_revisions():
    fragments = GroupActivity()
    speakers = GroupActivity()
    for i in range(8):
        fragments.record(UMO, "same", now=100 + i)
        speakers.record(UMO, str(i), now=100 + i)
    one = fragments.snapshot(UMO, now=107)
    many = speakers.snapshot(UMO, now=107)
    assert one["short_rate"] == 0.5
    assert many["short_rate"] == 4
    assert one["revision"] == many["revision"] == 8
    assert one["speakers"] == 1
    assert many["speakers"] == 8
    assert len(fragments.groups[UMO]["rows"]) == 8


def test_same_sender_can_remain_active_without_being_merged_forever():
    activity = GroupActivity()
    for i in range(30):
        activity.record(UMO, "same", now=100 + 2 * i)
    assert activity.snapshot(UMO, now=158)["short_rate"] == 4


def test_activity_windows_are_bounded():
    activity = GroupActivity()
    for i in range(5000):
        activity.record(UMO, str(i % 3), now=100 + i / 100)
    assert len(activity.groups[UMO]["rows"]) <= 4096
    assert activity.snapshot(UMO, now=1000)["speakers"] == 0
    assert not activity.groups[UMO]["rows"]


def test_only_intentional_silence_backs_off(scheduler):
    scheduler.group_activity.record(UMO, "user")
    start = scheduler._read_air_wait_seconds(UMO)
    event = Event()
    task = ReplyTask(
        UMO, 10, "read_air", event=event, extra_args={"attention_version": 0}
    )
    scheduler._finish_group_turn(task, replied=False, sent=False)
    assert scheduler._read_air_wait_seconds(UMO) == pytest.approx(start)
    scheduler._finish_group_turn(task, replied=True, sent=False)
    assert scheduler._read_air_wait_seconds(UMO) > start
    scheduler.group_activity.finish(UMO, sent=True)
    assert scheduler._read_air_wait_seconds(UMO) == pytest.approx(start)


@pytest.mark.asyncio
async def test_one_message_reaches_model_without_minimum_batch_size(scheduler):
    scheduler.state.update_last_message(UMO, "one", "hello", "User", "user")
    scheduler.state.set_message_meta(UMO, "one", {"is_directed": False})
    event = Event()
    scheduler.note_group_activity(UMO, event)
    scheduler.trigger_view_cycle = AsyncMock()
    scheduler.add_task = AsyncMock()
    await scheduler._cycle_wait_and_submit(UMO, "read_air", 0, event, [])
    task = scheduler.add_task.await_args.args[0]
    assert task.task_type == "read_air"
    await scheduler.task_queues[UMO].put(task)
    await scheduler._process_reply_queue(UMO)
    scheduler.llm_client.invoke_llm.assert_awaited_once()
    assert scheduler.state.get_pending_count(UMO) == 0


@pytest.mark.asyncio
async def test_no_pending_messages_does_not_poll_while_cooling(scheduler):
    event = Event()
    scheduler._cycle_events[UMO] = event
    scheduler._enter_read_air(UMO)
    scheduler.trigger_view_cycle = AsyncMock()
    await scheduler._rearm_group_cycle(UMO)
    scheduler.trigger_view_cycle.assert_not_awaited()
    scheduler.llm_client.invoke_llm.assert_not_awaited()


@pytest.mark.asyncio
async def test_arrivals_do_not_postpone_timer_and_direct_calls_override_it(scheduler):
    event = Event()
    scheduler.state.update_last_message(UMO, "one", "hello", "User", "user")
    scheduler.note_group_activity(UMO, event)
    await scheduler.trigger_view_cycle(UMO, "read_air", event, [])
    timer = scheduler.cycle_timers[UMO]
    for i in range(15):
        scheduler.note_group_activity(UMO, Event(str(i)))
        await scheduler.trigger_view_cycle(UMO, "read_air", event, [])
        assert scheduler.cycle_timers[UMO] is timer
    await scheduler.trigger_view_cycle(UMO, "at", event, [])
    assert scheduler.cycle_timers[UMO] is not timer
    assert scheduler.state.get_cycle_task_type(UMO) == "at"


@pytest.mark.asyncio
async def test_focus_cools_linearly_towards_current_activity_interval(
    scheduler, monkeypatch
):
    clock = [1000.0]
    monkeypatch.setattr("src.core.scheduler.time.monotonic", lambda: clock[0])
    scheduler.group_activity.record(UMO, "user", now=1000)
    scheduler._enter_read_air(UMO)
    assert scheduler._read_air_wait_seconds(UMO) == 20
    clock[0] += 150
    middle = scheduler._read_air_wait_seconds(UMO)
    target = scheduler._steady_read_air_wait(UMO)
    assert middle == pytest.approx(20 + (target - 20) / 2)
    clock[0] += 151
    assert scheduler._read_air_wait_seconds(UMO) == scheduler._steady_read_air_wait(UMO)


def prepare_bubble(scheduler, *, age=120):
    scheduler.config["group_bubble_enabled"] = True
    scheduler.group_activity.record(UMO, "user", now=time.monotonic() - age)
    revision = scheduler.group_activity.snapshot(UMO)["revision"]
    event = Event()
    event._mem_chat_group_bubble_revision = revision
    return ReplyTask(
        UMO,
        20,
        "read_air",
        event=event,
        extra_args={"group_bubble": True, "revision": revision},
    )


@pytest.mark.parametrize(
    "block",
    ["stale", "new_input", "recent_send", "focus", "pending", "disabled", "too_soon"],
)
def test_bubbles_require_fresh_quiet_context_and_no_active_exchange(scheduler, block):
    task = prepare_bubble(scheduler)
    revision = task.extra_args["revision"]
    assert scheduler._group_bubble_ready(UMO, revision)
    if block == "stale":
        scheduler.group_activity.groups[UMO]["last_at"] -= 1300
    elif block == "new_input":
        scheduler.group_activity.record(UMO, "other")
    elif block == "recent_send":
        scheduler._last_send_ts[UMO] = time.time() - 100
    elif block == "focus":
        scheduler.state.set_focus(UMO, 100)
    elif block == "pending":
        scheduler.state.update_last_message(UMO, "new", "hello", "User", "user")
    elif block == "disabled":
        scheduler.config["group_bubble_enabled"] = False
    else:
        scheduler.group_activity.groups[UMO]["last_at"] = time.monotonic() - 30
    assert not scheduler._group_bubble_ready(UMO, revision)


@pytest.mark.asyncio
@pytest.mark.parametrize("sent", [True, False])
async def test_bubble_uses_already_read_background_without_new_input_or_focus(
    scheduler, sent
):
    task = prepare_bubble(scheduler)

    async def invoke(*args, **kwargs):
        kwargs["event"]._mem_chat_last_send_ok = sent
        return True

    scheduler.llm_client.invoke_llm.side_effect = invoke
    scheduler.trigger_view_cycle = AsyncMock()
    await scheduler.task_queues[UMO].put(task)
    await scheduler._process_reply_queue(UMO)
    call = scheduler.llm_client.invoke_llm.await_args
    assert "已经读过" in call.args[1]
    assert call.kwargs["user_msgs_to_save"] == []
    assert call.kwargs["impression_mode"] == "off"
    assert scheduler.state.get_pending_count(UMO) == 0
    assert not scheduler.state.is_in_focus(UMO)
    assert bool(scheduler._last_send_ts.get(UMO)) is sent
    assert not scheduler.state.get_last_cycle_cutoff(UMO)


@pytest.mark.asyncio
async def test_bubble_yields_to_new_messages_after_history_read(scheduler):
    task = prepare_bubble(scheduler)

    async def history(*args, **kwargs):
        scheduler.group_activity.record(UMO, "other")
        scheduler.state.update_last_message(UMO, "new", "question", "User", "other")
        return ["old background"]

    scheduler.get_recent_history.side_effect = history
    await scheduler._run_group_bubble(task)
    scheduler.llm_client.invoke_llm.assert_not_awaited()
    assert scheduler.state.get_pending_count(UMO) == 1


@pytest.mark.asyncio
async def test_stale_generated_bubble_never_reaches_sender(scheduler):
    task = prepare_bubble(scheduler)
    scheduler.group_activity.record(UMO, "other")
    client = LLMClient(SimpleNamespace(config={}, scheduler=scheduler))
    result = await client._process_and_send_final_reply(
        '{"control":{"decision":"reply_only"},"response":{"text":"hello"}}',
        task.event,
        "read_air",
        [],
        [],
        [],
        UMO,
        [],
    )
    assert result is False


@pytest.mark.asyncio
async def test_bubble_timer_claims_context_once_and_randomizes_next_opportunity(
    scheduler, monkeypatch
):
    task = prepare_bubble(scheduler)
    scheduler.add_task = AsyncMock()
    sleeps = []

    async def immediate_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("src.core.scheduler.asyncio.sleep", immediate_sleep)
    scheduler._bubble_due[UMO] = time.time() - 1
    await scheduler._wait_group_bubble(UMO)
    scheduler.add_task.assert_awaited_once()
    assert scheduler.add_task.await_args.args[0].extra_args["group_bubble"]
    assert scheduler._bubble_seen[UMO] == task.extra_args["revision"]
    assert 2699 <= scheduler._bubble_due[UMO] - time.time() <= 5400
    await scheduler._wait_group_bubble(UMO)
    scheduler.add_task.assert_awaited_once()


@pytest.mark.asyncio
async def test_inactive_group_does_not_request_a_bubble(scheduler, monkeypatch):
    prepare_bubble(scheduler, age=1300)
    scheduler.add_task = AsyncMock()
    monkeypatch.setattr("src.core.scheduler.asyncio.sleep", AsyncMock())
    await scheduler._wait_group_bubble(UMO)
    scheduler.add_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_arrivals_arm_one_bubble_timer_and_shutdown_cancels_it(scheduler):
    scheduler.config["group_bubble_enabled"] = True
    scheduler.note_group_activity(UMO, Event())
    first = scheduler._bubble_timers[UMO]
    scheduler.note_group_activity(UMO, Event("other"))
    assert scheduler._bubble_timers[UMO] is first
    scheduler.shutdown()
    await asyncio.sleep(0)
    assert first.cancelled()


def test_explicit_activity_pacing_is_scoped_and_limits_focus_acceleration(scheduler):
    other = "activity:GroupMessage:99"
    scheduler.config["group_frequency_overrides"]["42"]["activity_cycle_multiplier"] = (
        1.8
    )
    for scope in (UMO, other):
        scheduler.group_activity.record(scope, "user")
    assert scheduler._steady_read_air_wait(UMO) == pytest.approx(
        scheduler._steady_read_air_wait(other) * 1.8
    )
    assert scheduler._read_air_min_messages(UMO) == 1
    minimum, maximum, _, baseline = scheduler._focus_cycle_bounds(UMO)
    assert minimum == baseline == 54
    assert maximum == 216
    assert scheduler._focus_cycle_bounds(other) == (8, 120, 20, 30)


@pytest.mark.asyncio
@pytest.mark.parametrize("decision_prompt", [True, False])
async def test_unaddressed_focus_can_use_group_decision_prompt(
    scheduler, decision_prompt
):
    scheduler.config["group_frequency_overrides"]["42"][
        "focus_non_at_use_decision_prompt"
    ] = decision_prompt
    scheduler.config["focus_prompt_template"] = "FOCUS_TEMPLATE"
    scheduler.config["decision_prompt_template"] = "DECISION_TEMPLATE"
    scheduler.state.update_last_message(UMO, "one", "hello", "User", "user")
    scheduler.state.set_message_meta(UMO, "one", {"is_directed": False})
    scheduler.state.set_focus(UMO, 100)
    event = Event()
    scheduler.note_group_activity(UMO, event)
    scheduler.trigger_view_cycle = AsyncMock()
    scheduler.add_task = AsyncMock()
    await scheduler._cycle_wait_and_submit(UMO, "focus", 0, event, [])
    task = scheduler.add_task.await_args.args[0]
    await scheduler.task_queues[UMO].put(task)
    await scheduler._process_reply_queue(UMO)
    scheduler.llm_client.invoke_llm.assert_awaited_once()
    prompt = scheduler.llm_client.invoke_llm.await_args.args[1]
    assert ("DECISION_TEMPLATE" in prompt) is decision_prompt
    assert ("FOCUS_TEMPLATE" in prompt) is not decision_prompt


@pytest.mark.asyncio
async def test_activity_pacing_does_not_delay_direct_calls(scheduler):
    scheduler.config["group_frequency_overrides"]["42"]["activity_cycle_multiplier"] = (
        1.8
    )
    scheduler._cycle_wait_and_submit = AsyncMock()
    event = Event()
    scheduler.state.update_last_message(UMO, "one", "hello", "User", "user")
    await scheduler.trigger_view_cycle(UMO, "at", event, [])
    await asyncio.sleep(0)
    assert scheduler._cycle_wait_and_submit.await_args.args[1:3] == ("at", 2.0)
