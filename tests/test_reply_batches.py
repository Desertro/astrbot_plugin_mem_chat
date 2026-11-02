"""Regression tests for bounded group reply windows and multi-message replies."""

import asyncio
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from src.core.models import ReplyTask
from src.core.scheduler import Scheduler
from src.core.state import MemChatState
from src.llm.prompt_builder import build_decision_prompt, build_focus_prompt

from astrbot.api.message_components import Plain

UMO = "audit-batch:GroupMessage:1234567890"


class Event:
    def __init__(self, mid="a", text="first question", *, private=False):
        self.message_obj = SimpleNamespace(message_id=mid, message=[Plain(text)])
        self.message_str = text
        self.unified_msg_origin = UMO
        self.private = private

    def get_group_id(self):
        return "" if self.private else "1234567890"

    def get_sender_id(self):
        return "alice-user"

    def get_sender_name(self):
        return "AliceUser"

    def get_self_id(self):
        return "bot"

    def get_messages(self):
        return self.message_obj.message


@pytest.fixture
def scheduler():
    config = {
        "min_reply_interval_sec": 0,
        "focus_prompt_template": "Read the conversation and use response.segments when useful.",
        "decision_prompt_template": "Choose reply_only or no_reply for the conversation.",
        "stale_pending_max_age_sec": 0,
    }
    state = MemChatState()
    state.set_self_id(UMO, "bot")
    state.set_focus(UMO, 100)
    manager = SimpleNamespace(
        get_curr_conversation_id=AsyncMock(return_value=None),
        get_conversation=AsyncMock(return_value=None),
    )
    instance = Scheduler(
        None, SimpleNamespace(conversation_manager=manager), config, state, None, None
    )
    instance.get_recent_history = AsyncMock(return_value=[])
    instance.build_memory_context = AsyncMock(return_value=("", ""))
    instance.trigger_view_cycle = AsyncMock()
    instance._notify_scene_after_turn = AsyncMock()
    instance._maybe_update_affinity_from_mood = lambda **kwargs: None
    instance._preapply_poke_affinity = lambda **kwargs: None
    instance.llm_client = SimpleNamespace(invoke_llm=AsyncMock())
    return instance


def add_message(state, mid, text, sender="alice-user", *, directed=False):
    state.update_last_message(UMO, mid, text, sender, sender)
    state.set_message_meta(
        UMO,
        mid,
        {
            "is_directed": directed,
            "explicit_target_type": "bot" if directed else "group",
            "inferred_target_type": "bot" if directed else "group",
            "explicit_target_id": "bot" if directed else "group",
        },
    )


@pytest.mark.parametrize("builder", [build_focus_prompt, build_decision_prompt])
def test_batch_prompt_preserves_every_speaker_without_promoting_the_tail(builder):
    lines = [
        "A: [REF:1] @bot question one",
        "B: [REF:2] @bot question two",
        "C: [REF:3] casual chat",
    ]
    prompt = builder(
        UMO,
        {},
        "casual chat",
        lines[-1],
        history_lines=lines,
        target_block="MISLEADING_SINGLE_TARGET",
        batch_mode=True,
    )
    assert all(line in prompt for line in lines)
    assert "【本轮回应安排】" in prompt
    assert "response.segments" in prompt
    assert "no_reply" in prompt
    assert "【需要你优先处理的最新一条】" not in prompt
    assert "MISLEADING_SINGLE_TARGET" not in prompt


def test_private_prompt_keeps_existing_target_semantics():
    prompt = build_focus_prompt("audit:FriendMessage:1", {}, "hello", "A: hello")
    assert "【需要你优先处理的最新一条】" in prompt


def test_consumption_keeps_unseen_rows_even_with_reused_message_ids():
    state = MemChatState(now=lambda: 10.0)
    add_message(state, "same-id", "seen")
    snapshot = state.get_pending_messages(UMO)
    add_message(state, "same-id", "new arrival")
    state.consume_pending_messages(UMO, snapshot)
    assert [row[1] for row in state.get_pending_messages(UMO)] == ["new arrival"]


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["at", "focus", "read_air"])
async def test_late_messages_survive_history_and_generation(scheduler, lane):
    add_message(scheduler.state, "a", "first question", directed=lane == "at")

    async def history(*args, **kwargs):
        add_message(scheduler.state, "during-history", "unseen during history", "b")
        return []

    async def invoke(*args, **kwargs):
        add_message(scheduler.state, "during-model", "unseen during generation", "c")
        kwargs["event"]._mem_chat_last_send_ok = True
        return True

    scheduler.get_recent_history.side_effect = history
    scheduler.llm_client.invoke_llm.side_effect = invoke
    task = ReplyTask(UMO, 0, lane, event=Event())
    await scheduler.task_queues[UMO].put(task)
    await scheduler._process_reply_queue(UMO)
    call = scheduler.llm_client.invoke_llm.call_args
    assert call is not None
    assert [row["_msg_id"] for row in call.kwargs["user_msgs_to_save"]] == ["a"]
    assert "unseen during history" not in call.args[1]
    assert "unseen during generation" not in call.args[1]
    assert [row[0] for row in scheduler.state.get_pending_messages(UMO)] == [
        "during-history",
        "during-model",
    ]


@pytest.mark.asyncio
async def test_at_keeps_original_question_and_other_mentions_in_one_window(scheduler):
    add_message(scheduler.state, "a", "first question", directed=True)
    add_message(scheduler.state, "b", "second question", "second-user", directed=True)
    add_message(scheduler.state, "c", "already bought ggx100", "bystander")

    async def invoke(*args, **kwargs):
        kwargs["event"]._mem_chat_last_send_ok = True
        return True

    scheduler.llm_client.invoke_llm.side_effect = invoke
    await scheduler.task_queues[UMO].put(ReplyTask(UMO, 0, "at", event=Event()))
    await scheduler._process_reply_queue(UMO)
    prompt = scheduler.llm_client.invoke_llm.call_args.args[1]
    assert all(
        text in prompt
        for text in ["first question", "second question", "already bought ggx100"]
    )
    assert "【需要你优先处理的最新一条】" not in prompt
    assert scheduler.build_memory_context.call_args.args[0] == "first question"
    assert scheduler.build_memory_context.call_args.args[4] == "alice-user"
    assert scheduler.state.get_pending_count(UMO) == 0


@pytest.mark.asyncio
async def test_no_reply_consumes_only_the_read_window(scheduler):
    add_message(scheduler.state, "a", "seen")

    async def invoke(*args, **kwargs):
        add_message(scheduler.state, "b", "new question", directed=True)
        kwargs["event"]._mem_chat_last_send_ok = False
        return True

    scheduler.llm_client.invoke_llm.side_effect = invoke
    await scheduler.task_queues[UMO].put(ReplyTask(UMO, 0, "at", event=Event()))
    await scheduler._process_reply_queue(UMO)
    assert [row[0] for row in scheduler.state.get_pending_messages(UMO)] == ["b"]
    assert UMO not in scheduler._last_send_ts


@pytest.mark.asyncio
async def test_failed_batch_is_available_to_the_next_cycle(scheduler):
    add_message(scheduler.state, "a", "first question", directed=True)
    scheduler.llm_client.invoke_llm.return_value = False
    await scheduler.task_queues[UMO].put(ReplyTask(UMO, 0, "at", event=Event()))
    await scheduler._process_reply_queue(UMO)
    assert [row[0] for row in scheduler.state.get_pending_messages(UMO)] == ["a"]
    scheduler.llm_client.invoke_llm.return_value = True
    await scheduler.task_queues[UMO].put(ReplyTask(UMO, 5, "focus", event=Event()))
    await scheduler._process_reply_queue(UMO)
    assert scheduler.llm_client.invoke_llm.call_count == 2
    assert "first question" in scheduler.llm_client.invoke_llm.call_args.args[1]
    assert scheduler.state.get_pending_count(UMO) == 0


@pytest.mark.asyncio
async def test_context_limit_does_not_discard_directed_messages(scheduler):
    scheduler.config["at_pending_max_msgs"] = 2
    add_message(scheduler.state, "a", "first question", directed=True)
    add_message(scheduler.state, "b", "context")
    add_message(scheduler.state, "c", "another context")
    add_message(scheduler.state, "d", "another direct question", directed=True)
    scheduler.llm_client.invoke_llm.return_value = True
    await scheduler.task_queues[UMO].put(ReplyTask(UMO, 0, "at", event=Event()))
    await scheduler._process_reply_queue(UMO)
    call = scheduler.llm_client.invoke_llm.call_args
    assert "another direct question" in call.args[1]
    assert {row["_msg_id"] for row in call.kwargs["user_msgs_to_save"]} == {
        "a",
        "b",
        "d",
    }
    assert [row[0] for row in scheduler.state.get_pending_messages(UMO)] == ["c"]


@pytest.mark.asyncio
async def test_group_timer_has_a_bounded_deadline_and_private_still_debounces():
    for private in [False, True]:
        state = MemChatState()
        instance = Scheduler(None, None, {}, state, None, None)
        blocker = asyncio.Event()
        instance._cycle_wait_and_submit = AsyncMock(
            side_effect=lambda *args: blocker.wait()
        )

        # Use a coroutine function so cancelling timers does not leave inner coroutines.
        async def wait(*args):
            await blocker.wait()

        instance._cycle_wait_and_submit.side_effect = wait
        event = Event(private=private)
        await instance.trigger_view_cycle(UMO, "at", event, [])
        first = instance.cycle_timers[UMO]
        await asyncio.sleep(0)
        await instance.trigger_view_cycle(UMO, "at", event, [])
        second = instance.cycle_timers[UMO]
        assert (first is second) is (not private)
        first.cancel()
        second.cancel()
        for timer in {first, second}:
            with suppress(asyncio.CancelledError):
                await timer


@pytest.mark.asyncio
async def test_at_queue_coalesces_group_events_while_a_cycle_is_running():
    instance = Scheduler(None, None, {}, MemChatState(), None, None)
    instance.active_queues.add(UMO)
    await instance.add_task(ReplyTask(UMO, 0, "at", event=Event("a")))
    await instance.add_task(ReplyTask(UMO, 0, "at", event=Event("b")))
    assert instance.task_queues[UMO].qsize() == 1
    await instance.trigger_view_cycle(UMO, "at", Event("c"), [])
    assert UMO not in instance.cycle_timers


@pytest.mark.asyncio
async def test_deferred_focus_is_armed_after_the_queue_becomes_idle(scheduler):
    add_message(scheduler.state, "a", "first question", directed=True)
    scheduler.trigger_view_cycle = Scheduler.trigger_view_cycle.__get__(scheduler)

    async def invoke(*args, **kwargs):
        add_message(scheduler.state, "b", "next window")
        await scheduler.trigger_view_cycle(UMO, "focus", Event("b"), [])
        kwargs["event"]._mem_chat_last_send_ok = True
        return True

    scheduler.llm_client.invoke_llm.side_effect = invoke
    await scheduler.task_queues[UMO].put(ReplyTask(UMO, 0, "at", event=Event()))
    await scheduler._process_reply_queue(UMO)
    assert UMO not in scheduler.active_queues
    assert scheduler.state.get_cycle_task_type(UMO) == "focus"
    assert UMO in scheduler.cycle_timers
    assert [row[0] for row in scheduler.state.get_pending_messages(UMO)] == ["b"]
    timer = scheduler.cycle_timers[UMO]
    timer.cancel()
    with suppress(asyncio.CancelledError):
        await timer


@pytest.mark.asyncio
async def test_mentions_arriving_during_generation_form_one_following_batch(scheduler):
    add_message(scheduler.state, "a", "first question", directed=True)
    windows = []

    async def invoke(*args, **kwargs):
        windows.append([row["_msg_id"] for row in kwargs["user_msgs_to_save"]])
        if len(windows) == 1:
            add_message(
                scheduler.state, "b", "second question", "second-user", directed=True
            )
            await scheduler.add_task(
                ReplyTask(UMO, 0, "at", event=Event("b", "second question"))
            )
            add_message(
                scheduler.state, "c", "third question", "third-user", directed=True
            )
            await scheduler.add_task(
                ReplyTask(UMO, 0, "at", event=Event("c", "third question"))
            )
        kwargs["event"]._mem_chat_last_send_ok = True
        return True

    scheduler.llm_client.invoke_llm.side_effect = invoke
    await scheduler.task_queues[UMO].put(ReplyTask(UMO, 0, "at", event=Event()))
    await scheduler._process_reply_queue(UMO)
    assert windows == [["a"], ["b", "c"]]
    assert scheduler.state.get_pending_count(UMO) == 0
    assert UMO not in scheduler._queued_group_at
