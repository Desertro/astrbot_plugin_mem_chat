import asyncio
import logging
import os
import re
import time
from collections import defaultdict

from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import At, AtAll, Face, Plain, Reply
from astrbot.api.star import Context

from ..db.database import DatabaseManager
from ..llm.prompt_builder import build_decision_prompt, build_focus_prompt
from ..llm_client import LLMClient
from .models import ReplyTask
from .state import MemChatState

logger = logging.getLogger("astrbot")


class Scheduler:
    def __init__(
        self,
        plugin,
        context: Context,
        config: dict,
        state: MemChatState,
        db_mgr: DatabaseManager,
        llm_client: LLMClient,
    ):
        self.plugin = plugin  # Reference to MemChat instance if needed (e.g. for callbacks or other properties)
        self.context = context
        self.config = config
        self.state = state
        self.db_mgr = db_mgr
        self.llm_client = llm_client

        self.task_queues = defaultdict(asyncio.PriorityQueue)
        self.active_queues = set()
        self.cycle_timers = {}  # umo -> asyncio.TimerHandle (or Task)
        self._last_send_ts = {}

    import random

    async def trigger_view_cycle(
        self, umo: str, cycle_type: str, event: AstrMessageEvent, image_urls: list
    ):
        """
        Trigger a view cycle.
        - AT: Wait 1-3s, then submit task.
        - FOCUS: Wait X seconds (dynamic), then submit task.
        - READ_AIR: Wait X seconds (config), then submit task.
        """

        curr_type = self.state.get_cycle_task_type(umo)

        # Priority mapping: smaller is higher
        priority = {"at": 0, "focus": 5, "read_air": 10}
        new_prio = priority.get(cycle_type, 10)
        curr_prio = priority.get(curr_type, 99)  # 99 if None

        if cycle_type != "at" and (
            self.state.is_processing(umo) or not self.task_queues[umo].empty()
        ):
            logger.debug(
                f"[mem_chat] Skip triggering {cycle_type} for {umo} as it is currently busy."
            )
            return
        if self.state.is_processing(umo) and cycle_type == "at":
            logger.info(
                f"[mem_chat] AT message received while processing for {umo}. Proceeding to trigger AT cycle (Interrupt flag expected)."
            )

        should_start = False
        if curr_type is None:
            should_start = True
        elif new_prio < curr_prio:
            # Higher priority overrides lower priority
            should_start = True
            if umo in self.cycle_timers:
                try:
                    self.cycle_timers[umo].cancel()
                    logger.info(
                        f"[mem_chat] Overriding {curr_type} cycle with {cycle_type} for {umo} (Higher Priority)."
                    )
                except Exception:
                    pass
        elif new_prio == curr_prio:
            if cycle_type == "at":
                # AT: Always restart (Debounce)
                should_start = True
                if umo in self.cycle_timers:
                    try:
                        self.cycle_timers[umo].cancel()
                    except Exception:
                        pass
            elif cycle_type == "focus":
                # FOCUS: Only restart if this was triggered by an AT message (to apply speed-up/reward)
                is_at_msg = False
                for msg in event.get_messages() or []:
                    if isinstance(msg, At | AtAll):
                        is_at_msg = True
                        break

                if is_at_msg:
                    should_start = True
                    if umo in self.cycle_timers:
                        try:
                            self.cycle_timers[umo].cancel()
                            logger.info(
                                f"[mem_chat] Restarting FOCUS cycle for {umo} due to AT (Apply Reward)."
                            )
                        except Exception:
                            pass
                else:
                    # Non-AT message in focus mode: just let the current timer run
                    pass
            else:
                # read_air: already running, do nothing
                pass

        if not should_start:
            return

        if cycle_type == "at":
            # Start At Cycle
            self.state.set_cycle_task_type(umo, "at")
            # Random delay 3-5s
            delay = 3.0 + self.random.random() * 2.0
            self.cycle_timers[umo] = asyncio.create_task(
                self._cycle_wait_and_submit(umo, "at", delay, event, image_urls)
            )
            logger.info(f"[mem_chat] Triggered AT cycle for {umo} (Wait {delay:.2f}s).")

        elif cycle_type == "focus":
            # Start Focus Cycle
            self.state.set_cycle_task_type(umo, "focus")
            # Use dynamic delay based on @at counts, baseline is focus_mode_cycle
            base_delay = float(self._get_conf("focus_mode_cycle", 15))
            cycle_delay = self.state.get_focus_delay(umo, default=base_delay)
            try:
                last_send_ts = float(self._last_send_ts.get(umo, 0.0) or 0.0)
            except Exception:
                last_send_ts = 0.0
            if last_send_ts > 0:
                cycle_delay = max(0.0, (last_send_ts + float(cycle_delay)) - time.time())
            self.cycle_timers[umo] = asyncio.create_task(
                self._cycle_wait_and_submit(
                    umo, "focus", float(cycle_delay), event, image_urls
                )
            )
            logger.info(
                f"[mem_chat] Triggered FOCUS cycle for {umo} (Wait {cycle_delay:.2f}s)."
            )

        elif cycle_type == "read_air":
            # Start Read Air Cycle
            self.state.set_cycle_task_type(umo, "read_air")
            cycle_delay = self._get_conf("heartbeat_cycle", 60)
            if cycle_delay < 5:
                cycle_delay = 5
            try:
                if not self.state.get_heartbeat_start(umo):
                    self.state.set_heartbeat_start(umo)
            except Exception:
                pass
            self.cycle_timers[umo] = asyncio.create_task(
                self._cycle_wait_and_submit(
                    umo, "read_air", float(cycle_delay), event, image_urls
                )
            )
            logger.info(
                f"[mem_chat] Triggered READ_AIR cycle for {umo} (Wait {cycle_delay:.2f}s)."
            )

    async def _cycle_wait_and_submit(
        self, umo: str, task_type: str, delay: float, event, image_urls
    ):
        restarted = False
        try:
            await asyncio.sleep(delay)

            # LOGIC CHECKS BEFORE SUBMISSION
            if task_type == "read_air":
                # Check message counts
                min_msgs = int(self._get_conf("decision_msg_count_min", 1))
                max_msgs = int(self._get_conf("decision_msg_count_max", 100))
                if max_msgs < min_msgs:
                    max_msgs = min_msgs
                curr_count = self.state.get_read_air_msg_count(umo)

                if curr_count < min_msgs or curr_count > max_msgs:
                    group_id = umo.split(":")[-1] if ":" in umo else umo
                    logger.debug(
                        f"[mem_chat] Read Air (Group: {group_id}): Message count {curr_count} not in [{min_msgs}, {max_msgs}]. Skipping."
                    )

                    try:
                        self.state.reset_read_air_msg_count(umo)
                        self.state.clear_heartbeat_start(umo)
                    except Exception:
                        pass
                    return

            elif task_type == "focus":
                # Check if still in focus
                if not self.state.is_in_focus(umo):
                    logger.info(
                        f"[mem_chat] Focus mode expired for {umo}. Switching to Read Air."
                    )
                    self.state.set_cycle_task_type(umo, None)
                    base_delay = float(self._get_conf("focus_mode_cycle", 15))
                    self.state.reset_focus_delay(
                        umo, default=base_delay
                    )  # Reset delay for next time
                    await self.trigger_view_cycle(umo, "read_air", event, [])
                    restarted = True
                    return

            # Create Task
            prio = 10
            if task_type == "at":
                prio = 0
            elif task_type == "focus":
                prio = 5
            elif task_type == "read_air":
                prio = 10

            task = ReplyTask(
                task_type=task_type,
                umo=umo,
                event=event,
                priority=prio,
                image_urls=image_urls,
            )

            await self.add_task(task)
            if task_type == "read_air":
                try:
                    self.state.reset_read_air_msg_count(umo)
                    self.state.clear_heartbeat_start(umo)
                except Exception:
                    pass

        except asyncio.CancelledError:
            logger.debug(f"[mem_chat] Cycle {task_type} for {umo} cancelled.")
        finally:
            if not restarted:
                # Only clear if we didn't restart already
                # Only pop and clear if it is THIS task (avoid removing the new task if restarted/overridden)
                current = asyncio.current_task()
                if self.cycle_timers.get(umo) == current:
                    if self.state.get_cycle_task_type(umo) == task_type:
                        self.state.set_cycle_task_type(umo, None)
                    self.cycle_timers.pop(umo, None)

    def _get_full_text(self, event: AstrMessageEvent) -> str:
        try:
            chain = event.message_obj.message
            if hasattr(chain, "chain"):  # It is a MessageChain
                parts = []
                for i in chain:
                    if isinstance(i, Plain):
                        parts.append(i.text)
                    elif isinstance(i, At):
                        parts.append(f"@{i.qq}")
                    elif isinstance(i, AtAll):
                        parts.append("@全体成员")
                    elif isinstance(i, Face):
                        parts.append("[表情]")
                    elif isinstance(i, Reply):
                        # Format: [回复 昵称: 内容]
                        reply_text = i.message_str or "[引用消息]"
                        parts.append(
                            f"[回复 {i.sender_nickname or '未知'}: {reply_text}] "
                        )
                    else:
                        pass
                return "".join(parts).strip()
        except Exception:
            pass
        return event.message_str or ""

    def _get_conf(self, key: str, default):
        try:
            if isinstance(self.config, dict):
                v = self.config.get(key, default)
            else:
                v = getattr(self.config, key, default)
        except Exception:
            v = default
        return v

    def _get_task_timeout_sec(self, task_type: str) -> float:
        mapping = {
            "focus": "task_timeout_sec_focus",
            "read_air": "task_timeout_sec_read_air",
        }
        k = mapping.get(str(task_type or "").strip(), "task_timeout_sec_focus")
        try:
            v = float(self._get_conf(k, 60))
            if v <= 0:
                return 0.0
            return v
        except Exception:
            return 60.0

    def _strip_old_decision_blocks(self, text: str) -> str:
        try:
            return re.sub(r"【决策】[\s\S]*?(?=\n【|\Z)", "", text).strip()
        except Exception:
            return text

    async def add_task(self, task: ReplyTask):
        await self.task_queues[task.umo].put(task)
        if task.umo not in self.active_queues:
            asyncio.create_task(self._process_reply_queue(task.umo))

    async def _process_reply_queue(self, umo):
        if umo in self.active_queues:
            return
        self.active_queues.add(umo)
        logger.info(f"[mem_chat] Queue processor started for {umo}")

        try:
            while True:
                queue = self.task_queues[umo]
                if queue.empty():
                    break

                task: ReplyTask = await queue.get()
                try:
                    # Check timeout
                    timeout_sec = self._get_task_timeout_sec(task.task_type)
                    if timeout_sec > 0 and time.time() - task.created_at > timeout_sec:
                        logger.warning(
                            f"[mem_chat] Task {task.task_type} for {umo} dropped due to timeout (waited {time.time() - task.created_at:.2f}s, timeout={timeout_sec:.0f}s).",
                        )
                        continue

                    logger.info(
                        f"[mem_chat] Processing task type={task.task_type} priority={task.priority}"
                    )

                    try:
                        min_interval = self._get_conf("min_reply_interval_sec", None)
                        if min_interval is None:
                            min_interval = self._get_conf("focus_mode_cycle_min", 0)
                        min_interval = float(min_interval or 0)
                    except Exception:
                        min_interval = 0.0
                    if min_interval > 0:
                        prev_ts = float(self._last_send_ts.get(umo, 0.0) or 0.0)
                        if prev_ts > 0:
                            delta = time.time() - prev_ts
                            if delta < min_interval:
                                await asyncio.sleep(max(0.0, min_interval - delta))
                                if (
                                    timeout_sec > 0
                                    and time.time() - task.created_at > timeout_sec
                                ):
                                    logger.warning(
                                        f"[mem_chat] Task {task.task_type} for {umo} dropped due to timeout after waiting min_interval (waited {time.time() - task.created_at:.2f}s, timeout={timeout_sec:.0f}s).",
                                    )
                                    continue

                    # Re-fetch context at execution time
                    history_count = int(self._get_conf("history_message_count", 12))
                    history_msgs = await self.get_recent_history(
                        umo, history_count, task.event
                    )

                    prompt = ""
                    memory_context = ""

                    # Ensure conv_mgr is available
                    conv_mgr = self.context.conversation_manager

                    if task.task_type == "at":
                        # We need access to the original 'event' to build prompt
                        if not task.event:
                            continue

                        # Fetch pending messages to save and include in context
                        # Consuming pending messages ensures Heartbeat doesn't process them again
                        raw_pending = self.state.get_pending_messages(umo)
                        max_at_msgs = int(self._get_conf("at_pending_max_msgs", 30))
                        if max_at_msgs > 0 and len(raw_pending) > max_at_msgs:
                            try:
                                cutoff_ts = raw_pending[-max_at_msgs][4]
                                raw_pending = self.state.prune_pending_messages(
                                    umo, cutoff_ts=cutoff_ts, keep_last=max_at_msgs
                                )
                            except Exception:
                                raw_pending = raw_pending[-max_at_msgs:]
                        user_msgs_to_save = []
                        quote_candidates = []
                        at_candidates = []
                        pend_lines = []

                        consumed_count = 0
                        if raw_pending:
                            cached_self_id = self.state.get_self_id(umo)
                            for i, (_mid, t, s, sid, _ts) in enumerate(raw_pending):
                                consumed_count += 1
                                if t:
                                    role = (
                                        "assistant"
                                        if str(sid) == str(cached_self_id)
                                        else "user"
                                    )
                                    user_msgs_to_save.append(
                                        {
                                            "role": role,
                                            "content": t,
                                            "sender_name": s or "User",
                                            "sender_id": sid,
                                        },
                                    )
                                    ts_str = time.strftime("%H:%M", time.localtime(_ts))
                                    line_content = f"{s}/{sid}: {t}".strip()
                                    if _mid:
                                        quote_candidates.append((_mid, t, sid))
                                        at_candidates.append(sid)
                                        pend_lines.append(
                                            f"[{ts_str}] [REF:{len(quote_candidates)}] {line_content}",
                                        )
                                    else:
                                        pend_lines.append(f"[{ts_str}] {line_content}")

                            self.state.set_last_processed(umo, time.time())
                        else:
                            # If raw_pending is empty, it means another task (processed while this one was in queue)
                            # has already consumed the messages.
                            # We should SKIP this task to avoid double-replying or replying to empty context.
                            logger.info(
                                "[mem_chat] AT task found no pending messages (consumed by previous task). Skipping."
                            )
                            continue

                        # Re-calculate context
                        gid = task.event.get_group_id() or task.event.get_sender_id()
                        sender_id = task.event.get_sender_id()
                        sender_name = task.event.get_sender_name()
                        msg_text = self._get_full_text(task.event)

                        conversation = await self._get_conversation(task.event)

                        # Re-build memory context
                        memory_context = await self.build_memory_context(
                            msg_text,
                            gid,
                            sender_name,
                            history_msgs,
                            sender_id,
                            conversation,
                        )

                        # Use last message from pending if available, otherwise event message
                        last_msg = msg_text
                        if user_msgs_to_save:
                            last_msg = user_msgs_to_save[-1]["content"]

                        reply_line = f"{sender_name}/{sender_id}: {last_msg}"

                        prompt = build_focus_prompt(
                            umo,
                            self.config,
                            last_msg,
                            reply_line,
                            history_lines=pend_lines,
                        )
                        try:
                            if history_msgs:
                                hist_lines = []
                                for ln in history_msgs[-8:]:
                                    s = str(ln or "").strip()
                                    if not s:
                                        continue
                                    if len(s) > 260:
                                        s = s[:260].rstrip() + "…"
                                    hist_lines.append(s)
                                if hist_lines:
                                    prompt = (
                                        "【历史对话】\n"
                                        + "\n".join(hist_lines)
                                        + "\n\n"
                                        + prompt
                                    )
                        except Exception:
                            pass

                        instruction = ""
                        self_id = self.state.get_self_id(umo)
                        if self_id:
                            self_name = self.state.get_self_name(umo)
                            if self_name:
                                instruction += f"你的ID是: {self_id}/{self_name}"
                            else:
                                instruction += f"你的ID是: {self_id}"

                        self.state.set_processing(umo, True)
                        try:
                            replied = await self.llm_client.invoke_llm(
                                umo,
                                prompt,
                                memory_context,
                                event=task.event,
                                task_type="at",
                                instruction_text=instruction,
                                user_msgs_to_save=user_msgs_to_save,
                                image_urls=task.image_urls,
                                created_at=task.created_at,
                                quote_candidates=quote_candidates,
                                at_candidates=at_candidates,
                            )
                        finally:
                            self.state.set_processing(umo, False)

                        focus_duration = int(self._get_conf("focus_duration_sec", 100))
                        self.state.set_focus(umo, focus_duration)
                        if replied:
                            if consumed_count > 0:
                                self.state.clear_pending_first_n(umo, consumed_count)
                            send_ts = time.time()
                            self._last_send_ts[umo] = send_ts
                            self.state.clear_pending_up_to(umo, send_ts)
                            logger.info(
                                f"[mem_chat] AT phase finished for {umo}. Refreshed Focus Mode (Duration: {focus_duration}s)."
                            )
                        else:
                            logger.info(
                                f"[mem_chat] AT phase failed/interrupted for {umo}. Focus Mode still active. Messages retained in pending."
                            )

                        if replied and self.state.is_in_focus(umo):
                            await self.trigger_view_cycle(umo, "focus", task.event, [])

                    elif task.task_type == "focus" or task.task_type == "read_air":
                        raw_pending = self.state.get_pending_messages(umo)
                        if not raw_pending:
                            logger.info(
                                f"[mem_chat] {task.task_type} task found no pending messages for {umo}. Skipping."
                            )
                            continue
                        now_ts = time.time()
                        stale_sec_key = f"stale_pending_max_age_sec_{task.task_type}"
                        stale_sec = float(
                            self._get_conf(
                                stale_sec_key,
                                self._get_conf("stale_pending_max_age_sec", 180),
                            )
                        )
                        if stale_sec > 0:
                            cutoff = now_ts - stale_sec
                            if raw_pending and raw_pending[-1][4] < cutoff:
                                logger.info(
                                    f"[mem_chat] {task.task_type} pending messages are stale for {umo}. Dropping."
                                )
                                self.state.clear_pending_messages(umo)
                                self.state.set_last_processed(umo, now_ts)
                                continue
                            try:
                                raw_pending = self.state.prune_pending_messages(
                                    umo, cutoff_ts=cutoff, keep_last=1
                                )
                            except Exception:
                                raw_pending = [
                                    m for m in raw_pending if m[4] >= cutoff
                                ] or raw_pending[-1:]
                            if not raw_pending:
                                logger.info(
                                    f"[mem_chat] {task.task_type} task has no pending messages after stale pruning for {umo}. Skipping."
                                )
                                continue
                        if task.task_type == "read_air":
                            max_msgs = int(self._get_conf("decision_msg_count_max", 23))
                            if max_msgs > 0 and len(raw_pending) > max_msgs:
                                raw_pending = raw_pending[-max_msgs:]
                        elif task.task_type == "focus":
                            max_focus_msgs = int(
                                self._get_conf("focus_pending_max_msgs", 25)
                            )
                            if max_focus_msgs > 0 and len(raw_pending) > max_focus_msgs:
                                try:
                                    cutoff_ts = raw_pending[-max_focus_msgs][4]
                                    raw_pending = self.state.prune_pending_messages(
                                        umo,
                                        cutoff_ts=cutoff_ts,
                                        keep_last=max_focus_msgs,
                                    )
                                except Exception:
                                    raw_pending = raw_pending[-max_focus_msgs:]

                        # Guard: For READ_AIR, skip if pending contains AT message (交给 AT 任务处理)。
                        # 对于 FOCUS，允许在专注期间处理历史中带 @ 的消息。
                        if task.task_type == "read_air":
                            try:
                                cached_self_id = self.state.get_self_id(umo)
                                if cached_self_id:
                                    has_at = False
                                    for _mid, _t, _s, _sid, _ts in raw_pending:
                                        if (
                                            f"[At:{cached_self_id}]" in _t
                                            or f"@{cached_self_id}" in _t
                                        ):
                                            has_at = True
                                            break
                                    if has_at:
                                        logger.info(
                                            f"[mem_chat] {task.task_type} pending contains AT message. Skipping to avoid conflict with AT task."
                                        )
                                        continue
                            except Exception as e:
                                logger.error(
                                    f"[mem_chat] Error in AT guard for {task.task_type}: {e}"
                                )

                        # Guard: skip if pending contains only assistant/self messages
                        try:
                            cached_self_id = self.state.get_self_id(umo)
                            only_self_msgs = True if cached_self_id else False
                            if cached_self_id:
                                for _mid, _t, _s, _sid, _ts in raw_pending:
                                    if str(_sid) != str(cached_self_id):
                                        only_self_msgs = False
                                        break
                            if only_self_msgs:
                                logger.info(
                                    f"[mem_chat] {task.task_type} pending contains only assistant messages. Skipping."
                                )
                                # Clear to avoid repeated triggering
                                self.state.clear_pending_messages(umo)
                                self.state.set_last_processed(umo, time.time())
                                continue
                        except Exception:
                            pass

                        if task.task_type == "focus":
                            try:
                                last_send_ts = float(self._last_send_ts.get(umo, 0.0) or 0.0)
                            except Exception:
                                last_send_ts = 0.0
                            if last_send_ts > 0 and raw_pending[-1][4] < last_send_ts:
                                self.state.clear_pending_up_to(umo, last_send_ts)
                                self.state.set_last_processed(umo, time.time())
                                continue

                        # Consume pending
                        pend_lines = [
                            f"{s}/{sid}: {t}".strip()
                            for (_mid, t, s, sid, _ts) in raw_pending
                            if t
                        ]

                        # Prepare saving
                        user_msgs_to_save = []
                        for _mid, t, s, sid, _ts in raw_pending:
                            if t:
                                cached_self_id = self.state.get_self_id(umo)
                                role = (
                                    "assistant"
                                    if str(sid) == str(cached_self_id)
                                    else "user"
                                )
                                user_msgs_to_save.append(
                                    {
                                        "role": role,
                                        "content": t,
                                        "sender_name": s or "User",
                                        "sender_id": sid,
                                    }
                                )

                        # DO NOT clear pending here. We'll clear after LLM success.
                        # self.state.clear_pending_messages(umo)
                        self.state.set_last_processed(umo, time.time())

                        # Build Prompt
                        # Get last message from pending
                        last_msg_info = raw_pending[-1]
                        last_msg = last_msg_info[1]
                        sname = last_msg_info[2]
                        sid = last_msg_info[3]

                        reply_line = f"{sname}/{sid}: {last_msg}"

                        gid = umo.split(":")[-1] if ":" in umo else ""  # Approximate

                        # Context
                        cid = await conv_mgr.get_curr_conversation_id(umo)
                        conversation = (
                            await conv_mgr.get_conversation(umo, cid) if cid else None
                        )
                        memory_context = await self.build_memory_context(
                            last_msg, gid, sname, history_msgs, sid, conversation
                        )

                        # Prepare quote candidates for Focus mode
                        quote_candidates = []
                        # Separate list for prompt to include [REF:index]
                        prompt_lines = []
                        # Mapping for At candidates: index -> sender_id
                        at_candidates = []

                        for i, (_mid, t, s, sid, _ts) in enumerate(raw_pending):
                            line_content = f"{s}/{sid}: {t}".strip()
                            ts_str = time.strftime("%H:%M", time.localtime(_ts))
                            if _mid:
                                quote_candidates.append((_mid, t, sid))
                                at_candidates.append(sid)
                                prompt_lines.append(
                                    f"[{ts_str}] [REF:{len(quote_candidates)}] {line_content}"
                                )
                            else:
                                prompt_lines.append(f"[{ts_str}] {line_content}")

                        if task.task_type == "read_air":
                            # 若最后一条消息是明确 @ 机器人，则不进入读空气判定，避免与显式 AT 冲突
                            self_id = self.state.get_self_id(umo)
                            if self_id and f"@{self_id}" in last_msg:
                                logger.info(
                                    f"[mem_chat] Skip read_air decision because last message explicitly @self for {umo}"
                                )
                                continue
                            prompt = build_decision_prompt(
                                umo,
                                self.config,
                                last_msg,
                                reply_line,
                                history_lines=prompt_lines,
                            )
                            hb_ts = self.state.get_heartbeat_start(umo)
                            if hb_ts:
                                ts_str = time.strftime(
                                    "%Y-%m-%d %H:%M:%S", time.localtime(hb_ts)
                                )
                                prompt = f"【心跳开始】{ts_str}\n" + prompt
                        else:
                            prompt = build_focus_prompt(
                                umo,
                                self.config,
                                last_msg,
                                reply_line,
                                history_lines=prompt_lines,
                            )
                            fs_ts = self.state.get_focus_start(umo)
                            if fs_ts:
                                ts_str = time.strftime(
                                    "%Y-%m-%d %H:%M:%S", time.localtime(fs_ts)
                                )
                                prompt = f"【专注开始】{ts_str}\n" + prompt

                        prompt = self._strip_old_decision_blocks(prompt)

                        self_id = self.state.get_self_id(umo)
                        if self_id:
                            self_name = self.state.get_self_name(umo)
                            if self_name:
                                prompt += f"\n你的ID是: {self_id}/{self_name}"
                            else:
                                prompt += f"\n你的ID是: {self_id}"

                        self.state.set_processing(umo, True)
                        try:
                            replied = await self.llm_client.invoke_llm(
                                umo,
                                prompt,
                                memory_context,
                                event=task.event,
                                task_type=task.task_type,
                                instruction_text="",
                                user_msgs_to_save=user_msgs_to_save,
                                image_urls=task.image_urls,
                                quote_candidates=quote_candidates,
                                at_candidates=at_candidates,
                                created_at=task.created_at,
                            )
                        finally:
                            self.state.set_processing(umo, False)

                        # Update last send time for dedup window
                        if replied:
                            consumed_count = len(raw_pending)
                            if consumed_count > 0:
                                self.state.clear_pending_first_n(umo, consumed_count)
                            send_ts = time.time()
                            self._last_send_ts[umo] = send_ts
                            self.state.clear_pending_up_to(umo, send_ts)
                            logger.info(
                                f"[mem_chat] {task.task_type.upper()} phase finished for {umo}."
                            )
                        else:
                            logger.info(
                                f"[mem_chat] {task.task_type.upper()} task failed/interrupted for {umo}. Messages retained in pending."
                            )

                        # If read_air mode and replied, enter focus mode
                        if task.task_type == "read_air" and replied:
                            focus_duration = int(
                                self._get_conf("focus_duration_sec", 100)
                            )
                            self.state.set_focus(
                                umo, focus_duration
                            )  # This sets state to FOCUS
                            logger.info(
                                f"[mem_chat] Read Air triggered reply. Entering Focus Mode (Duration: {focus_duration}s)."
                            )

                        # If focus mode task completed, apply cooling
                        if replied:
                            base_delay = float(self._get_conf("focus_mode_cycle", 15))
                            min_delay = float(
                                self._get_conf("focus_mode_cycle_min", 10)
                            )
                            after_reply_add = float(
                                self._get_conf("focus_mode_cycle_after_reply_add", 20)
                            )
                            max_delay = base_delay + max(0.0, after_reply_add)

                            if task.task_type == "focus":
                                self.state.set_focus_delay(
                                    umo, max_delay, min_val=min_delay
                                )
                                logger.info(
                                    f"[mem_chat] FOCUS reply applied. Next focus delay set to {self.state.get_focus_delay(umo, default=base_delay):.2f}s."
                                )

                        if replied and self.state.is_in_focus(umo):
                            await self.trigger_view_cycle(umo, "focus", task.event, [])

                        # Note: Next cycle triggering is now handled in finally block of _cycle_wait_and_submit
                        # via trigger_view_cycle which checks state.

                except Exception as e:
                    logger.exception(
                        f"[mem_chat] Task execution failed: type={type(e).__name__} umo={getattr(task, 'umo', None)} task_type={getattr(task, 'task_type', None)} error={e}"
                    )
                finally:
                    queue.task_done()

        finally:
            self.active_queues.remove(umo)
            logger.info(f"[mem_chat] Queue processor stopped for {umo}")

    async def _get_conversation(self, event: AstrMessageEvent):
        try:
            uid = event.unified_msg_origin
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(
                uid
            )
            if curr_cid:
                return await self.context.conversation_manager.get_conversation(
                    uid, curr_cid
                )
            return None
        except Exception:
            return None

    async def get_recent_history(
        self, umo: str, count: int, event: AstrMessageEvent | None = None
    ) -> list[str]:
        try:
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(
                umo
            )
            messages: list[str] = []
            if curr_cid:
                conversation = await self.context.conversation_manager.get_conversation(
                    umo, curr_cid
                )
                if conversation and getattr(conversation, "history", None):
                    import json

                    try:
                        hist = json.loads(conversation.history)
                        take = hist[-count:]

                        def _normalize_msg_content(val) -> str:
                            if val is None:
                                return ""
                            if isinstance(val, str):
                                return val
                            if isinstance(val, list):
                                parts: list[str] = []
                                for item in val:
                                    if isinstance(item, str):
                                        s = item.strip()
                                        if s:
                                            parts.append(s)
                                        continue
                                    if isinstance(item, dict):
                                        if item.get("type") == "text" and isinstance(
                                            item.get("text"), str
                                        ):
                                            s = item.get("text", "").strip()
                                            if s:
                                                parts.append(s)
                                            continue
                                        if isinstance(item.get("text"), str):
                                            s = item.get("text", "").strip()
                                            if s:
                                                parts.append(s)
                                            continue
                                        if isinstance(item.get("content"), str):
                                            s = item.get("content", "").strip()
                                            if s:
                                                parts.append(s)
                                            continue
                                return " ".join(parts).strip()
                            if isinstance(val, dict):
                                if val.get("type") == "text" and isinstance(
                                    val.get("text"), str
                                ):
                                    return val.get("text", "")
                                if isinstance(val.get("text"), str):
                                    return val.get("text", "")
                                if isinstance(val.get("content"), str):
                                    return val.get("content", "")
                                return ""
                            return str(val)

                        for m in take:
                            try:
                                role = str(m.get("role") or "").strip()
                                sname = str(m.get("sender_name") or "").strip()
                                sid = str(m.get("sender_id") or "").strip()
                                content = _normalize_msg_content(
                                    m.get("content")
                                ).strip()

                                if not content:
                                    if isinstance(m, dict) and m.get("tool_calls"):
                                        content = "[tool_calls]"
                                    else:
                                        continue

                                if not sname:
                                    if role == "assistant":
                                        sname = (
                                            self.state.get_self_name(umo) or "Assistant"
                                        )
                                    elif role == "user":
                                        sname = "User"

                                content = self.db_mgr.clean_legacy_content(content)
                                try:
                                    content = (
                                        self.llm_client._strip_search_listing_text(
                                            content
                                        )
                                    )
                                except Exception:
                                    pass

                                sender_part = sname
                                if sid:
                                    sender_part = f"{sname}/{sid}"
                                elif role == "assistant":
                                    sender_part = (
                                        self.state.get_self_name(umo) or "Assistant"
                                    )

                                if sender_part and content:
                                    messages.append(f"{sender_part}: {content}")
                                elif content:
                                    messages.append(content)
                            except Exception:
                                continue
                    except Exception:
                        pass

            # If no messages found from conversation (or it was empty), try fallback to event if provided
            if not messages and event:
                mt = event.message_str or event.get_message_outline()
                if mt:
                    sname = event.get_sender_name() or ""
                    sid = event.get_sender_id() or ""
                    sp = sname
                    if sid:
                        sp = f"{sname}/{sid}"
                    messages.append(f"{sp}: {mt}")

            return messages
        except Exception:
            return []

    def compose_contexts(
        self, event: AstrMessageEvent, history_msgs: list[str]
    ) -> list[str]:
        ctx: list[str] = []
        if history_msgs:
            ctx.append("【历史对话】\n" + "\n".join(history_msgs))
        try:
            extra = getattr(event, "context_extra", None)
            if isinstance(extra, dict):
                mc = extra.get("memory_context") or ""
                if mc:
                    ctx.append(mc)
        except Exception:
            pass
        return ctx

    async def summarize_session(
        self, umo: str, group_id: str, sender_id: str, sender_name: str
    ):
        """Summarize recent conversation and update impression."""
        # Get recent messages from this user (50 messages)
        history = await self.db_mgr.get_user_specific_history(
            group_id, sender_id, limit=50
        )
        if not history:
            return

        history_text = "\n".join(history)

        prompt = (
            f"请根据以下 {sender_name} (ID: {sender_id}) 的最近聊天记录，"
            f"总结对该用户的印象，以及提取关键的对话信息（如偏好、正在讨论的话题等）。"
            f"请保持简洁，不要超过200字。\n\n"
            f"{history_text}"
        )

        # Use a generic system prompt for summarization
        system_prompt = "你是一个善于总结对话的助手。请客观地总结用户的特征和对话内容。"

        summary = await self.llm_client.generate_text(
            prompt, system_prompt=system_prompt, umo=umo
        )
        if summary:
            logger.info(f"[mem_chat] Generated summary for {sender_name}: {summary}")
            # 1. Save to impressions table (for prompt injection)
            self.db_mgr.save_impression(group_id, sender_id, summary, sender_name)

            # 2. Save to memory_embeddings as 'impression' (for vector recall)
            # This allows "impression vector database" usage as requested.
            impression_id = self.db_mgr.hash_id(summary, group_id, "impression")
            # We use ensure_message_embedded logic but with type='impression'
            # Since ensure_message_embedded is for messages, we call get_or_compute_embedding directly
            model_name = str(
                self.config.get("embedding_openai_model", "text-embedding-3-small")
                or "text-embedding-3-small"
            )
            await self.db_mgr.get_or_compute_embedding(
                impression_id, summary, group_id, model_name, type="impression"
            )

    async def build_memory_context(
        self,
        query: str,
        group_id: str,
        sender_name: str | None,
        history_msgs: list[str] | None = None,
        sender_id: str | None = None,
        conversation=None,
    ) -> str:
        """Build the full context including memory, persona, and knowledge."""
        context_parts = []

        def _compact_injected_text(text: str, max_chars: int) -> str:
            try:
                raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
                lines = []
                for ln in raw.split("\n"):
                    s = ln.strip()
                    if not s:
                        continue
                    if re.match(r"^\[\d{2}:\d{2}:\d{2}\]\s+\[[^\]]+\]\s+\[[A-Z]+\]", s):
                        continue
                    if (
                        s.startswith("[mem_chat]")
                        or "src.llm_client:" in s
                        or "src.scheduler:" in s
                        or "src.core.scheduler:" in s
                    ):
                        continue
                    lines.append(s)
                t = re.sub(r"\s+", " ", " ".join(lines)).strip()
                if not t:
                    return ""
                if max_chars > 0 and len(t) <= max_chars:
                    return t
                parts = re.split(r"(?<=[。！？.!?])\s+", t)
                out = []
                seen = set()
                total = 0
                for p in parts:
                    p = p.strip()
                    if not p:
                        continue
                    key = p
                    if key in seen:
                        continue
                    if max_chars > 0 and total + len(p) + 1 > max_chars:
                        break
                    seen.add(key)
                    out.append(p)
                    total += len(p) + 1
                if not out:
                    return t[:max_chars] + (
                        "…" if max_chars > 0 and len(t) > max_chars else ""
                    )
                res = " ".join(out).strip()
                if max_chars > 0 and len(t) > len(res):
                    res = (res[:max_chars]).rstrip() + "…"
                return res
            except Exception:
                return str(text or "")

        personas_text = ""
        try:
            personas_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "personas",
                "personas.txt",
            )
            if os.path.exists(personas_path):
                with open(personas_path, encoding="utf-8") as f:
                    personas_text = f.read().strip()
        except Exception:
            personas_text = ""

        if personas_text:
            context_parts.append("【人物印象/设定】\n" + personas_text)

        qemb = await self.db_mgr.get_embedding(query)
        knowledge_task = self.db_mgr.semantic_recall_with_qemb(
            qemb,
            group_id,
            limit=5,
            threshold=0.0,
            type="knowledge",
        )

        limit_mem_conf = int(self._get_conf("max_injected_memories", 40))
        limit_mem = max(1, limit_mem_conf)
        memories_task = self.db_mgr.semantic_recall_with_qemb(
            qemb,
            group_id,
            limit=limit_mem,
            threshold=0.0,
            type="memory",
        )

        max_persons_conf = int(self._get_conf("max_impression_persons", 3))
        max_persons = max(3, min(5, max_persons_conf))
        max_per_person = int(self._get_conf("max_impressions_per_person", 3))
        min_score_impr = float(self._get_conf("impression_min_score", 0.35))
        impressions_task = self.db_mgr.semantic_recall_with_qemb(
            qemb,
            group_id,
            limit=max_persons * max_per_person,
            threshold=min_score_impr,
            type="impression",
        )
        raw_knowledge, memories, recalled = await asyncio.gather(
            knowledge_task,
            memories_task,
            impressions_task,
        )

        knowledge = []
        high_quality = [k for k in raw_knowledge if k[1] >= 0.5]
        if len(high_quality) >= 2:
            knowledge = high_quality
        else:
            knowledge = raw_knowledge[:2]
        if knowledge:
            k_text = "【相关知识】\n"
            for k in knowledge:
                content = _compact_injected_text(k[0], 450)
                if content:
                    k_text += f"- {content}\n"
            context_parts.append(k_text)

        if memories:
            m_text = "【相关记忆】\n"
            for m in memories:
                content = _compact_injected_text(m[0], 220)
                if content:
                    m_text += f"- {content}\n"
            context_parts.append(m_text)
        if recalled:
            rows = self.db_mgr.list_impressions(group_id)
            by_summary = {}
            for uid, name, summary in rows:
                by_summary.setdefault(summary, (uid, name))
            grouped = {}
            for content, score in recalled:
                info = by_summary.get(content)
                if not info:
                    continue
                uid, name = info
                if uid not in grouped:
                    grouped[uid] = {"name": name, "items": []}
                # split summary into up to max_per_person lines
                segs = []
                try:
                    segs = [
                        s.strip()
                        for s in re.split(r"[。！？.!?]\s*", content)
                        if s.strip()
                    ]
                except Exception:
                    segs = [content]
                for seg in segs:
                    if len(grouped[uid]["items"]) >= max_per_person:
                        break
                    grouped[uid]["items"].append(seg)
                if len(grouped) >= max_persons:
                    pass
            if grouped:
                i_text = "【人物印象】\n"
                cnt = 0
                for uid, data in grouped.items():
                    if cnt >= max_persons:
                        break
                    name = data.get("name") or "User"
                    items = data.get("items") or []
                    if not items:
                        continue
                    i_text += f"{name}/{uid}:\n"
                    for it in items[:max_per_person]:
                        content = _compact_injected_text(it, 120)
                        if content:
                            i_text += f"- {content}\n"
                    cnt += 1
                if cnt > 0:
                    context_parts.append(i_text)
        elif sender_id and group_id:
            # Fallback to legacy exact match impression if vector search fails
            impression = self.db_mgr.get_impression(group_id, sender_id)
            if impression:
                context_parts.append(
                    f"【人物印象】\n关于 {sender_name} 的印象：\n{impression}"
                )

        # 5. Recent History
        if history_msgs:
            cleaned = []
            for ln in history_msgs:
                s = _compact_injected_text(ln, 260)
                if s:
                    cleaned.append(s)
            if cleaned:
                h_text = "【历史对话】\n" + "\n".join(cleaned)
                context_parts.append(h_text)

        return "\n\n".join(context_parts)


__all__ = ["Scheduler"]
