import asyncio
import json
import logging
import os
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import At, AtAll, Face, Plain, Poke, Reply
from astrbot.api.star import Context

from ..db.database import DatabaseManager
from ..llm.prompt_builder import build_decision_prompt, build_focus_prompt
from ..llm_client import LLMClient
from .member_tags import format_sender, load_sender_tags
from .models import ReplyTask
from .state import MemChatState

logger = logging.getLogger("astrbot")


async def _empty_recall() -> list[tuple[str, float]]:
    """Awaitable placeholder so gather keeps a fixed shape when a recall is off."""
    return []

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
        self._deferred_focus = {}  # umo -> bool
        self._admin_protocols_cache: str | None = None
        self._affinity_decay_last_ts: dict[str, float] = {}

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

        if cycle_type == "read_air" and not bool(self._get_conf("read_air_enable", True)):
            return

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
            if cycle_type == "focus":
                self._deferred_focus[umo] = True
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
            cycle_min, cycle_max, _cycle_step_max, base_delay = self._focus_cycle_bounds()
            cycle_delay = self.state.get_focus_delay(umo, default=base_delay)
            cycle_delay = self._clamp_float(float(cycle_delay), cycle_min, cycle_max)
            self.state.set_focus_delay(umo, cycle_delay, min_val=cycle_min)
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
                        now_cutoff = time.time()
                        self.state.set_last_cycle_cutoff(umo, now_cutoff)
                        self.state.clear_pending_up_to(umo, now_cutoff)
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
                    cycle_min, cycle_max, _cycle_step_max, base_delay = self._focus_cycle_bounds()
                    base_delay = self._clamp_float(base_delay, cycle_min, cycle_max)
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
                    elif isinstance(i, Poke):
                        parts.append(self._render_poke_text(event, i))
                    else:
                        pass
                return "".join(parts).strip()
        except Exception:
            pass
        return event.message_str or ""

    def _render_poke_text(self, event: AstrMessageEvent, seg: Poke) -> str:
        action_text = "戳了戳"
        suffix_text = ""
        try:
            raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
            raw_info = raw.get("raw_info", []) if hasattr(raw, "get") else []
            for item in raw_info or []:
                if not isinstance(item, dict):
                    continue
                txt = str(item.get("txt") or "").strip()
                if not txt:
                    continue
                if txt in ("戳了戳", "拍了拍", "抱了抱"):
                    action_text = txt
                elif action_text:
                    suffix_text += txt
        except Exception:
            pass

        operator_id = str(
            getattr(seg, "id", None) or event.get_sender_id() or ""
        ).strip()
        target_id = str(getattr(seg, "qq", None) or "").strip()
        self_id = str(event.get_self_id() or "").strip()
        sender_id = str(event.get_sender_id() or "").strip()
        sender_name = str(event.get_sender_name() or "").strip()

        operator_label = operator_id or "有人"
        if sender_name and sender_id and operator_id and sender_id == operator_id:
            operator_label = f"{sender_name}/{operator_id}"

        if target_id and self_id and target_id == self_id:
            return f"{operator_label} {action_text}你{suffix_text}".strip()
        if target_id:
            return f"{operator_label} {action_text} {target_id}{suffix_text}".strip()
        return f"{operator_label} 触发了戳一戳".strip()

    def _get_conf(self, key: str, default):
        try:
            if isinstance(self.config, dict):
                v = self.config.get(key, default)
            else:
                v = getattr(self.config, key, default)
        except Exception:
            v = default
        return v

    def _to_float(self, value, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _clamp_float(self, value: float, min_v: float, max_v: float) -> float:
        if max_v < min_v:
            max_v = min_v
        if value < min_v:
            return min_v
        if value > max_v:
            return max_v
        return value

    def _step_towards(
        self, current: float, target: float, step_max: float, min_v: float, max_v: float
    ) -> float:
        current = self._clamp_float(current, min_v, max_v)
        target = self._clamp_float(target, min_v, max_v)
        step_max = max(0.0, float(step_max))
        if step_max <= 0:
            return target
        delta = target - current
        if abs(delta) <= step_max:
            return target
        return current + (step_max if delta > 0 else -step_max)

    def _focus_cycle_bounds(self) -> tuple[float, float, float, float]:
        base = self._to_float(self._get_conf("focus_mode_cycle", 15), 15.0)
        if base < 1:
            base = 1.0
        min_v = self._to_float(
            self._get_conf("focus_cycle_min_sec", self._get_conf("focus_mode_cycle_min", 10)),
            min(10.0, base),
        )
        max_v = self._to_float(
            self._get_conf("focus_cycle_max_sec", base + 20.0),
            base + 20.0,
        )
        step_max = self._to_float(
            self._get_conf("focus_cycle_step_max_sec", 20),
            20.0,
        )
        if min_v < 1:
            min_v = 1.0
        if max_v < min_v:
            max_v = min_v
        if step_max <= 0:
            step_max = max_v - min_v
        base = self._clamp_float(base, min_v, max_v)
        return min_v, max_v, step_max, base

    def _focus_duration_bounds(self) -> tuple[int, int, int, int]:
        base = int(self._to_float(self._get_conf("focus_duration_sec", 100), 100.0))
        min_v = int(
            self._to_float(self._get_conf("focus_duration_min_sec", 60), 60.0)
        )
        max_v = int(
            self._to_float(self._get_conf("focus_duration_max_sec", 900), 900.0)
        )
        step_max = int(
            self._to_float(self._get_conf("focus_duration_step_max_sec", 120), 120.0)
        )
        if min_v < 1:
            min_v = 1
        if max_v < min_v:
            max_v = min_v
        if step_max <= 0:
            step_max = max(1, max_v - min_v)
        if base < min_v:
            base = min_v
        elif base > max_v:
            base = max_v
        return min_v, max_v, step_max, base

    def _event_targets_bot(self, umo: str, event: AstrMessageEvent | None) -> bool:
        if event is None:
            return False
        try:
            if bool(getattr(event, "is_at_or_wake_command", False)):
                return True
        except Exception:
            pass
        self_id = str(self.state.get_self_id(umo) or "").strip()
        try:
            for seg in event.get_messages() or []:
                if isinstance(seg, AtAll):
                    return True
                if isinstance(seg, At):
                    qq = str(getattr(seg, "qq", "") or "").strip()
                    if not self_id:
                        return True
                    if qq and qq == self_id:
                        return True
                if isinstance(seg, Reply):
                    reply_sender_id = str(getattr(seg, "sender_id", "") or "").strip()
                    if self_id and reply_sender_id and reply_sender_id == self_id:
                        return True
                if isinstance(seg, Poke):
                    target_id = str(getattr(seg, "qq", None) or "").strip()
                    if self_id and target_id and target_id == self_id:
                        return True
        except Exception:
            pass
        try:
            txt = self._get_full_text(event)
            if self_id and (f"@{self_id}" in txt or f"[At:{self_id}]" in txt):
                return True
        except Exception:
            pass
        return False

    def _text_targets_bot(self, text: str, self_id: str | None) -> bool:
        t = str(text or "")
        sid = str(self_id or "").strip()
        if not t.strip():
            return False
        if "戳了戳你" in t:
            return True
        if sid:
            if f"@{sid}" in t or f"[At:{sid}]" in t:
                return True
        return False

    def _meta_target_label(self, meta: dict | None) -> str:
        if not isinstance(meta, dict):
            return "群聊"
        t = str(
            meta.get("inferred_target_name")
            or meta.get("explicit_target_name")
            or ""
        ).strip()
        if t:
            return t
        t_type = str(
            meta.get("inferred_target_type")
            or meta.get("explicit_target_type")
            or ""
        ).strip()
        if t_type == "bot":
            return "机器人"
        if t_type == "group_all":
            return "群聊全体"
        return "群聊"

    def _build_conversation_target_block(
        self,
        *,
        umo: str,
        meta: dict | None,
        sender_name: str | None,
        sender_id: str | None,
        sender_tags: dict[str, str] | None = None,
    ) -> str:
        self_id = str(self.state.get_self_id(umo) or "").strip()
        self_name = str(self.state.get_self_name(umo) or "").strip()
        m = meta or {}
        explicit_type = str(m.get("explicit_target_type") or "group").strip()
        explicit_name = str(m.get("explicit_target_name") or "").strip()
        explicit_id = str(m.get("explicit_target_id") or "").strip()
        inferred_type = str(m.get("inferred_target_type") or explicit_type or "group").strip()
        inferred_name = str(m.get("inferred_target_name") or explicit_name or "").strip()
        inferred_id = str(m.get("inferred_target_id") or explicit_id or "").strip()
        reason = str(m.get("target_reason") or "").strip()
        confidence = str(m.get("target_confidence") or "low").strip()
        reply_sender_name = str(m.get("reply_to_sender_name") or "").strip()
        reply_sender_id = str(m.get("reply_to_sender_id") or "").strip()
        at_targets = m.get("at_targets") if isinstance(m.get("at_targets"), list) else []
        poke_operator_name = str(m.get("poke_operator_name") or "").strip()
        poke_operator_id = str(m.get("poke_operator_id") or "").strip()
        poke_target_name = str(m.get("poke_target_name") or "").strip()
        poke_target_id = str(m.get("poke_target_id") or "").strip()
        poke_action_text = str(m.get("poke_action_text") or "").strip()
        poke_suffix_text = str(m.get("poke_suffix_text") or "").strip()

        def _fmt_target(t_type: str, t_name: str, t_id: str) -> str:
            if t_type == "bot":
                return "机器人"
            if t_type == "group_all":
                return "群聊全体"
            if t_type == "group":
                return "群聊"
            if t_name or t_id:
                return format_sender(t_name, t_id, sender_tags) or "群聊"
            return "群聊"

        lines = ["【对话对象判断】"]
        current_sender = format_sender(
            sender_name or "User", sender_id, sender_tags
        ) or "User"
        lines.append(f"当前发言人: {current_sender}")
        bot_label = f"{self_name}/{self_id}" if self_name and self_id else (self_id or "未知")
        lines.append(f"机器人ID: {bot_label}")
        lines.append("")
        lines.append("显式信号:")
        if reply_sender_id or reply_sender_name:
            lines.append(f"- 回复对象: {_fmt_target('user', reply_sender_name, reply_sender_id)}")
        else:
            lines.append("- 回复对象: 无")
        if at_targets:
            at_labels = []
            for item in at_targets:
                iid = str(item.get("id") or "").strip()
                iname = str(item.get("name") or "").strip()
                at_labels.append(_fmt_target("user", iname, iid))
            lines.append(f"- @对象: {', '.join(at_labels)}")
        else:
            lines.append("- @对象: 无")
        if poke_operator_id or poke_target_id:
            poke_from = _fmt_target(
                "user",
                poke_operator_name,
                poke_operator_id,
            )
            poke_to = _fmt_target(
                "bot" if self_id and poke_target_id == self_id else "user",
                poke_target_name,
                poke_target_id,
            )
            poke_action = f"{poke_action_text or '戳了戳'}{poke_suffix_text}".strip()
            lines.append(f"- 戳一戳: {poke_from} -> {poke_to} ({poke_action})")
        else:
            lines.append("- 戳一戳: 无")
        lines.append(f"- @全体: {'是' if explicit_type == 'group_all' else '否'}")
        lines.append(f"- 是否明确点名机器人: {'是' if explicit_type == 'bot' else '否'}")
        lines.append("")
        lines.append("系统判断:")
        lines.append(f"- 这句话最可能是在对: {_fmt_target(inferred_type, inferred_name, inferred_id)}")
        lines.append(f"- 判断类型: {reason or inferred_type or explicit_type or 'unknown'}")
        lines.append(f"- 置信度: {confidence or 'low'}")
        return "\n".join(lines).strip()

    def _event_pokes_bot(self, umo: str, event: AstrMessageEvent | None) -> bool:
        if event is None:
            return False
        self_id = str(self.state.get_self_id(umo) or "").strip()
        try:
            for seg in event.get_messages() or []:
                if not isinstance(seg, Poke):
                    continue
                target_id = str(getattr(seg, "qq", None) or "").strip()
                if self_id and target_id and target_id == self_id:
                    return True
        except Exception:
            pass
        try:
            txt = self._get_full_text(event)
            if "戳了戳你" in txt:
                return True
        except Exception:
            pass
        return False

    def _preapply_poke_affinity(
        self,
        umo: str,
        event: AstrMessageEvent | None,
        group_id: str,
        sender_id: str,
    ) -> bool:
        if not bool(self._get_conf("affinity_enabled", True)):
            return False
        if not bool(self._get_conf("affinity_poke_preapply_before_prompt", True)):
            return False
        if not self._event_pokes_bot(umo, event):
            return False
        gid = str(group_id or "").strip()
        uid = str(sender_id or "").strip()
        if not gid or not uid:
            return False
        msg_id = ""
        try:
            msg_id = str(getattr(event.message_obj, "message_id", "") or "").strip()
        except Exception:
            msg_id = ""
        if msg_id and self.state.is_poke_preapplied_msg(umo, msg_id):
            return False
        try:
            delta = float(self._get_conf("affinity_poke_min_delta", 10.0))
        except Exception:
            delta = 10.0
        if delta <= 0:
            return False
        try:
            cooldown_sec = float(self._get_conf("affinity_delta_cooldown_sec", 120))
        except Exception:
            cooldown_sec = 120.0
        if bool(self._get_conf("affinity_poke_ignore_cooldown", True)):
            cooldown_sec = 0.0
        self.db_mgr.ensure_affinity_state(gid, uid)
        ret = self.db_mgr.apply_affinity_delta(
            gid,
            uid,
            delta_affinity=delta,
            reason="pre_poke:before_prompt",
            source_msg_id=msg_id,
            cooldown_sec=cooldown_sec,
        )
        if ret is None:
            return False
        if msg_id:
            self.state.mark_poke_preapplied_msg(umo, msg_id)
        logger.info(
            f"[mem_chat] Affinity pre-applied for poke gid={gid} uid={uid} delta={delta:.2f}."
        )
        return True

    def _mood_to_affinity_delta(
        self, mood_state: str, mood_text: str, intensity: float
    ) -> tuple[float, str]:
        state = str(mood_state or "").strip()
        text = str(mood_text or "").strip()
        try:
            inten = float(intensity or 0.0)
        except Exception:
            inten = 0.0
        if inten < 0.0:
            inten = 0.0
        if inten > 1.0:
            inten = 1.0
        # Early-stage relation should be able to move fast; final cap is enforced in DB layer.
        scale = 0.7 + 1.3 * inten
        base_map = {
            "中性": 0.0,
            "高兴": 12.0,
            "喜欢": 20.0,
            "色色": 8.0,
            "讥讽": -14.0,
            "生气": -30.0,
            "软懦": -18.0,
        }
        base = float(base_map.get(state, 0.0))
        delta = base * scale

        # Text hints can slightly adjust direction.
        positive_hints = ("哄", "原谅", "开心", "喜欢", "可爱", "谢谢", "抱抱")
        negative_hints = ("烦", "滚", "讨厌", "闭嘴", "恶心", "生气", "拉黑")
        if any(k in text for k in positive_hints):
            delta += 6.0
        if any(k in text for k in negative_hints):
            delta -= 6.0

        if delta > 40.0:
            delta = 40.0
        elif delta < -40.0:
            delta = -40.0
        reason = f"mood={state or '中性'} intensity={inten:.2f} text={text[:80]}"
        return delta, reason

    def _maybe_decay_affinity(self, group_id: str):
        gid = str(group_id or "").strip()
        if not gid:
            return
        if not bool(self._get_conf("affinity_enabled", True)):
            return
        if not bool(self._get_conf("affinity_decay_enabled", True)):
            return
        now = time.time()
        try:
            interval = float(self._get_conf("affinity_decay_interval_sec", 1800))
        except Exception:
            interval = 1800.0
        if interval <= 0:
            interval = 1800.0
        last = float(self._affinity_decay_last_ts.get(gid, 0.0) or 0.0)
        if last > 0 and now - last < interval:
            return
        try:
            step = float(self._get_conf("affinity_decay_step", 1.0))
        except Exception:
            step = 1.0
        try:
            idle_sec = float(self._get_conf("affinity_decay_idle_sec", 3600))
        except Exception:
            idle_sec = 3600.0
        changed = self.db_mgr.decay_affinity_towards_neutral(
            gid, step=step, idle_sec=idle_sec
        )
        self._affinity_decay_last_ts[gid] = now
        if changed > 0:
            logger.info(f"[mem_chat] Affinity decay applied for group={gid}, changed={changed}.")

    def _maybe_update_affinity_from_mood(
        self,
        umo: str,
        task_type: str,
        event: AstrMessageEvent | None,
        group_id: str | None,
        sender_id: str | None,
        history_msgs: list[str] | None = None,
    ):
        if not bool(self._get_conf("affinity_enabled", True)):
            return
        gid = str(group_id or "").strip()
        uid = str(sender_id or "").strip()
        if not gid or not uid:
            return
        require_at = bool(self._get_conf("affinity_update_require_at", True))
        targets_bot = self._event_targets_bot(umo, event)
        poke_hit = self._event_pokes_bot(umo, event)
        if require_at and not (task_type == "at" or targets_bot):
            return

        mood_state = ""
        try:
            mood_state = self.state.get_last_mood_state(umo)
        except Exception:
            pass

        self.db_mgr.ensure_affinity_state(gid, uid)
        try:
            self.db_mgr.set_affinity_state(gid, uid, mood_bias=(mood_state or "中性"))
        except Exception:
            pass

        llm_delta = None
        llm_reason = ""
        try:
            max_age = float(self._get_conf("affinity_llm_signal_max_age_sec", 600))
        except Exception:
            max_age = 600.0
        try:
            rec = self.state.pop_last_affinity_delta(umo, uid, max_age_sec=max_age)
        except Exception:
            rec = None
        if rec is not None:
            llm_delta, llm_reason = rec

        if llm_delta is None:
            if poke_hit:
                llm_delta = 0.0
                llm_reason = "poke_only"
            else:
                use_fallback = bool(self._get_conf("affinity_llm_fallback_to_mood", False))
                if not use_fallback:
                    logger.info(
                        f"[mem_chat] Affinity skip gid={gid} uid={uid}: missing llm affinity signal."
                    )
                    return
                mood_text = ""
                intensity = 0.0
                try:
                    mood_text = self.state.get_last_mood(umo)
                    intensity = self.state.get_last_mood_intensity(umo)
                except Exception:
                    pass
                llm_delta, llm_reason = self._mood_to_affinity_delta(
                    mood_state, mood_text, intensity
                )
                llm_reason = f"fallback:{llm_reason}"

        delta = float(llm_delta or 0.0)
        newcomer_factor = 1.0
        if bool(self._get_conf("affinity_new_user_correction_enabled", True)):
            try:
                event_count = int(self.db_mgr.get_affinity_event_count(gid, uid))
            except Exception:
                event_count = 0
            try:
                window = int(self._get_conf("affinity_new_user_window_events", 8))
            except Exception:
                window = 8
            try:
                start_factor = float(self._get_conf("affinity_new_user_factor_start", 0.4))
                end_factor = float(self._get_conf("affinity_new_user_factor_end", 1.0))
            except Exception:
                start_factor, end_factor = 0.4, 1.0
            if window > 0 and event_count < window:
                progress = float(event_count) / float(window)
                newcomer_factor = start_factor + (end_factor - start_factor) * progress
                if newcomer_factor < 0.05:
                    newcomer_factor = 0.05
                if newcomer_factor > 1.5:
                    newcomer_factor = 1.5
                delta *= newcomer_factor
                llm_reason = (
                    f"{llm_reason} | local_factor={newcomer_factor:.2f}"
                    f"(new_user_count={event_count}/{window})"
                )

        poke_preapply = bool(self._get_conf("affinity_poke_preapply_before_prompt", True))
        # Poke is treated as a strong positive interaction.
        if poke_hit and not poke_preapply:
            try:
                poke_bonus = float(self._get_conf("affinity_poke_bonus_force", 12.0))
            except Exception:
                poke_bonus = 12.0
            if poke_bonus > 0:
                delta += poke_bonus
            try:
                poke_min = float(self._get_conf("affinity_poke_min_delta", 10.0))
            except Exception:
                poke_min = 10.0
            if delta < poke_min:
                delta = poke_min
            if llm_reason:
                llm_reason += f" | poke_bonus={poke_bonus:.1f}"
            else:
                llm_reason = f"poke_bonus={poke_bonus:.1f}"
        elif poke_hit and poke_preapply:
            if llm_reason:
                llm_reason += " | poke_pre_applied"
            else:
                llm_reason = "poke_pre_applied"

        if delta > 40.0:
            delta = 40.0
        elif delta < -40.0:
            delta = -40.0
        if abs(delta) < 0.01:
            return
        reason = f"llm:{llm_reason}" if llm_reason else "llm"
        try:
            cooldown_sec = float(self._get_conf("affinity_delta_cooldown_sec", 120))
        except Exception:
            cooldown_sec = 120.0
        if poke_hit and bool(self._get_conf("affinity_poke_ignore_cooldown", True)):
            cooldown_sec = 0.0
        msg_id = ""
        try:
            msg_id = str(getattr(event.message_obj, "message_id", "") or "")
        except Exception:
            msg_id = ""
        ret = self.db_mgr.apply_affinity_delta(
            gid,
            uid,
            delta_affinity=delta,
            reason=reason,
            source_msg_id=msg_id,
            cooldown_sec=cooldown_sec,
        )
        if ret is not None:
            logger.info(
                f"[mem_chat] Affinity updated gid={gid} uid={uid} delta={delta:.2f} reason={reason[:60]}"
            )

    def _load_admin_protocols_text(self) -> str:
        if isinstance(self._admin_protocols_cache, str) and self._admin_protocols_cache:
            return self._admin_protocols_cache
        text = ""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            p = os.path.join(base_dir, "admin_protocols", "admin_protocols.txt")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    text = (f.read() or "").strip()
        except Exception:
            text = ""
        self._admin_protocols_cache = text
        return text

    def _should_inject_admin_protocols(self, text: str) -> bool:
        s = str(text or "").lower()
        if not s.strip():
            return False
        keywords = [
            "禁言",
            "解禁",
            "解除禁言",
            "撤回",
            "精华",
            "群精华",
            "群管",
            "管理员",
            "踢",
            "踢人",
            "封禁",
            "mute",
            "unmute",
            "ban",
            "recall",
            "essence",
            "set_group_essence",
            "set_essence",
            "群名片",
            "头衔",
            "group_card",
            "group_title",
            "违规",
            "刷屏",
            "spam",
        ]
        return any(k in s for k in keywords)

    def _get_task_timeout_sec(self, task_type: str) -> float:
        task_norm = str(task_type or "").strip()
        mapping = {
            "at": "task_timeout_sec_at",
            "focus": "task_timeout_sec_focus",
            "read_air": "task_timeout_sec_read_air",
        }
        defaults = {
            "at": 90,
            "focus": 60,
            "read_air": 120,
        }
        k = mapping.get(task_norm, "task_timeout_sec_focus")
        try:
            v = float(self._get_conf(k, defaults.get(task_norm, 60)))
            if v <= 0:
                return 0.0
            return v
        except Exception:
            return float(defaults.get(task_norm, 60))

    def _get_task_remaining_timeout_sec(self, task: ReplyTask) -> float:
        timeout_sec = self._get_task_timeout_sec(getattr(task, "task_type", ""))
        if timeout_sec <= 0:
            return 0.0
        elapsed = time.time() - float(getattr(task, "created_at", time.time()) or time.time())
        return max(0.0, timeout_sec - elapsed)

    async def _await_task_stage(self, task: ReplyTask, stage: str, awaitable):
        start = time.time()
        remaining = self._get_task_remaining_timeout_sec(task)
        timeout_text = "none" if remaining <= 0 else f"{remaining:.1f}s"
        logger.info(
            f"[mem_chat] Stage start: {stage} task={task.task_type} umo={task.umo} timeout={timeout_text}"
        )
        try:
            if remaining > 0:
                result = await asyncio.wait_for(awaitable, timeout=max(1.0, remaining))
            else:
                result = await awaitable
            dur = time.time() - start
            logger.info(
                f"[mem_chat] Stage done: {stage} task={task.task_type} umo={task.umo} dur={dur:.2f}s"
            )
            return result
        except asyncio.TimeoutError:
            dur = time.time() - start
            logger.warning(
                f"[mem_chat] Stage timeout: {stage} task={task.task_type} umo={task.umo} dur={dur:.2f}s"
            )
            raise

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
                            min_interval = self._get_conf(
                                "focus_cycle_min_sec",
                                self._get_conf("focus_mode_cycle_min", 0),
                            )
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
                        umo, history_count, task.event, task_type=task.task_type
                    )

                    prompt = ""
                    memory_context = ""

                    # Ensure conv_mgr is available
                    conv_mgr = self.context.conversation_manager

                    if task.task_type == "at":
                        # We need access to the original 'event' to build prompt
                        if not task.event:
                            continue

                        msg_text = self._get_full_text(task.event)

                        # Fetch pending messages to save and include in context
                        # Consuming pending messages ensures Heartbeat doesn't process them again
                        raw_pending = self.state.get_pending_messages(umo)
                        raw_pending = [
                            m
                            for m in (raw_pending or [])
                            if not self.state.is_no_reply_message_id(
                                umo, m[0] if m else None
                            )
                        ]
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
                        index_to_id = {}
                        sender_tags = {}

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
                                            "_msg_id": _mid,
                                        },
                                    )
                                    if sid:
                                        at_candidates.append(sid)

                            self.state.set_last_processed(umo, time.time())
                        try:
                            from .ref_context import build_ref_context

                            sender_tags = load_sender_tags(
                                task.event.get_group_id()
                                or task.event.get_sender_id()
                            )
                            ref_window = int(self._get_conf("ref_context_window", 0))
                            if ref_window <= 0:
                                ref_window = max_at_msgs
                            include_referenced = bool(
                                self._get_conf("ref_context_include_referenced", True)
                            )
                            pend_lines, quote_candidates, index_to_id = build_ref_context(
                                raw_pending if raw_pending else [],
                                None,
                                None,
                                get_reply_target=lambda mid: self.state.get_reply_target(
                                    umo, mid
                                ),
                                get_reply_summary=lambda rid: self.state.get_reply_summary(
                                    umo, rid
                                ),
                                get_message_meta=lambda mid: self.state.get_message_meta(
                                    umo, mid
                                ),
                                max_items=ref_window,
                                include_referenced=include_referenced,
                                time_gap_min_minutes=int(
                                    self._get_conf("history_time_gap_minutes", 15)
                                ),
                                time_marker_every_n_messages=int(
                                    self._get_conf("history_time_marker_every_n", 4)
                                ),
                                sender_tags=sender_tags,
                            )
                        except Exception:
                            pend_lines = []
                            quote_candidates = []
                            index_to_id = {}

                        # Re-calculate context
                        gid = task.event.get_group_id() or task.event.get_sender_id()
                        sender_id = task.event.get_sender_id()
                        sender_name = task.event.get_sender_name()

                        # Use latest message from pending if available; AT event may be older than newest pending.
                        last_msg = msg_text
                        reply_sender_name = sender_name
                        reply_sender_id = sender_id
                        latest_mid = None
                        if user_msgs_to_save:
                            tail = user_msgs_to_save[-1]
                            last_msg = tail.get("content") or msg_text
                            reply_sender_name = (
                                tail.get("sender_name") or sender_name or "User"
                            )
                            reply_sender_id = tail.get("sender_id") or sender_id
                            latest_mid = tail.get("_msg_id")

                        conversation = await self._get_conversation(task.event)
                        try:
                            self._preapply_poke_affinity(
                                umo=umo,
                                event=task.event,
                                group_id=str(gid or ""),
                                sender_id=str(reply_sender_id or ""),
                            )
                        except Exception:
                            pass

                        # Re-build memory context using latest pending message
                        memory_context, admin_protocols_text = (
                            await self._await_task_stage(
                                task,
                                "build_memory_context",
                                self.build_memory_context(
                                    last_msg,
                                    gid,
                                    reply_sender_name,
                                    history_msgs,
                                    reply_sender_id,
                                    conversation,
                                    is_group=bool(task.event.get_group_id()),
                                    task_type="at",
                                    self_id=(
                                        task.event.get_self_id()
                                        if task.event
                                        else ""
                                    ),
                                    recent_participants=[
                                        (str(_sid), str(_s or "User"))
                                        for (_mid, _t, _s, _sid, _ts) in (raw_pending or [])
                                        if _sid
                                    ],
                                ),
                            )
                        )

                        cur_mid = None
                        try:
                            cur_mid = getattr(task.event.message_obj, "message_id", None)
                        except Exception:
                            cur_mid = None
                        cur_mid_str = str(latest_mid or cur_mid or "").strip()
                        cur_ref_idx = None
                        if cur_mid_str and index_to_id:
                            for k, v in index_to_id.items():
                                if str(v) == cur_mid_str:
                                    cur_ref_idx = k
                                    break
                        if cur_ref_idx is not None:
                            reply_line = f"{format_sender(reply_sender_name, reply_sender_id, sender_tags)}: [REF:{cur_ref_idx}] {last_msg}"
                        else:
                            reply_line = f"{format_sender(reply_sender_name, reply_sender_id, sender_tags)}: {last_msg}"
                        target_block = self._build_conversation_target_block(
                            umo=umo,
                            meta=self.state.get_message_meta(umo, cur_mid_str),
                            sender_name=reply_sender_name,
                            sender_id=str(reply_sender_id or ""),
                            sender_tags=sender_tags,
                        )

                        prompt = build_focus_prompt(
                            umo,
                            self.config,
                            last_msg,
                            reply_line,
                            history_lines=pend_lines,
                            mood_text=self.state.get_last_mood(umo),
                            target_block=target_block,
                        )

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
                            replied = await self._await_task_stage(
                                task,
                                "invoke_llm",
                                self.llm_client.invoke_llm(
                                    umo,
                                    prompt,
                                    memory_context,
                                    event=task.event,
                                    task_type="at",
                                    instruction_text=instruction,
                                    admin_protocols_text=admin_protocols_text,
                                    user_msgs_to_save=user_msgs_to_save,
                                    image_urls=task.image_urls,
                                    created_at=task.created_at,
                                    quote_candidates=quote_candidates,
                                    at_candidates=at_candidates,
                                    impression_mode="off",
                                ),
                            )
                        finally:
                            self.state.set_processing(umo, False)

                        if replied:
                            self._maybe_update_affinity_from_mood(
                                umo=umo,
                                task_type="at",
                                event=task.event,
                                group_id=str(gid or ""),
                                sender_id=str(sender_id or ""),
                                history_msgs=history_msgs,
                            )

                        dur_min, dur_max, dur_step_max, dur_base = self._focus_duration_bounds()
                        cycle_min, _cycle_max, _cycle_step_max, _base_delay = self._focus_cycle_bounds()
                        remaining = float(self.state.get_focus_remaining_sec(umo))
                        at_extend_base = max(1, int(cycle_min or 1))
                        at_extend_cap = max(1, int(dur_step_max))
                        decay = 1.0
                        if dur_max > 0:
                            decay = 1.0 - (remaining / float(dur_max))
                        decay = self._clamp_float(decay, 0.35, 1.0)
                        focus_at_extend = int(
                            max(
                                1,
                                min(float(at_extend_base), float(at_extend_cap)) * decay,
                            )
                        )
                        remaining_int = int(remaining) + (1 if (remaining - int(remaining)) > 0 else 0)
                        focus_duration = max(int(dur_base), remaining_int) + focus_at_extend
                        if focus_duration < dur_min:
                            focus_duration = dur_min
                        elif focus_duration > dur_max:
                            focus_duration = dur_max
                        self.state.set_focus(umo, focus_duration)
                        if replied:
                            if consumed_count > 0:
                                self.state.clear_pending_first_n(umo, consumed_count)
                            send_ts = time.time()
                            self._last_send_ts[umo] = send_ts
                            self.state.clear_pending_up_to(umo, send_ts)
                            logger.info(
                                f"[mem_chat] AT phase finished for {umo}. Refreshed Focus Mode (Duration: {focus_duration}s, +{focus_at_extend}s)."
                            )
                        else:
                            logger.info(
                                f"[mem_chat] AT phase failed/interrupted for {umo}. Focus Mode still active. Messages retained in pending."
                            )
                            if self.state.is_in_focus(umo):
                                self._deferred_focus[umo] = True

                        deferred_focus = self._deferred_focus.pop(umo, False)
                        if replied and self.state.is_in_focus(umo):
                            await self.trigger_view_cycle(umo, "focus", task.event, [])
                        elif (
                            deferred_focus
                            and self.state.is_in_focus(umo)
                        ):
                            await self.trigger_view_cycle(umo, "focus", task.event, [])

                    elif task.task_type == "focus" or task.task_type == "read_air":
                        raw_pending = self.state.get_pending_messages(umo)
                        raw_pending = [
                            m
                            for m in (raw_pending or [])
                            if not self.state.is_no_reply_message_id(
                                umo, m[0] if m else None
                            )
                        ]
                        if not raw_pending:
                            logger.info(
                                f"[mem_chat] {task.task_type} task found no pending messages for {umo}. Skipping."
                            )
                            self.state.set_last_cycle_cutoff(umo, float(task.created_at or time.time()))
                            continue
                        cycle_cutoff_ts = float(task.created_at or time.time())
                        last_cycle_cutoff = float(self.state.get_last_cycle_cutoff(umo) or 0.0)
                        raw_pending = [
                            m
                            for m in raw_pending
                            if float(m[4] or 0.0) > last_cycle_cutoff
                            and float(m[4] or 0.0) <= cycle_cutoff_ts
                        ]
                        if not raw_pending:
                            logger.info(
                                f"[mem_chat] {task.task_type} task has no new pending window for {umo}. Skipping."
                            )
                            self.state.set_last_cycle_cutoff(umo, cycle_cutoff_ts)
                            self.state.clear_pending_up_to(umo, cycle_cutoff_ts)
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
                                self.state.clear_pending_up_to(umo, cycle_cutoff_ts)
                                self.state.set_last_cycle_cutoff(umo, cycle_cutoff_ts)
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
                                self.state.set_last_cycle_cutoff(umo, cycle_cutoff_ts)
                                self.state.clear_pending_up_to(umo, cycle_cutoff_ts)
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
                                        self.state.set_last_cycle_cutoff(umo, cycle_cutoff_ts)
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
                                self.state.clear_pending_up_to(umo, cycle_cutoff_ts)
                                self.state.set_last_cycle_cutoff(umo, cycle_cutoff_ts)
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
                                self.state.set_last_cycle_cutoff(
                                    umo, max(cycle_cutoff_ts, last_send_ts)
                                )
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
                                        "_msg_id": _mid,
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

                        gid = umo.split(":")[-1] if ":" in umo else ""  # Approximate
                        sender_tags = load_sender_tags(gid)

                        reply_line = f"{format_sender(sname, sid, sender_tags)}: {last_msg}"
                        prompt_last_msg = last_msg
                        prompt_reply_line = reply_line

                        # Context
                        cid = await conv_mgr.get_curr_conversation_id(umo)
                        conversation = (
                            await conv_mgr.get_conversation(umo, cid) if cid else None
                        )
                        try:
                            self._preapply_poke_affinity(
                                umo=umo,
                                event=task.event,
                                group_id=str(gid or ""),
                                sender_id=str(sid or ""),
                            )
                        except Exception:
                            pass
                        memory_context, admin_protocols_text = (
                            await self._await_task_stage(
                                task,
                                "build_memory_context",
                                self.build_memory_context(
                                    last_msg,
                                    gid,
                                    sname,
                                    history_msgs,
                                    sid,
                                    conversation,
                                    is_group=bool(task.event.get_group_id())
                                    if task.event
                                    else False,
                                    task_type=task.task_type,
                                    self_id=(
                                        task.event.get_self_id()
                                        if task.event
                                        else ""
                                    ),
                                    recent_participants=[
                                        (str(_sid), str(_s or "User"))
                                        for (_mid, _t, _s, _sid, _ts) in (raw_pending or [])
                                        if _sid
                                    ],
                                ),
                            )
                        )

                        quote_candidates = []
                        prompt_lines = []
                        index_to_id = {}
                        at_candidates = []

                        try:
                            from .ref_context import build_ref_context

                            ref_window = int(self._get_conf("ref_context_window", 0))
                            if ref_window <= 0:
                                ref_window = int(self._get_conf("at_pending_max_msgs", 30))
                            include_referenced = bool(
                                self._get_conf("ref_context_include_referenced", True)
                            )
                            prompt_lines, quote_candidates, index_to_id = build_ref_context(
                                raw_pending if raw_pending else [],
                                None,
                                None,
                                get_reply_target=lambda mid: self.state.get_reply_target(
                                    umo, mid
                                ),
                                get_reply_summary=lambda rid: self.state.get_reply_summary(
                                    umo, rid
                                ),
                                get_message_meta=lambda mid: self.state.get_message_meta(
                                    umo, mid
                                ),
                                max_items=ref_window,
                                include_referenced=include_referenced,
                                time_gap_min_minutes=int(
                                    self._get_conf("history_time_gap_minutes", 15)
                                ),
                                time_marker_every_n_messages=int(
                                    self._get_conf("history_time_marker_every_n", 4)
                                ),
                                sender_tags=sender_tags,
                            )
                            seen_sids = set()
                            for _m, _t, _s, _sid, _ts in raw_pending or []:
                                if _sid and _sid not in seen_sids:
                                    seen_sids.add(_sid)
                                    at_candidates.append(_sid)
                        except Exception:
                            quote_candidates = []
                            prompt_lines = []
                            index_to_id = {}

                        cur_ref_idx = None
                        cur_mid_str = str(last_msg_info[0] or "").strip()
                        if cur_mid_str and index_to_id:
                            for k, v in index_to_id.items():
                                if str(v) == cur_mid_str:
                                    cur_ref_idx = k
                                    break
                        if cur_ref_idx is not None:
                            reply_line = f"{format_sender(sname, sid, sender_tags)}: [REF:{cur_ref_idx}] {last_msg}"
                            prompt_reply_line = reply_line
                        target_block = self._build_conversation_target_block(
                            umo=umo,
                            meta=self.state.get_message_meta(umo, cur_mid_str),
                            sender_name=sname,
                            sender_id=str(sid or ""),
                            sender_tags=sender_tags,
                        )

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
                                prompt_last_msg,
                                prompt_reply_line,
                                history_lines=prompt_lines,
                                mood_text=self.state.get_last_mood(umo),
                                target_block=target_block,
                            )
                            hb_ts = self.state.get_heartbeat_start(umo)
                            if hb_ts:
                                ts_str = time.strftime(
                                    "%Y-%m-%d %H:%M:%S", time.localtime(hb_ts)
                                )
                                prompt = f"【心跳开始】{ts_str}\n" + prompt
                        elif task.task_type == "focus":
                            use_decision = bool(
                                self._get_conf("focus_non_at_use_decision_prompt", False)
                            )
                            self_id = self.state.get_self_id(umo)
                            if use_decision and self_id:
                                has_at = False
                                for _mid, _t, _s, _sid, _ts in raw_pending:
                                    if _t and (
                                        f"@{self_id}" in _t
                                        or f"[At:{self_id}]" in _t
                                    ):
                                        has_at = True
                                        break
                                if has_at:
                                    use_decision = False
                            if use_decision:
                                prompt = build_decision_prompt(
                                    umo,
                                    self.config,
                                    prompt_last_msg,
                                    prompt_reply_line,
                                    history_lines=prompt_lines,
                                    mood_text=self.state.get_last_mood(umo),
                                    target_block=target_block,
                                )
                            else:
                                prompt = build_focus_prompt(
                                    umo,
                                    self.config,
                                    prompt_last_msg,
                                    prompt_reply_line,
                                    history_lines=prompt_lines,
                                    mood_text=self.state.get_last_mood(umo),
                                    target_block=target_block,
                                )
                            fs_ts = self.state.get_focus_start(umo)
                            if fs_ts:
                                ts_str = time.strftime(
                                    "%Y-%m-%d %H:%M:%S", time.localtime(fs_ts)
                                )
                                prompt = f"【专注开始】{ts_str}\n" + prompt
                        else:
                            prompt = build_focus_prompt(
                                umo,
                                self.config,
                                prompt_last_msg,
                                prompt_reply_line,
                                history_lines=prompt_lines,
                                mood_text=self.state.get_last_mood(umo),
                                target_block=target_block,
                            )

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
                            impression_mode = "off"
                            if bool(self._get_conf("impression_stream_enabled", True)):
                                if task.task_type == "read_air":
                                    if bool(
                                        self._get_conf(
                                            "impression_stream_read_air_delta", True
                                        )
                                    ):
                                        impression_mode = "delta"
                                elif task.task_type == "focus":
                                    rem = self.state.get_focus_remaining_sec(umo)
                                    end_thr = float(
                                        self._get_conf(
                                            "impression_focus_end_threshold_sec", 10
                                        )
                                    )
                                    if bool(
                                        self._get_conf(
                                            "impression_stream_focus_final", True
                                        )
                                    ) and rem > 0 and rem <= max(1.0, end_thr):
                                        impression_mode = "final"
                                    elif bool(
                                        self._get_conf(
                                            "impression_stream_focus_delta", True
                                        )
                                    ):
                                        impression_mode = "delta"

                            replied = await self._await_task_stage(
                                task,
                                "invoke_llm",
                                self.llm_client.invoke_llm(
                                    umo,
                                    prompt,
                                    memory_context,
                                    event=task.event,
                                    task_type=task.task_type,
                                    instruction_text="",
                                    admin_protocols_text=admin_protocols_text,
                                    user_msgs_to_save=user_msgs_to_save,
                                    image_urls=task.image_urls,
                                    quote_candidates=quote_candidates,
                                    at_candidates=at_candidates,
                                    created_at=task.created_at,
                                    impression_mode=impression_mode,
                                ),
                            )
                        finally:
                            self.state.set_processing(umo, False)

                        if replied:
                            self._maybe_update_affinity_from_mood(
                                umo=umo,
                                task_type=str(task.task_type or ""),
                                event=task.event,
                                group_id=str(gid or ""),
                                sender_id=str(sid or ""),
                                history_msgs=history_msgs,
                            )

                        # Update last send time for dedup window
                        if replied:
                            consumed_count = len(raw_pending)
                            if consumed_count > 0:
                                self.state.clear_pending_first_n(umo, consumed_count)
                            send_ts = time.time()
                            self._last_send_ts[umo] = send_ts
                            self.state.clear_pending_up_to(umo, send_ts)
                            self.state.set_last_cycle_cutoff(umo, cycle_cutoff_ts)
                            logger.info(
                                f"[mem_chat] {task.task_type.upper()} phase finished for {umo}."
                            )
                        else:
                            logger.info(
                                f"[mem_chat] {task.task_type.upper()} task failed/interrupted for {umo}. Messages retained in pending."
                            )
                            if task.task_type == "focus" and self.state.is_in_focus(umo):
                                self._deferred_focus[umo] = True

                        # If read_air mode and replied, enter focus mode
                        if task.task_type == "read_air" and replied:
                            dur_min, dur_max, _dur_step_max, dur_base = self._focus_duration_bounds()
                            focus_duration = int(dur_base)
                            if focus_duration < dur_min:
                                focus_duration = dur_min
                            elif focus_duration > dur_max:
                                focus_duration = dur_max
                            self.state.set_focus(
                                umo, focus_duration
                            )  # This sets state to FOCUS
                            logger.info(
                                f"[mem_chat] Read Air triggered reply. Entering Focus Mode (Duration: {focus_duration}s)."
                            )

                        # If focus mode task completed, apply cooling
                        if replied:
                            cycle_min, cycle_max, cycle_step_max, base_delay = self._focus_cycle_bounds()

                            if task.task_type == "focus":
                                curr_delay = self.state.get_focus_delay(
                                    umo, default=base_delay
                                )
                                next_delay = self._step_towards(
                                    float(curr_delay),
                                    float(cycle_max),
                                    float(cycle_step_max),
                                    float(cycle_min),
                                    float(cycle_max),
                                )
                                self.state.set_focus_delay(umo, next_delay, min_val=cycle_min)
                                logger.info(
                                    f"[mem_chat] FOCUS reply applied. Next focus delay set to {self.state.get_focus_delay(umo, default=base_delay):.2f}s."
                                )

                        if replied and self.state.is_in_focus(umo):
                            await self.trigger_view_cycle(umo, "focus", task.event, [])
                        elif self._deferred_focus.pop(umo, False) and self.state.is_in_focus(umo):
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
        self,
        umo: str,
        count: int,
        event: AstrMessageEvent | None = None,
        task_type: str = "",
    ) -> list[str]:
        try:
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(
                umo
            )
            messages: list[str] = []
            task_norm = str(task_type or "").strip().lower()
            max_lines = max(1, int(count or 1))
            fetch_factor = int(self._get_conf("history_fetch_factor", 4) or 4)
            if fetch_factor < 1:
                fetch_factor = 1
            fetch_cap = int(self._get_conf("history_fetch_cap", 120) or 120)
            if fetch_cap < max_lines:
                fetch_cap = max_lines
            fetch_count = min(fetch_cap, max_lines * fetch_factor)

            group_id = ""
            try:
                if event is not None:
                    group_id = str(event.get_group_id() or "").strip()
            except Exception:
                group_id = ""
            if not group_id:
                group_id = str(umo or "").rsplit(":", 1)[-1].strip()
            sender_tags = load_sender_tags(group_id) if group_id else {}

            char_budget_default = int(self._get_conf("history_char_budget", 3200) or 3200)
            char_budget_at = int(
                self._get_conf("history_char_budget_at", max(char_budget_default, 4200))
                or max(char_budget_default, 4200)
            )
            char_budget_focus = int(
                self._get_conf(
                    "history_char_budget_focus", min(char_budget_default, 3200)
                )
                or min(char_budget_default, 3200)
            )
            char_budget_read_air = int(
                self._get_conf(
                    "history_char_budget_read_air", min(char_budget_default, 2400)
                )
                or min(char_budget_default, 2400)
            )
            if task_norm == "at":
                char_budget = char_budget_at
            elif task_norm == "focus":
                char_budget = char_budget_focus
            elif task_norm == "read_air":
                char_budget = char_budget_read_air
            else:
                char_budget = char_budget_default
            if char_budget < 400:
                char_budget = 400

            min_lines_default = int(self._get_conf("history_min_lines", 4) or 4)
            min_lines_focus = int(
                self._get_conf("history_min_lines_focus", min_lines_default) or min_lines_default
            )
            min_lines_read_air = int(
                self._get_conf("history_min_lines_read_air", min_lines_default) or min_lines_default
            )
            if task_norm == "focus":
                min_lines = min_lines_focus
            elif task_norm == "read_air":
                min_lines = min_lines_read_air
            else:
                min_lines = min_lines_default
            min_lines = max(1, min(min_lines, max_lines))

            include_assistant = bool(self._get_conf("history_include_assistant", True))

            if curr_cid:
                conversation = await self.context.conversation_manager.get_conversation(
                    umo, curr_cid
                )
                if conversation and getattr(conversation, "history", None):
                    import json

                    try:
                        hist = json.loads(conversation.history)
                        take = hist[-fetch_count:]

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

                        normalized_rows: list[tuple[str, str, str, str]] = []
                        for m in take:
                            try:
                                if isinstance(m, dict) and bool(m.get("_no_reply", False)):
                                    continue
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

                                sender_part = format_sender(sname, sid, sender_tags) or sname
                                if not sid and role == "assistant":
                                    sender_part = (
                                        self.state.get_self_name(umo) or "Assistant"
                                    )

                                if sender_part and content:
                                    normalized_rows.append((role, sender_part, sid, content))
                                elif content:
                                    normalized_rows.append((role, "", sid, content))
                            except Exception:
                                continue
                        total_chars = 0
                        selected: list[str] = []
                        for role, sender_part, _sid, content in reversed(normalized_rows):
                            if not include_assistant and role == "assistant":
                                continue
                            line = (
                                f"{sender_part}: {content}"
                                if sender_part and content
                                else str(content or "").strip()
                            )
                            if not line:
                                continue
                            line_chars = len(line)
                            if (
                                total_chars + line_chars > char_budget
                                and len(selected) >= min_lines
                            ):
                                continue
                            selected.append(line)
                            total_chars += line_chars
                            if len(selected) >= max_lines:
                                break
                        messages = list(reversed(selected))
                    except Exception:
                        pass

            # If no messages found from conversation (or it was empty), try fallback to event if provided
            if not messages and event:
                mt = event.message_str or event.get_message_outline()
                if mt:
                    sname = event.get_sender_name() or ""
                    sid = event.get_sender_id() or ""
                    sp = format_sender(sname, sid, sender_tags) or sname
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
        logger.info(
            f"[mem_chat] Summarize session start: umo={umo} group={group_id} sender={sender_name}/{sender_id}"
        )
        # Get recent messages from this user (50 messages)
        history = await self.db_mgr.get_user_specific_history(
            group_id, sender_id, limit=50
        )
        if not history:
            logger.info(
                f"[mem_chat] Summarize session skipped: no history for {sender_name}/{sender_id}"
            )
            return

        history_text = "\n".join(history)

        v2_enabled = bool(self._get_conf("profile_v2_enabled", True))
        if not v2_enabled:
            prompt = (
                f"请根据以下 {sender_name} (ID: {sender_id}) 的最近聊天记录，"
                f"总结对该用户的印象，以及提取关键的对话信息（如偏好、正在讨论的话题等）。"
                f"请保持简洁，不要超过200字。\n\n"
                f"{history_text}"
            )
            system_prompt = "你是一个善于总结对话的助手。请客观地总结用户的特征和对话内容。"
            summary = await self.llm_client.generate_text(
                prompt, system_prompt=system_prompt, umo=umo
            )
            if summary:
                self.db_mgr.save_impression(group_id, sender_id, summary, sender_name)
                impression_id = self.db_mgr.hash_id(summary, group_id, "impression")
                model_name = str(
                    self.config.get("embedding_openai_model", "text-embedding-3-small")
                    or "text-embedding-3-small"
                )
                await self.db_mgr.get_or_compute_embedding(
                    impression_id, summary, group_id, model_name, type="impression"
                )
            return

        prompt = (
            "你是一个“用户画像抽取器”。请根据下方聊天记录，输出一个 JSON 对象（只输出 JSON，不要 Markdown）。\n"
            "要求：\n"
            "1) 只基于对话内容，不要推测隐私信息（年龄/地域/职业/性取向等）。\n"
            "2) impression_summary 为 1-2 句短摘要，总长不超过 80 字。\n"
            "3) tags 为短标签数组，最多 8 条，每条 2-8 字。\n"
            "4) traits/preferences/dislikes/current_topics/boundaries/uncertain 每项用 5-20 字短语；每个数组最多 5 条；可以为空数组。\n"
            "5) 用于群聊辅助记忆：尽量提取可复用的长期特征/偏好，也可记录近期话题。\n"
            "JSON 字段：\n"
            '{"impression_summary":"","tags":[],"traits":[],"preferences":[],"dislikes":[],"current_topics":[],"boundaries":[],"uncertain":[]}\n\n'
            f"【用户】{sender_name} / {sender_id}\n"
            f"【聊天记录】\n{history_text}"
        )
        system_prompt = "你擅长从对话中提取稳定且可验证的用户画像要点。"

        raw = await self.llm_client.generate_text(prompt, system_prompt=system_prompt, umo=umo)
        if not raw:
            logger.info(
                f"[mem_chat] Summarize session skipped: empty output for {sender_name}/{sender_id}"
            )
            return

        data = None
        try:
            data = json.loads(raw.strip())
        except Exception:
            try:
                m = re.search(r"\{[\s\S]*\}", raw)
                if m:
                    data = json.loads(m.group(0))
            except Exception:
                data = None
        if not isinstance(data, dict):
            data = {
                "impression_summary": "",
                "tags": [],
                "traits": [],
                "preferences": [],
                "dislikes": [],
                "current_topics": [],
                "boundaries": [],
                "uncertain": [],
            }

        def _as_list(key: str) -> list[str]:
            v = data.get(key)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            if isinstance(v, str) and v.strip():
                return [v.strip()]
            return []

        def _as_text(key: str, max_len: int) -> str:
            v = data.get(key)
            if v is None:
                return ""
            s = str(v).replace("\r\n", "\n").replace("\r", "\n").strip()
            s = " ".join(s.split()).strip()
            if not s:
                return ""
            if max_len > 0 and len(s) > max_len:
                return s[:max_len].rstrip()
            return s

        profile = {
            "impression_summary": _as_text("impression_summary", 80),
            "tags": _as_list("tags")[:8],
            "traits": _as_list("traits")[:5],
            "preferences": _as_list("preferences")[:5],
            "dislikes": _as_list("dislikes")[:5],
            "current_topics": _as_list("current_topics")[:5],
            "boundaries": _as_list("boundaries")[:5],
            "uncertain": _as_list("uncertain")[:5],
        }

        def _build_canonical(p: dict) -> str:
            parts: list[str] = []
            if p.get("impression_summary"):
                parts.append("印象：" + str(p["impression_summary"]))
            if p.get("tags"):
                try:
                    tags = [str(x).strip() for x in (p.get("tags") or []) if str(x).strip()]
                except Exception:
                    tags = []
                if tags:
                    parts.append("标签：" + "、".join(tags[:8]))
            if p.get("traits"):
                parts.append("特征：" + "；".join(p["traits"]))
            if p.get("preferences"):
                parts.append("偏好：" + "；".join(p["preferences"]))
            if p.get("dislikes"):
                parts.append("不喜欢：" + "；".join(p["dislikes"]))
            if p.get("current_topics"):
                parts.append("近期话题：" + "；".join(p["current_topics"]))
            if p.get("boundaries"):
                parts.append("边界：" + "；".join(p["boundaries"]))
            if p.get("uncertain"):
                parts.append("不确定：" + "；".join(p["uncertain"]))
            txt = "。".join([s for s in parts if s]).strip()
            return txt[:600]

        canonical_text = _build_canonical(profile)
        profile_json = json.dumps(profile, ensure_ascii=False)
        self.db_mgr.upsert_user_profile(
            group_id=group_id,
            user_id=str(sender_id),
            display_name=str(sender_name or "User"),
            profile_json=profile_json,
            canonical_text=canonical_text,
            msg_count=None,
            version=3,
        )
        try:
            self.db_mgr.increment_user_profile_msg_count(group_id, str(sender_id), len(history))
        except Exception:
            pass

        try:
            for t in profile.get("tags") or []:
                self.db_mgr.upsert_profile_fact(group_id, str(sender_id), "tag", t, polarity=1, confidence=0.6, evidence_count=1)
            for t in profile.get("traits") or []:
                self.db_mgr.upsert_profile_fact(group_id, str(sender_id), "trait", t, polarity=1, confidence=0.7, evidence_count=1)
            for t in profile.get("preferences") or []:
                self.db_mgr.upsert_profile_fact(group_id, str(sender_id), "like", t, polarity=1, confidence=0.7, evidence_count=1)
            for t in profile.get("dislikes") or []:
                self.db_mgr.upsert_profile_fact(group_id, str(sender_id), "dislike", t, polarity=-1, confidence=0.7, evidence_count=1)
            for t in profile.get("current_topics") or []:
                self.db_mgr.upsert_profile_fact(group_id, str(sender_id), "topic", t, polarity=1, confidence=0.6, evidence_count=1)
            for t in profile.get("boundaries") or []:
                self.db_mgr.upsert_profile_fact(group_id, str(sender_id), "boundary", t, polarity=1, confidence=0.8, evidence_count=1)
            for t in profile.get("uncertain") or []:
                self.db_mgr.upsert_profile_fact(group_id, str(sender_id), "uncertain", t, polarity=0, confidence=0.4, evidence_count=1)
        except Exception:
            pass

        model_name = str(
            self.config.get("embedding_openai_model", "text-embedding-3-small")
            or "text-embedding-3-small"
        )
        impr_content = f"IMPR2 gid={group_id} uid={sender_id} name={sender_name} :: {canonical_text}"
        impr_mid = f"impr2::{group_id}::{sender_id}"
        await self.db_mgr.get_or_compute_embedding(
            impr_mid, impr_content, group_id, model_name, type="impression"
        )

    async def build_memory_context(
        self,
        query: str,
        group_id: str,
        sender_name: str | None,
        history_msgs: list[str] | None = None,
        sender_id: str | None = None,
        conversation=None,
        is_group: bool = True,
        task_type: str = "",
        recent_participants: list[tuple[str, str]] | None = None,
        self_id: str | None = None,
    ) -> tuple[str, str]:
        """Build the full context including memory, persona, and knowledge.

        Args:
            self_id: The bot's own QQ id. Used to detect her own lines in the
                history window; when omitted, the style rewrite is skipped
                rather than applied to everyone's messages.
        """
        # Blocks are stored as (priority, text); higher priority claims the
        # global budget first and the survivors keep their original order.
        context_parts: list[tuple[int, str]] = []

        def _alice_lines_only(text: str, max_lines: int = 6) -> str:
            """Keep only the lines Alice herself speaks in a script excerpt."""
            lines = []
            for ln in str(text or "").split("\n"):
                m = re.match(r"^\[(?:アリス|爱丽丝)\]\s*(.+)$", ln.strip())
                if not m:
                    continue
                said = m.group(1).strip()
                if len(said) <= 2:
                    continue
                lines.append(said)
                if len(lines) >= max_lines:
                    break
            if not lines:
                return _compact_injected_text(text, 320)
            return " ".join(f"「{x}」" for x in lines)

        def _add_block(text: str, priority: int) -> None:
            cleaned = str(text or '').strip()
            if cleaned:
                context_parts.append((priority, cleaned))
        admin_protocols_text = ""
        task_norm = str(task_type or "").strip().lower()
        is_at_task = task_norm == "at"

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

        def _normalize_impression_text(text: str) -> str:
            raw = str(text or "").strip()
            if not raw:
                return ""
            if not ((raw.startswith("{") and raw.endswith("}")) or (raw.startswith("[") and raw.endswith("]"))):
                return raw
            try:
                data = json.loads(raw)
            except Exception:
                return raw
            parts = []
            seen = set()

            def _add_part(val: str):
                v = str(val).strip()
                if not v:
                    return
                if v in seen:
                    return
                seen.add(v)
                parts.append(v)

            def _walk(obj):
                if isinstance(obj, str):
                    _add_part(obj)
                    return
                if isinstance(obj, list):
                    for it in obj:
                        _walk(it)
                    return
                if isinstance(obj, dict):
                    for k in [
                        "summary",
                        "impression_summary",
                        "impression",
                        "profile",
                        "description",
                        "persona",
                        "tags",
                        "traits",
                        "text",
                        "content",
                        "notes",
                        "facts",
                        "preferences",
                        "likes",
                        "dislikes",
                        "current_topics",
                        "topics",
                        "boundaries",
                        "uncertain",
                    ]:
                        if k in obj:
                            _walk(obj.get(k))
                    for v in obj.values():
                        _walk(v)

            _walk(data)
            if parts:
                return "；".join(parts)
            return raw

        def _emoji_desc_key(text: str) -> str:
            s = str(text or "")
            if "[图片内容:" not in s:
                return ""
            try:
                m = re.search(r"\[图片内容:\s*(.*?)\]", s)
                core = m.group(1) if m else s[s.find("[图片内容:") :]
            except Exception:
                core = s
            core = core.strip().lower()
            # Drop sender prefix and most punctuation for coarse semantic grouping.
            core = re.sub(r"^[^:]{0,40}:\s*", "", core)
            core = re.sub(r"[^\w\u4e00-\u9fff]+", "", core)
            if len(core) > 96:
                core = core[:96]
            return core

        def _memory_body_key(text: str) -> str:
            s = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
            s = re.sub(r"\s+", " ", s).strip()
            if not s:
                return ""
            # Strip optional sender prefix like "name/qq: " for echo filtering.
            s = re.sub(r"^[^:\n]{1,80}:\s*", "", s)
            # Strip inline ref marker if present.
            s = re.sub(r"^\[REF:\d+\]\s*", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\s+", " ", s).strip().lower()
            return s[:400]

        qemb = await self.db_mgr.get_embedding(query)
        knowledge_task = self.db_mgr.semantic_recall_with_qemb(
            qemb,
            group_id,
            limit=5,
            threshold=0.0,
            type="knowledge",
        )
        persona_rag_limit = max(5, int(self._get_conf("max_injected_persona_rag", 5)))
        persona_rag_task = self.db_mgr.semantic_recall_with_qemb(
            qemb,
            group_id,
            limit=persona_rag_limit,
            threshold=0.0,
            type="persona_rag",
        )

        limit_mem_conf = int(self._get_conf("max_injected_memories", 40))
        limit_mem = max(1, limit_mem_conf)
        memory_scan_limit = None
        impression_scan_limit = None
        impression_v3_scan_limit = None
        if is_at_task:
            limit_mem = max(
                1,
                min(
                    limit_mem,
                    int(self._get_conf("max_injected_memories_at", 18)),
                ),
            )
            memory_scan_limit = max(
                0,
                int(self._get_conf("semantic_recall_scan_limit_memory_at", 1200) or 0),
            )
            impression_scan_limit = max(
                0,
                int(self._get_conf("semantic_recall_scan_limit_impression_at", 800) or 0),
            )
            impression_v3_scan_limit = max(
                0,
                int(self._get_conf("semantic_recall_scan_limit_impression_v3_at", 800) or 0),
            )
        if task_norm in ("focus", "read_air"):
            limit_mem = max(
                1,
                min(
                    limit_mem,
                    int(self._get_conf("max_injected_memories_non_at", 18)),
                ),
            )
        memories_task = self.db_mgr.semantic_recall_with_qemb(
            qemb,
            group_id,
            limit=limit_mem,
            threshold=0.0,
            type="memory",
            scan_limit=memory_scan_limit,
        )

        # Reflections are induced conclusions ("she dodges tech-support asks
        # with jokes"), not raw lines, so a handful of them replaces a much
        # larger block of individual memories.
        reflections_task = None
        if bool(self._get_conf("reflection_inject_enabled", True)):
            try:
                limit_reflect = max(
                    1, int(self._get_conf("max_injected_reflections", 5))
                )
            except (TypeError, ValueError):
                limit_reflect = 5
            reflections_task = self.db_mgr.semantic_recall_with_qemb(
                qemb,
                group_id,
                limit=limit_reflect,
                threshold=0.0,
                type="reflection",
                scan_limit=memory_scan_limit,
            )

        impression_inject_enabled = bool(
            self._get_conf("enable_impression_injection", True)
        )
        impression_semantic_enabled = bool(
            self._get_conf("impression_semantic_enabled", True)
        )
        if is_at_task:
            impression_semantic_enabled = bool(
                self._get_conf("impression_semantic_at_enabled", False)
            )
        impression_v3_enabled = bool(self._get_conf("impression_v3_enabled", True))
        impression_active_only = bool(
            self._get_conf("impression_active_only", True)
        )
        impression_merge_recent_participants = bool(
            self._get_conf("impression_merge_recent_participants", True)
        )
        impression_recent_max_persons = int(
            self._get_conf("impression_recent_max_persons", 6)
        )
        if impression_recent_max_persons < 1:
            impression_recent_max_persons = 1
        if is_at_task:
            impression_recent_max_persons = max(
                1,
                min(
                    impression_recent_max_persons,
                    int(self._get_conf("impression_recent_max_persons_at", 6)),
                ),
            )
        impression_sender_pinned = bool(
            self._get_conf("impression_sender_pinned", True)
        )
        max_persons_conf = int(
            self._get_conf(
                "impression_related_max_persons",
                self._get_conf("max_impression_persons", 2),
            )
        )
        max_persons = max(1, min(5, max_persons_conf))
        if is_at_task:
            max_persons = max(
                1,
                min(max_persons, int(self._get_conf("impression_related_max_persons_at", 3))),
            )
        max_per_person = int(self._get_conf("max_impressions_per_person", 3))
        if is_at_task:
            max_per_person = max(
                1,
                min(max_per_person, int(self._get_conf("max_impressions_per_person_at", 2))),
            )
        min_score_impr = float(
            self._get_conf(
                "impression_related_score_min",
                self._get_conf("impression_min_score", 0.35),
            )
        )
        impression_token_budget = int(
            self._get_conf("impression_token_budget", 400)
        )
        if is_at_task:
            impression_token_budget = max(
                0,
                min(
                    impression_token_budget,
                    int(self._get_conf("impression_token_budget_at", 700)),
                ),
            )
        if impression_inject_enabled and impression_semantic_enabled:
            impressions_task = self.db_mgr.semantic_recall_with_qemb(
                qemb,
                group_id,
                limit=max_persons * max_per_person,
                threshold=min_score_impr,
                type="impression",
                scan_limit=impression_scan_limit,
            )
        else:
            impressions_task = asyncio.sleep(0, result=[])
        if impression_inject_enabled and impression_semantic_enabled and impression_v3_enabled:
            impressions_v3_task = self.db_mgr.semantic_recall_impression_v3_with_qemb(
                qemb,
                group_id,
                limit=max_persons * max_per_person * 2,
                threshold=min_score_impr,
                scan_limit=impression_v3_scan_limit,
            )
        else:
            impressions_v3_task = asyncio.sleep(0, result=[])
        (
            raw_knowledge,
            raw_persona_rag,
            memories,
            recalled,
            recalled_v3,
            reflections,
        ) = await asyncio.gather(
            knowledge_task,
            persona_rag_task,
            memories_task,
            impressions_task,
            impressions_v3_task,
            reflections_task
            if reflections_task is not None
            else _empty_recall(),
        )

        knowledge_limit = 5
        if task_norm in ("focus", "read_air"):
            knowledge_limit = max(
                1, int(self._get_conf("max_injected_knowledge_non_at", 2))
            )

        knowledge = []
        high_quality = [k for k in raw_knowledge if k[1] >= 0.5]
        if len(high_quality) >= 2:
            knowledge = high_quality
        else:
            knowledge = raw_knowledge[:2]
        knowledge = knowledge[:knowledge_limit]
        if knowledge:
            k_text = "【相关知识】\n"
            for k in knowledge:
                content = _compact_injected_text(k[0], 450)
                if content:
                    k_text += f"- {content}\n"
            _add_block(k_text, 40)

        persona_rag_non_at_enabled = bool(self._get_conf("persona_rag_non_at_enabled", False))
        if task_norm in ("focus", "read_air") and not persona_rag_non_at_enabled:
            persona_rag = []
        else:
            persona_rag = list(raw_persona_rag[:persona_rag_limit])
        if len(persona_rag) < 5:
            try:
                fallback_items = await self.db_mgr.get_all_embeddings_by_type("persona_rag", limit=50)
            except Exception:
                fallback_items = []
            seen = {" ".join(str(x[0] or "").split()).strip() for x in persona_rag}
            for txt in fallback_items:
                key = " ".join(str(txt or "").split()).strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                persona_rag.append((str(txt), 0.0))
                if len(persona_rag) >= 5:
                    break
        persona_rag = persona_rag[:persona_rag_limit]
        if persona_rag:
            p_text = "【爱丽丝语料片段】\n（以下是爱丽丝本人在剧情里说过的话，学她的语气、用词和反应方式）\n"
            for p in persona_rag:
                content = _alice_lines_only(p[0])
                if content:
                    p_text += f"- {content}\n"
            _add_block(p_text, 30)

        admin_mode = str(self._get_conf("admin_protocols_inject_mode", "auto") or "")
        admin_mode = admin_mode.strip().lower()
        if is_group and admin_mode not in ("off", "false", "0", "none"):
            should = admin_mode == "always"
            if not should and admin_mode == "auto":
                should = self._should_inject_admin_protocols(query)
            if should:
                protocol_raw = self._load_admin_protocols_text()
                if protocol_raw:
                    admin_protocols_text = "【管理协议】\n" + protocol_raw + "\n"

        if reflections:
            r_text = (
                "【我的观察】\n"
                "（这是你自己从过去的聊天里归纳出来的结论，可以直接当作你的判断"
                "来用；但别照搬原句，也别在群里宣读它）\n"
            )
            seen_reflect: set[str] = set()
            for item in reflections:
                line = _compact_injected_text(item[0], 160)
                if not line:
                    continue
                # Drop the evidence pointer; it exists for offline auditing and
                # would only burn context if shown to the model.
                line = line.split("  [依据:", 1)[0].strip()
                key = re.sub(r"\s+", " ", line).strip().lower()
                if not key or key in seen_reflect:
                    continue
                seen_reflect.add(key)
                r_text += f"- {line}\n"
            if seen_reflect:
                _add_block(r_text, 80)

        if memories:
            m_text = "【相关记忆】\n"
            emoji_seen_cnt: dict[str, int] = {}
            mem_seen: set[str] = set()
            emoji_dropped = False
            recent_echo_keys: set[str] = set()
            try:
                qk = _memory_body_key(query)
                if qk:
                    recent_echo_keys.add(qk)
                for ln in (history_msgs or []):
                    hk = _memory_body_key(ln)
                    if hk:
                        recent_echo_keys.add(hk)
            except Exception:
                recent_echo_keys = set()
            echo_filtered = 0
            for m in memories:
                mem_max_chars = 180 if task_norm in ("focus", "read_air") else 220
                content = _compact_injected_text(m[0], mem_max_chars)
                if content:
                    body_key = _memory_body_key(content)
                    if body_key and body_key in recent_echo_keys:
                        echo_filtered += 1
                        continue
                    mem_key = re.sub(r"\s+", " ", content).strip().lower()
                    if mem_key in mem_seen:
                        continue
                    mem_seen.add(mem_key)
                    k = _emoji_desc_key(content)
                    if k:
                        c = int(emoji_seen_cnt.get(k, 0))
                        if c >= 2:
                            emoji_dropped = True
                            continue
                        emoji_seen_cnt[k] = c + 1
                    m_text += f"- {content}\n"
            if echo_filtered > 0:
                m_text += f"- （已跳过{echo_filtered}条与当前轮对话重复的记忆）\n"
            if emoji_dropped:
                m_text += "- （同类表情包描述已去重，每类最多保留2条）\n"
            _add_block(m_text, 70)

        def _build_alias_line(uid: str, name_hint: str | None = None) -> str:
            uid = str(uid or "").strip()
            if not uid or not group_id:
                return ""
            hint = re.sub(r"\s+", " ", str(name_hint or "")).strip()
            if hint.lower() in {"user", "unknown", "system"} or hint in {"未知", "匿名"}:
                hint = ""
            try:
                aliases = self.db_mgr.list_user_aliases(group_id, uid, limit=8)
            except Exception:
                aliases = []
            current = hint
            historical: list[str] = []
            seen: set[str] = set()
            if current:
                seen.add(current)
            for alias, is_current, _count, _first_seen, _last_seen in aliases or []:
                name = re.sub(r"\s+", " ", str(alias or "")).strip()
                if not name or name in seen:
                    continue
                if (not current) and is_current:
                    current = name
                    seen.add(name)
                    continue
                historical.append(name)
                seen.add(name)
                if len(historical) >= 5:
                    break
            if not current and not historical:
                return ""
            if current and historical:
                return f"- 昵称：当前 {current}；曾用 {'、'.join(historical[:5])}"
            if historical:
                return f"- 曾用昵称：{'、'.join(historical[:5])}"
            return f"- 当前昵称：{current}"

        def _build_user_impression_lines(uid: str, name_hint: str | None = None) -> list[str]:
            uid = str(uid or "").strip()
            if not uid or not group_id:
                return []
            lines: list[str] = []
            alias_line = _build_alias_line(uid, name_hint)
            if impression_v3_enabled:
                try:
                    prof_v3 = self.db_mgr.get_impression_profile_v3(group_id, uid)
                except Exception:
                    prof_v3 = None
                if prof_v3:
                    _v3_name, v3_summary, v3_tags, _v3_conf, _v3_ts = prof_v3
                    if v3_summary:
                        lines.append(f"- 近期印象：{v3_summary}")
                    if v3_tags:
                        lines.append("- 标签：" + "、".join(v3_tags[:8]))
                    try:
                        v3_facts = self.db_mgr.list_impression_facts_v3(
                            group_id, uid, limit=4
                        )
                    except Exception:
                        v3_facts = []
                    for fact, _reason, _conf, _ev, _ts in v3_facts or []:
                        if fact:
                            lines.append(f"- {fact}")
            if not lines:
                try:
                    prof = self.db_mgr.get_user_profile(group_id, uid)
                except Exception:
                    prof = None
                if prof:
                    _name, pjson, canon = prof
                    txt = _normalize_impression_text(canon or "") or _normalize_impression_text(
                        pjson or ""
                    )
                    if txt:
                        lines.append(f"- 印象：{txt}")
            if not lines:
                try:
                    impression = self.db_mgr.get_impression(group_id, uid)
                except Exception:
                    impression = None
                if impression:
                    txt = _normalize_impression_text(impression)
                    if txt:
                        lines.append(f"- 印象：{txt}")
            if alias_line:
                lines.insert(0, alias_line)
            return lines

        def _build_affinity_line(uid: str, name_hint: str | None = None) -> str:
            uid = str(uid or "").strip()
            if not uid or not group_id:
                return ""
            try:
                self.db_mgr.ensure_affinity_state(group_id, uid)
                # Per-user idle decay: if this user has been inactive for >idle_sec,
                # trigger immediate per-user decay before reading state so the LLM always
                # sees fresh (already-decayed) affinity.
                idle_sec = float(self._get_conf("affinity_decay_idle_sec", 3600))
                aff_state_raw = self.db_mgr.get_affinity_state(group_id, uid)
                if aff_state_raw:
                    _last_evt = aff_state_raw[3]
                    if (_last_evt or 0) > 0 and idle_sec > 0 and time.time() - float(_last_evt) > idle_sec:
                        self.db_mgr.decay_affinity_towards_neutral(
                            group_id,
                            step=float(self._get_conf("affinity_decay_step", 1.0)),
                            idle_sec=idle_sec,
                        )
                aff_state = self.db_mgr.get_affinity_state(group_id, uid)
            except Exception:
                aff_state = None
            if not aff_state:
                return ""
            affinity, trust, mood_bias, _last_evt, _updated = aff_state
            relation = "中性"
            if affinity >= 35:
                relation = "明显喜欢"
            elif affinity >= 12:
                relation = "偏喜欢"
            elif affinity <= -35:
                relation = "明显反感"
            elif affinity <= -12:
                relation = "偏反感"
            return f"- 关系倾向：{relation}"

        participants_ordered: list[tuple[str, str]] = []
        recent_window_uids: set[str] = set()

        def _extract_participants_from_lines(lines: list[str] | None) -> list[tuple[str, str]]:
            out: list[tuple[str, str]] = []
            for ln in lines or []:
                txt = str(ln or "").strip()
                if not txt:
                    continue
                m = re.match(r"^(?:\[[^\]]+\]\s*)?(.+?)/(\d{5,20})\s*:\s*", txt)
                if not m:
                    continue
                name = (m.group(1) or "").strip() or "User"
                uid = (m.group(2) or "").strip()
                if uid:
                    out.append((uid, name))
            return out
        if impression_merge_recent_participants:
            seen_uid: set[str] = set()
            for uid_raw, name_raw in recent_participants or []:
                uid = str(uid_raw or "").strip()
                if not uid or uid in seen_uid:
                    continue
                seen_uid.add(uid)
                participants_ordered.append((uid, str(name_raw or "User")))
            # Also include speakers appearing in the latest dialogue window.
            # history_msgs here is already the recent prompt window (read_air/focus limited by config).
            for uid, name in _extract_participants_from_lines((history_msgs or []) + [query]):
                if uid in seen_uid:
                    continue
                seen_uid.add(uid)
                participants_ordered.append((uid, name))
            if sender_id and str(sender_id).strip() not in seen_uid:
                participants_ordered.append(
                    (str(sender_id).strip(), str(sender_name or "User"))
                )
            if participants_ordered:
                merged_text = "【人物印象】\n"
                used = 0
                for uid, nm in participants_ordered:
                    if used >= impression_recent_max_persons:
                        break
                    lines: list[str] = []
                    rel = _build_affinity_line(uid, nm)
                    if rel:
                        lines.append(rel)
                    lines.extend(_build_user_impression_lines(uid, nm))
                    if not lines:
                        continue
                    merged_text += f"{nm}/{uid}:\n" + "\n".join(lines) + "\n"
                    used += 1
                    recent_window_uids.add(uid)
                if used > 0:
                    _add_block(merged_text, 50)

        def _build_sender_impression_block() -> str:
            if not sender_id or not group_id:
                return ""
            alias_line = _build_alias_line(sender_id, sender_name)
            if impression_v3_enabled:
                try:
                    prof_v3 = self.db_mgr.get_impression_profile_v3(group_id, sender_id)
                except Exception:
                    prof_v3 = None
                if prof_v3:
                    v3_name, v3_summary, v3_tags, _v3_conf, _v3_ts = prof_v3
                    lines: list[str] = []
                    if alias_line:
                        lines.append(alias_line)
                    if v3_summary:
                        lines.append(f"- 近期印象：{v3_summary}")
                    if v3_tags:
                        lines.append("- 标签：" + "、".join(v3_tags[:8]))
                    try:
                        v3_facts = self.db_mgr.list_impression_facts_v3(
                            group_id, sender_id, limit=4
                        )
                    except Exception:
                        v3_facts = []
                    for fact, _reason, _conf, _ev, _ts in v3_facts or []:
                        if fact:
                            lines.append(f"- {fact}")
                    if lines:
                        return f"【人物印象】\n{v3_name}/{sender_id}:\n" + "\n".join(lines)
            prof = None
            try:
                prof = self.db_mgr.get_user_profile(group_id, sender_id)
            except Exception:
                prof = None
            if prof:
                name, pjson, canon = prof
                txt = _normalize_impression_text(canon or "") or _normalize_impression_text(
                    pjson or ""
                )
                if txt:
                    lines = []
                    if alias_line:
                        lines.append(alias_line)
                    lines.append(f"- {txt}")
                    return f"【人物印象】\n{name}/{sender_id}:\n" + "\n".join(lines)
            try:
                impression = self.db_mgr.get_impression(group_id, sender_id)
            except Exception:
                impression = None
            if impression:
                impression = _normalize_impression_text(impression)
                if not impression:
                    try:
                        impression = self.db_mgr.get_impression(group_id, sender_id)
                    except Exception:
                        impression = ""
                if impression:
                    lines = []
                    if alias_line:
                        lines.append(alias_line)
                    lines.append(str(impression))
                    return f"【人物印象】\n关于 {sender_name} 的印象：\n" + "\n".join(lines)
            return ""

        sender_impression_block = (
            _build_sender_impression_block()
            if (impression_inject_enabled and not impression_merge_recent_participants)
            else ""
        )
        if impression_sender_pinned and sender_impression_block and not impression_merge_recent_participants:
            _add_block(sender_impression_block, 90)

        # Only keep impressions for users who are actually present in this turn context.
        active_uids: set[str] = set()
        if impression_active_only:
            try:
                if sender_id:
                    active_uids.add(str(sender_id).strip())
                for uid_raw, _nm in recent_participants or []:
                    uid = str(uid_raw or "").strip()
                    if uid:
                        active_uids.add(uid)
                for src in (history_msgs or []) + [query]:
                    txt = str(src or "")
                    for m in re.finditer(r"/(\d{5,20})\s*:", txt):
                        active_uids.add(m.group(1))
                    for m in re.finditer(r"<at:(\d{5,20})>", txt):
                        active_uids.add(m.group(1))
                    for m in re.finditer(r"@(\d{5,20})", txt):
                        active_uids.add(m.group(1))
            except Exception:
                pass

        injected_related_impression = False
        semantic_extra_nonrecent_max = int(
            self._get_conf("impression_semantic_extra_nonrecent_max_persons", 3)
        )
        if semantic_extra_nonrecent_max < 1:
            semantic_extra_nonrecent_max = 1
        if semantic_extra_nonrecent_max > 4:
            semantic_extra_nonrecent_max = 4

        if impression_inject_enabled and impression_semantic_enabled and recalled_v3:
            grouped_v3: dict[str, dict] = {}
            for content, score, uid, scope in recalled_v3:
                if not uid:
                    continue
                if impression_merge_recent_participants:
                    # In merge mode, semantic block appends 1-3 related persons
                    # that are NOT present in the latest dialogue window.
                    if uid in recent_window_uids:
                        continue
                else:
                    if impression_active_only and active_uids and uid not in active_uids:
                        continue
                if impression_sender_pinned and sender_id and uid == str(sender_id):
                    continue
                txt = _compact_injected_text(content, 150)
                if not txt:
                    continue
                if uid not in grouped_v3:
                    name = "User"
                    try:
                        prof_v3 = self.db_mgr.get_impression_profile_v3(group_id, uid)
                        if prof_v3 and prof_v3[0]:
                            name = str(prof_v3[0])
                    except Exception:
                        pass
                    grouped_v3[uid] = {"name": name, "items": []}
                tag = "摘要" if str(scope or "").strip() == "profile" else "事实"
                grouped_v3[uid]["items"].append((tag, txt, float(score)))
            if grouped_v3:
                i_text = "【人物印象-相关人物】\n"
                cnt = 0
                used_budget = len(i_text)
                max_related_persons = (
                    min(max_persons, semantic_extra_nonrecent_max)
                    if impression_merge_recent_participants
                    else max_persons
                )
                for uid, data in grouped_v3.items():
                    if cnt >= max_related_persons:
                        break
                    items = data.get("items") or []
                    if not items:
                        continue
                    line_head = f"{data.get('name') or 'User'}/{uid}:\n"
                    if (
                        impression_token_budget > 0
                        and used_budget + len(line_head) > impression_token_budget
                    ):
                        break
                    i_text += line_head
                    used_budget += len(line_head)
                    added = 0
                    for tag, txt, _score in items:
                        if added >= max_per_person:
                            break
                        line = f"- [{tag}] {txt}\n"
                        if (
                            impression_token_budget > 0
                            and used_budget + len(line) > impression_token_budget
                        ):
                            break
                        i_text += line
                        used_budget += len(line)
                        added += 1
                    if added > 0:
                        cnt += 1
                if cnt > 0:
                    _add_block(i_text, 80)
                    injected_related_impression = True

        if (
            impression_inject_enabled
            and impression_semantic_enabled
            and recalled
            and not injected_related_impression
        ):
            grouped: dict[str, dict] = {}

            def _parse_impr2(text: str) -> tuple[str, str, str] | None:
                raw = str(text or "").strip()
                if not raw.startswith("IMPR2"):
                    return None
                head, sep, body = raw.partition("::")
                if not sep:
                    return None
                uid_m = re.search(r"\buid=([^\s]+)", head)
                if not uid_m:
                    return None
                uid = uid_m.group(1).strip()
                name = "User"
                nm = re.search(r"\bname=(.+)$", head.strip())
                if nm:
                    name = nm.group(1).strip() or name
                body_txt = _normalize_impression_text(body.strip())
                return (uid, name, body_txt)

            for content, score in recalled:
                parsed = _parse_impr2(content)
                if not parsed:
                    continue
                uid, name, body_txt = parsed
                if impression_merge_recent_participants:
                    if uid in recent_window_uids:
                        continue
                else:
                    if impression_active_only and active_uids and uid not in active_uids:
                        continue
                if impression_sender_pinned and sender_id and uid == str(sender_id):
                    continue
                if not body_txt:
                    continue
                if uid not in grouped:
                    grouped[uid] = {"name": name, "items": []}
                segs: list[str] = []
                try:
                    segs = [
                        s.strip()
                        for s in re.split(r"[。！？.!?]\s*", body_txt)
                        if s.strip()
                    ]
                except Exception:
                    segs = [body_txt]
                for seg in segs:
                    if len(grouped[uid]["items"]) >= max_per_person:
                        break
                    grouped[uid]["items"].append(seg)
            if grouped:
                i_text = "【人物印象-相关人物】\n"
                cnt = 0
                used_budget = len(i_text)
                max_related_persons = (
                    min(max_persons, semantic_extra_nonrecent_max)
                    if impression_merge_recent_participants
                    else max_persons
                )
                for uid, data in grouped.items():
                    if cnt >= max_related_persons:
                        break
                    name = data.get("name") or "User"
                    items = data.get("items") or []
                    if not items:
                        continue
                    line_head = f"{name}/{uid}:\n"
                    if impression_token_budget > 0 and used_budget + len(line_head) > impression_token_budget:
                        break
                    i_text += line_head
                    used_budget += len(line_head)
                    for it in items[:max_per_person]:
                        content = _compact_injected_text(it, 120)
                        if content:
                            line = f"- {content}\n"
                            if (
                                impression_token_budget > 0
                                and used_budget + len(line) > impression_token_budget
                            ):
                                break
                            i_text += line
                            used_budget += len(line)
                    cnt += 1
                if cnt > 0:
                    _add_block(i_text, 80)
                    injected_related_impression = True

        if (
            impression_inject_enabled
            and (not impression_sender_pinned)
            and (not injected_related_impression)
            and sender_impression_block
            and (not impression_merge_recent_participants)
        ):
            _add_block(sender_impression_block, 90)

        if sender_id and group_id and not impression_merge_recent_participants:
            self._maybe_decay_affinity(group_id)
            try:
                self.db_mgr.ensure_affinity_state(group_id, sender_id)
                # Per-user idle decay: if sender has been inactive for >idle_sec,
                # trigger immediate per-user decay before reading state.
                idle_sec = float(self._get_conf("affinity_decay_idle_sec", 3600))
                aff_state = self.db_mgr.get_affinity_state(group_id, sender_id)
                if aff_state and idle_sec > 0:
                    _last_evt = aff_state[3] or 0
                    if _last_evt > 0 and time.time() - float(_last_evt) > idle_sec:
                        self.db_mgr.decay_affinity_towards_neutral(
                            group_id,
                            step=float(self._get_conf("affinity_decay_step", 1.0)),
                            idle_sec=idle_sec,
                        )
                        aff_state = self.db_mgr.get_affinity_state(group_id, sender_id)
            except Exception:
                aff_state = None
            if aff_state:
                affinity, trust, mood_bias, _last_evt, _updated = aff_state
                relation = "中性"
                if affinity >= 35:
                    relation = "明显喜欢"
                elif affinity >= 12:
                    relation = "偏喜欢"
                elif affinity <= -35:
                    relation = "明显反感"
                elif affinity <= -12:
                    relation = "偏反感"
                _add_block(
                    "【关系倾向】\n"
                    f"- 对 {sender_name or '该用户'}/{sender_id}：{relation}\n"
                    "- 关系倾向用于语气强弱，不覆盖事实与规则。",
                    85,
                )

        # 5. Recent History
        if history_msgs:
            cleaned = []
            emoji_seen_cnt: dict[str, int] = {}
            emoji_dropped = False
            # ConversationV2 carries no self_id field, so the previous
            # getattr(conversation, "self_id") always yielded "" and the
            # rewrite below ran on every history line, including other
            # people's messages. Callers now pass the bot id explicitly and
            # the rewrite is skipped whenever it is unknown.
            self_id_hist = str(self_id or "")
            for ln in history_msgs:
                s = _compact_injected_text(ln, 260)
                if s:
                    # Retired lowercase tags like [baka] belong to an older voice;
                    # leaving them in history teaches the model to keep emitting them.
                    s = re.sub(r"\[[a-z_]+\]", "", s).strip()
                if s and self_id_hist and self_id_hist in s.split(":", 1)[0]:
                    # Her own old lines are the strongest style template; strip the
                    # "不过/但是" hinge so history stops modelling the tic we banned.
                    s = re.sub(
                        r"(?<=[。！？!?…\s—\-])(?:不过|但是|可是|只是|然而)[，,]?\s*",
                        "",
                        s,
                    )
                    s = re.sub(r"^(?:不过|但是|可是|只是|然而)[，,]?\s*", "", s)
                    # Drop the interjection she opens almost every line with, so
                    # history stops teaching "语气词 + 复述问题" as the template.
                    _m = re.match(
                        r"^(?P<head>[^:：]{0,40}[:：]\s*)?"
                        r"(?:诶+|咦+|呜哇+|哇|哈)[。、，,…～~\-—!！?？\s]*",
                        s,
                    )
                    if _m:
                        _rest = s[_m.end() :]
                        if len(_rest) >= 8:
                            s = (_m.group("head") or "") + _rest
                if s:
                    k = _emoji_desc_key(s)
                    if k:
                        c = int(emoji_seen_cnt.get(k, 0))
                        if c >= 2:
                            emoji_dropped = True
                            continue
                        emoji_seen_cnt[k] = c + 1
                    cleaned.append(s)
            if task_norm == "read_air":
                max_hist = int(self._get_conf("history_lines_limit_read_air", 12))
                if max_hist > 0 and len(cleaned) > max_hist:
                    cleaned = cleaned[-max_hist:]
            elif task_norm == "focus":
                max_hist = int(self._get_conf("history_lines_limit_focus", 20))
                if max_hist > 0 and len(cleaned) > max_hist:
                    cleaned = cleaned[-max_hist:]
            if emoji_dropped:
                cleaned.append("（同类表情包描述已去重，每类最多保留2条）")
            if cleaned:
                h_text = (
                    "【历史对话】\n"
                    "（只用来了解聊过什么。以前那些「哈？」「不过……」的腔调已经过时了，"
                    "别照着历史里的句式说话）\n" + "\n".join(cleaned)
                )
                _add_block(h_text, 100)

        # ── 全局 memory_context token budget ──────────────────────────────────
        # 高优先级块先占预算（历史 100 > 印象 90 > 记忆 70 > 知识 40），再按原顺序输出。
        try:
            global_budget = int(
                self._get_conf("memory_context_total_budget_chars", 12000)
            )
        except Exception:
            global_budget = 12000
        if global_budget > 0:
            remaining = global_budget
            keep: list[tuple[int, str]] = []
            for idx in sorted(
                range(len(context_parts)), key=lambda i: -context_parts[i][0]
            ):
                text = context_parts[idx][1]
                if len(text) <= remaining:
                    keep.append((idx, text))
                    remaining -= len(text)
                elif not keep and remaining > 0:
                    # Always keep at least the highest-priority block, truncated.
                    keep.append((idx, text[:remaining]))
                    remaining = 0
            selected = [text for _idx, text in sorted(keep, key=lambda x: x[0])]
        else:
            selected = [text for _prio, text in context_parts]

        return "\n\n".join(selected), admin_protocols_text


__all__ = ["Scheduler"]
