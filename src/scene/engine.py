from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from typing import Any

from ..core.models import DummyEvent

logger = logging.getLogger("astrbot")

_PHASES = {"start", "rising", "plateau", "finish", "aftercare", "self_play"}
_CONTINUOUS_INTENT_RE = re.compile(
    r"(?:连着|连续|持续|一直|不停|别停|不要停|不许停|刷屏|多连发|发好多|多发|多来)"
)
_DURATION_RE = re.compile(
    r"(半|[零一二两三四五六七八九十百]+|\d+(?:\.\d+)?)\s*(?:个)?\s*(小时|分钟|分|秒)"
)


class SceneEngine:
    """Persistent autonomous private-chat scene coordinator.

    The LLM is responsible for the next in-character beat and for proposing the
    next wake interval.  This class owns persistence, timing, interruption,
    bounds, and the final stop conditions.
    """

    def __init__(self, plugin, config: dict, db_mgr, llm_client, scheduler):
        self.plugin = plugin
        self.config = config or {}
        self.db_mgr = db_mgr
        self.llm_client = llm_client
        self.scheduler = scheduler
        self._sessions: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._generations: dict[str, int] = {}
        self._processing: set[str] = set()

    # ------------------------------------------------------------------
    # Configuration and small helpers
    # ------------------------------------------------------------------
    def _get_conf(self, name: str, default):
        return self.config.get(name, default)

    def _get_bool(self, name: str, default: bool = False) -> bool:
        value = self._get_conf(name, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "启用", "开启"}
        return bool(value)

    def _get_int(self, name: str, default: int) -> int:
        try:
            return int(self._get_conf(name, default))
        except Exception:
            return int(default)

    def _get_float(self, name: str, default: float) -> float:
        try:
            return float(self._get_conf(name, default))
        except Exception:
            return float(default)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        if high < low:
            high = low
        return max(low, min(high, value))

    def is_enabled(self) -> bool:
        # Adult mode is a content ceiling, not the whole engine switch.  This
        # keeps ordinary persistent role-play/companionship scenes available
        # when the user explicitly asks for them.
        return self._get_bool("scene_engine_enabled", True)

    @staticmethod
    def is_private_umo(umo: str) -> bool:
        parts = str(umo or "").split(":")
        if len(parts) < 2:
            return False
        kind = parts[1].strip().lower()
        return kind in {
            "friend",
            "private",
            "friendmessage",
            "friend_message",
            "private_message",
        }

    def _max_duration_sec(self) -> int:
        return max(60, self._get_int("scene_max_duration_sec", 480))

    def _max_messages(self) -> int:
        return max(1, self._get_int("scene_max_messages", 18))

    def _max_messages_for(self, session: dict[str, Any]) -> int:
        base = self._max_messages()
        if self._continuous_active(session):
            return max(base, self._get_int("scene_continuous_max_messages", 160))
        return base

    def _max_ticks(self) -> int:
        return max(1, self._get_int("scene_max_ticks", 18))

    def _min_interval_sec(self) -> float:
        return max(4.0, self._get_float("scene_min_interval_sec", 8.0))

    def _max_interval_sec(self) -> float:
        return max(
            self._min_interval_sec(),
            self._get_float("scene_max_interval_sec", 45.0),
        )

    def _after_reply_delay_sec(self) -> float:
        return max(
            2.0,
            self._get_float("scene_after_reply_delay_sec", 8.0),
        )

    def _continuous_interval_sec(self) -> float:
        """Short cadence for a user-requested continuous private-chat burst."""
        low = self._min_interval_sec()
        high = min(self._max_interval_sec(), max(low, 15.0))
        return self._clamp(
            self._get_float("scene_continuous_interval_sec", 8.0),
            low,
            high,
        )

    def _idle_soft_limit_sec(self) -> float:
        return max(20.0, self._get_float("scene_idle_soft_limit_sec", 90.0))

    def _stop_keywords(self) -> list[str]:
        raw = self._get_conf(
            "scene_stop_keywords",
            [
                "停下",
                "停止",
                "结束",
                "不要了",
                "不继续",
                "退出场景",
                "取消场景",
                "算了",
            ],
        )
        if isinstance(raw, str):
            raw = [x.strip() for x in re.split(r"[,，\n]+", raw) if x.strip()]
        if not isinstance(raw, list):
            return []
        return [str(x).strip().lower() for x in raw if str(x).strip()]

    @staticmethod
    def _text_key(text: str) -> str:
        return re.sub(r"\s+", "", str(text or "").lower())

    def is_stop_text(self, text: str) -> bool:
        key = self._text_key(text)
        if not key:
            return False
        # 别停, 不要停, 继续 are requests to continue, not stop.
        if re.search(r"(别|不要|不许|不准|继续)(停|结束|退出|取消)", key):
            return False
        if "继续" in key and "结束" not in key and "退出" not in key:
            return False
        return any(k and k in key for k in self._stop_keywords())

    @staticmethod
    def _number_from_token(token: str) -> float:
        token = str(token or "").strip()
        if not token:
            return 0.0
        if token == "半":
            return 0.5
        try:
            return float(token)
        except ValueError:
            pass
        digits = {
            "零": 0, "一": 1, "二": 2, "两": 2, "三": 3,
            "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
        }
        total = 0
        current = 0
        for ch in token:
            if ch in digits:
                current = digits[ch]
            elif ch == "十":
                total += (current or 1) * 10
                current = 0
            elif ch == "百":
                total += (current or 1) * 100
                current = 0
            else:
                return 0.0
        return float(total + current)

    @classmethod
    def _parse_duration_sec(cls, text: str) -> int:
        key = cls._text_key(text)
        for match in _DURATION_RE.finditer(key):
            value = cls._number_from_token(match.group(1))
            unit = match.group(2)
            factor = 3600 if unit == "小时" else 60 if unit in {"分钟", "分"} else 1
            seconds = int(round(value * factor))
            if seconds > 0:
                return seconds
        return 0

    def _continuous_request(self, text: str) -> int | None:
        """Return an explicit continuous-send window in seconds, if requested.

        Keep this narrow.  Ordinary role play must retain the slower,
        model-chosen cadence; only explicit burst wording gets the fast lane.
        """
        key = self._text_key(text)
        if not key:
            return None
        intent = bool(_CONTINUOUS_INTENT_RE.search(key))
        duration = self._parse_duration_sec(key)
        if not intent:
            if not duration:
                return None
            if not re.search(r"(?:发|说|聊|陪).{0,10}(?:小时|分钟|分|秒)", key):
                return None
        if duration <= 0:
            if re.search(r"(?:一直发|不要停|别停|不许停|不停|刷屏)", key):
                duration = self._max_duration_sec()
            else:
                duration = self._get_int("scene_continuous_default_sec", 300)
        return max(10, min(self._max_duration_sec(), int(duration)))

    @staticmethod
    def _continuous_active(session: dict[str, Any], now: float | None = None) -> bool:
        now = time.time() if now is None else float(now)
        try:
            until = float(session.get("continuous_until", 0) or 0)
        except Exception:
            until = 0.0
        return bool(session.get("continuous")) and now < until

    def _arm_continuous(
        self,
        session: dict[str, Any],
        text: str,
        now: float,
        *,
        reset_deadline: bool = False,
    ) -> bool:
        duration = self._continuous_request(text)
        if not duration:
            return False
        started_at = float(session.get("started_at", now) or now)
        hard_deadline = started_at + self._max_duration_sec()
        until = min(float(now) + duration, hard_deadline)
        session["continuous"] = True
        session["continuous_until"] = until
        session["phase"] = "self_play"
        current_deadline = float(session.get("deadline_at", 0) or 0)
        session["deadline_at"] = until if reset_deadline else max(current_deadline, until)
        return True

    # ------------------------------------------------------------------
    # DB-backed session helpers
    # ------------------------------------------------------------------
    async def _load_active_session(self, umo: str) -> dict[str, Any] | None:
        if not umo or not self.is_enabled() or not self.is_private_umo(umo):
            return None
        cached = self._sessions.get(umo)
        if isinstance(cached, dict) and cached.get("status") == "active":
            return cached
        try:
            session = await self.db_mgr.get_scene_session(umo)
        except Exception:
            session = None
        if not isinstance(session, dict) or session.get("status") != "active":
            return None
        self._sessions[umo] = session
        return session

    async def _persist(self, session: dict[str, Any]):
        session["updated_at"] = time.time()
        self._sessions[str(session.get("umo") or "")] = session
        try:
            await self.db_mgr.save_scene_session(session)
        except Exception as e:
            logger.warning(f"[mem_chat] scene session save failed: {e}")

    def _cancel_timer(self, umo: str):
        self._generations[umo] = self._generations.get(umo, 0) + 1
        task = self._tasks.pop(umo, None)
        # An autonomous tick schedules its own successor.  Cancelling the
        # current task here would abort that tick at the next await and leave
        # the scene alive in memory but without any timer.
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _schedule_tick(self, umo: str, delay_sec: float):
        if not umo:
            return
        self._cancel_timer(umo)
        delay = self._clamp(
            float(delay_sec or self._min_interval_sec()),
            self._min_interval_sec(),
            self._max_interval_sec(),
        )
        session = self._sessions.get(umo)
        if isinstance(session, dict):
            session["next_tick_at"] = time.time() + delay
            await self._persist(session)
        generation = self._generations.get(umo, 0)
        task = asyncio.create_task(self._wait_then_tick(umo, generation, delay))
        self._tasks[umo] = task

    async def _wait_then_tick(self, umo: str, generation: int, delay: float):
        try:
            await asyncio.sleep(max(0.1, float(delay)))
            await self._run_tick(umo, generation)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"[mem_chat] scene tick crashed for {umo}: {e}")
        finally:
            if self._tasks.get(umo) is asyncio.current_task():
                self._tasks.pop(umo, None)

    # ------------------------------------------------------------------
    # Scene payload handling
    # ------------------------------------------------------------------
    @staticmethod
    def _get_scene_payload(event) -> dict[str, Any] | None:
        if event is None:
            return None
        direct = getattr(event, "_mem_chat_scene_payload", None)
        if isinstance(direct, dict):
            return direct
        payload = getattr(event, "_mem_chat_last_structured_payload", None)
        if isinstance(payload, dict) and isinstance(payload.get("scene"), dict):
            return payload.get("scene")
        return None

    def normalize_scene_payload(self, payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        active = payload.get("active")
        start = payload.get("start")
        end = payload.get("end")
        if active is None and start is None and end is None:
            return None
        phase = str(payload.get("phase") or "start").strip().lower()
        if phase not in _PHASES:
            phase = "start"
        try:
            intensity = float(payload.get("intensity", 0))
        except Exception:
            intensity = 0.0
        intensity = self._clamp(intensity, 0.0, 10.0)
        try:
            next_tick = float(payload.get("next_tick_sec", 0) or 0)
        except Exception:
            next_tick = 0.0
        return {
            "active": bool(active) if active is not None else not bool(end),
            "start": bool(start),
            "end": bool(end) or active is False or payload.get("continue") is False,
            "phase": phase,
            "intensity": intensity,
            "next_tick_sec": next_tick,
        }

    async def start_from_scene_payload(
        self,
        umo: str,
        event,
        scene_payload: dict[str, Any],
        *,
        user_text: str = "",
        initial_sent_count: int | None = None,
    ) -> dict[str, Any] | None:
        if not self.is_enabled() or not self.is_private_umo(umo):
            return None
        scene = self.normalize_scene_payload(scene_payload)
        if not scene or not scene.get("start") or not scene.get("active"):
            return None
        now = time.time()
        if initial_sent_count is None:
            initial_sent_count = (
                self._sent_message_count(event)
                if bool(getattr(event, "_mem_chat_last_send_ok", False))
                else 0
            )
        initial_sent_count = max(0, int(initial_sent_count or 0))
        if self._continuous_request(user_text):
            scene = dict(scene)
            scene["phase"] = "self_play"
        scope_id = ""
        try:
            scope_id = str(event.get_sender_id() or "").strip() if event else ""
        except Exception:
            scope_id = ""
        if not scope_id:
            scope_id = str(umo).split(":")[-1]
        session = {
            "umo": umo,
            "status": "active",
            "phase": scene["phase"],
            "intensity": scene["intensity"],
            "started_at": now,
            "updated_at": now,
            "deadline_at": now + self._max_duration_sec(),
            "next_tick_at": 0.0,
            "tick_count": 0,
            "sent_count": initial_sent_count,
            "last_user_at": now,
            "last_bot_at": now if initial_sent_count else 0.0,
            "consent_at": now,
            "scope_id": scope_id,
            "last_user_text": str(user_text or "")[-2000:],
            "last_seen_message_id": str(
                getattr(getattr(event, "message_obj", None), "message_id", "") or ""
            ),
            "last_scene_payload": scene,
            "last_error": "",
        }
        self._arm_continuous(session, user_text, now, reset_deadline=True)
        self._sessions[umo] = session
        await self._persist(session)
        delay = self._next_delay(session, scene)
        await self._schedule_tick(umo, delay)
        logger.info(
            f"[mem_chat] scene started for {umo}: phase={session['phase']} "
            f"sent={session['sent_count']} continuous={self._continuous_active(session)} "
            f"next={delay:.1f}s"
        )
        return session

    async def close_session(self, umo: str, status: str, reason: str = ""):
        if not umo:
            return
        self._cancel_timer(umo)
        session = self._sessions.pop(umo, None)
        if not isinstance(session, dict):
            try:
                session = await self.db_mgr.get_scene_session(umo)
            except Exception:
                session = None
        if not isinstance(session, dict):
            return
        session["status"] = str(status or "done")
        session["close_reason"] = str(reason or "")
        session["updated_at"] = time.time()
        try:
            await self.db_mgr.save_scene_session(session)
        except Exception as e:
            logger.warning(f"[mem_chat] scene session close save failed: {e}")
        logger.info(
            f"[mem_chat] scene closed for {umo}: status={status} reason={reason}"
        )

    async def restore_sessions(self):
        if not self.is_enabled():
            return
        try:
            rows = await self.db_mgr.list_scene_sessions(status="active")
        except Exception as e:
            logger.warning(f"[mem_chat] scene restore failed: {e}")
            return
        now = time.time()
        for session in rows:
            if not isinstance(session, dict):
                continue
            umo = str(session.get("umo") or "")
            if not umo or not self.is_private_umo(umo):
                continue
            if self._is_expired(session, now):
                await self.close_session(umo, "expired", "restore_expired")
                continue
            self._sessions[umo] = session
            delay = max(2.0, float(session.get("next_tick_at", 0) or 0) - now)
            if delay > self._max_interval_sec():
                delay = self._min_interval_sec()
            await self._schedule_tick(umo, delay)
            logger.info(f"[mem_chat] scene restored for {umo}, next={delay:.1f}s")

    # ------------------------------------------------------------------
    # Incoming private messages
    # ------------------------------------------------------------------
    async def note_private_message(self, umo: str, text: str, event) -> str:
        if not self.is_enabled() or not self.is_private_umo(umo):
            return "disabled"
        session = await self._load_active_session(umo)
        if not session:
            return "none"

        self._cancel_timer(umo)
        if umo in self._processing:
            try:
                self.plugin.state.set_interrupt(umo)
            except Exception:
                pass

        if self.is_stop_text(text):
            try:
                setattr(event, "_mem_chat_scene_suppress_start", True)
            except Exception:
                pass
            await self.close_session(umo, "cancelled", "user_stop")
            return "stopped"

        now = time.time()
        session["last_user_at"] = now
        session["last_user_text"] = str(text or "")[-2000:]
        if self._arm_continuous(session, text, now):
            logger.info(
                f"[mem_chat] continuous scene armed for {umo}: "
                f"until={float(session.get('continuous_until', 0)):.0f}"
            )
        try:
            session["last_seen_message_id"] = str(
                getattr(getattr(event, "message_obj", None), "message_id", "") or ""
            )
        except Exception:
            pass
        session["status"] = "active"
        session["last_scene_payload"] = session.get("last_scene_payload") or {}
        await self._persist(session)
        try:
            setattr(event, "_mem_chat_scene_instruction", self._build_user_turn_instruction(session))
        except Exception:
            pass
        return "active"

    async def on_scheduler_reply_completed(
        self,
        umo: str,
        event,
        task_type: str,
        sent: bool = False,
        replied: bool = False,
    ):
        if not self.is_enabled() or not self.is_private_umo(umo):
            return
        if event is None or bool(getattr(event, "_mem_chat_scene_suppress_start", False)):
            return

        session = await self._load_active_session(umo)
        scene = self.normalize_scene_payload(self._get_scene_payload(event))
        user_text = str(
            getattr(event, "message_str", "")
            or self._event_text(event)
            or ""
        )
        initial_sent_count = self._sent_message_count(event) if sent else 0

        if not session:
            if not scene and self._continuous_request(user_text):
                scene = self.normalize_scene_payload(
                    {
                        "active": True,
                        "start": True,
                        "phase": "self_play",
                        "intensity": 5,
                        "end": False,
                        "next_tick_sec": self._continuous_interval_sec(),
                    }
                )
            if scene and scene.get("start") and scene.get("active") and not scene.get("end"):
                await self.start_from_scene_payload(
                    umo,
                    event,
                    scene,
                    user_text=user_text,
                    initial_sent_count=initial_sent_count,
                )
            return

        if sent:
            session["sent_count"] = int(
                session.get("sent_count", 0) or 0
            ) + initial_sent_count
            session["last_bot_at"] = time.time()
        if user_text:
            session["last_user_text"] = user_text[-2000:]
        new_continuous_request = self._arm_continuous(
            session, user_text, time.time()
        )

        if scene:
            session["intensity"] = scene["intensity"]
            if self._continuous_active(session) and new_continuous_request:
                # A fresh explicit request such as "连着发五分钟" owns the
                # window.  On a later ordinary user turn the model may still
                # decide that the human wants to stop; only autonomous ticks
                # are prevented from ending the requested burst early.
                if scene.get("end") or not scene.get("active"):
                    logger.info(
                        f"[mem_chat] ignored early scene end during continuous window for {umo}"
                    )
                scene = dict(scene)
                scene["active"] = True
                scene["start"] = False
                scene["end"] = False
                scene["phase"] = "self_play"
                session["phase"] = "self_play"
            else:
                session["phase"] = scene["phase"]
            session["last_scene_payload"] = scene
            if scene.get("end") or not scene.get("active"):
                await self.close_session(umo, "done", "scene_end")
                return

        # A normal reply has just gone out.  Give the user a little room before
        # the next autonomous beat.
        await self._persist(session)
        await self._schedule_tick(umo, self._next_delay(session, scene, after_reply=True))

    @staticmethod
    def _sent_message_count(event) -> int:
        """Best-effort count of the message segments actually handed to send()."""
        try:
            payload = getattr(event, "_mem_chat_last_structured_payload", None)
            if isinstance(payload, dict):
                response = payload.get("response")
                if isinstance(response, dict):
                    segments = response.get("segments")
                    if isinstance(segments, list):
                        count = sum(1 for seg in segments if isinstance(seg, dict))
                        if count:
                            return count
        except Exception:
            pass
        return 1

    @staticmethod
    def _event_text(event) -> str:
        try:
            text = str(getattr(event, "message_str", "") or "").strip()
            if text:
                return text
        except Exception:
            pass
        try:
            return str(getattr(event, "message_str", "") or "").strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Autonomous ticks
    # ------------------------------------------------------------------
    def _is_expired(self, session: dict[str, Any], now: float | None = None) -> bool:
        now = time.time() if now is None else float(now)
        if now >= float(session.get("deadline_at", 0) or 0):
            return True
        if int(session.get("sent_count", 0) or 0) >= self._max_messages_for(session):
            return True
        if int(session.get("tick_count", 0) or 0) >= self._max_ticks():
            return True
        return False

    def _next_delay(
        self,
        session: dict[str, Any],
        scene: dict[str, Any] | None,
        *,
        after_reply: bool = False,
    ) -> float:
        if self._continuous_active(session):
            base = (
                self._after_reply_delay_sec()
                if after_reply
                else self._continuous_interval_sec()
            )
        elif after_reply:
            base = self._after_reply_delay_sec()
        elif scene and float(scene.get("next_tick_sec", 0) or 0) > 0:
            base = float(scene["next_tick_sec"])
        else:
            phase = str(session.get("phase") or "start")
            if phase in {"finish", "aftercare"}:
                base = self._max_interval_sec() * 0.8
            elif phase == "plateau":
                base = self._min_interval_sec() * 1.6
            else:
                base = self._min_interval_sec() * 1.2
        base = float(base) * random.uniform(0.92, 1.08)
        return self._clamp(base, self._min_interval_sec(), self._max_interval_sec())

    async def _run_tick(self, umo: str, generation: int):
        if generation != self._generations.get(umo, 0):
            return
        session = await self._load_active_session(umo)
        if not session:
            return
        now = time.time()
        if self._is_expired(session, now):
            await self.close_session(umo, "expired", "limits_reached")
            return

        send_event = DummyEvent(umo, self.plugin.context)
        try:
            send_event._mem_chat_scene_tick = True
        except Exception:
            pass

        self._processing.add(umo)
        try:
            self.plugin.state.set_processing(umo, True)
        except Exception:
            pass

        try:
            query = str(session.get("last_user_text") or "").strip() or "持续私聊场景"
            scope_id = str(session.get("scope_id") or "").strip()
            try:
                memory_context, admin_protocols_text = await self.scheduler.build_memory_context(
                    query=query,
                    group_id=scope_id or umo.split(":")[-1],
                    sender_name="User",
                    sender_id=scope_id or None,
                    is_group=False,
                    task_type="focus",
                    self_id=self.plugin.state.get_self_id(umo),
                )
            except Exception:
                memory_context, admin_protocols_text = "", ""

            ok = await self.llm_client.invoke_llm(
                umo=umo,
                prompt=self._build_autonomous_prompt(session),
                memory_context=memory_context,
                event=send_event,
                task_type="focus",
                instruction_text=self._build_autonomous_instruction(session),
                admin_protocols_text=admin_protocols_text,
                image_urls=None,
                created_at=0,
                impression_mode="off",
            )
            scene = self.normalize_scene_payload(self._get_scene_payload(send_event))
            if self._continuous_active(session):
                if not scene:
                    scene = self.normalize_scene_payload(
                        {
                            "active": True,
                            "start": False,
                            "phase": "self_play",
                            "intensity": session.get("intensity", 5),
                            "end": False,
                            "next_tick_sec": self._continuous_interval_sec(),
                        }
                    )
                if scene:
                    if scene.get("end") or not scene.get("active"):
                        logger.info(
                            f"[mem_chat] ignored early scene end during continuous tick for {umo}"
                        )
                    scene = dict(scene)
                    scene["active"] = True
                    scene["start"] = False
                    scene["end"] = False
                    scene["phase"] = "self_play"
            sent = bool(getattr(send_event, "_mem_chat_last_send_ok", False))
            session["tick_count"] = int(session.get("tick_count", 0) or 0) + 1
            if sent:
                session["sent_count"] = int(
                    session.get("sent_count", 0) or 0
                ) + self._sent_message_count(send_event)
                session["last_bot_at"] = time.time()
            if scene:
                session["phase"] = scene["phase"]
                session["intensity"] = scene["intensity"]
                session["last_scene_payload"] = scene

            if scene and (scene.get("end") or not scene.get("active")):
                await self.close_session(umo, "done", "scene_end")
                return
            if self._is_expired(session, time.time()):
                await self.close_session(umo, "expired", "limits_reached")
                return
            if not ok:
                session["last_error"] = "llm_failed"
            else:
                session["last_error"] = ""

            await self._persist(session)
            await self._schedule_tick(umo, self._next_delay(session, scene))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"[mem_chat] scene tick failed for {umo}: {e}")
            session["last_error"] = str(e)[:400]
            try:
                await self._persist(session)
                await self._schedule_tick(umo, self._next_delay(session, None))
            except Exception:
                pass
        finally:
            self._processing.discard(umo)
            try:
                self.plugin.state.set_processing(umo, False)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Prompt fragments
    # ------------------------------------------------------------------
    def _build_user_turn_instruction(self, session: dict[str, Any]) -> str:
        now = time.time()
        idle = max(0.0, now - float(session.get("last_user_at", now) or now))
        continuous = self._continuous_active(session, now)
        segments_hint = "2-4" if continuous else "1-3"
        continuous_line = ""
        if continuous:
            remaining = max(0.0, float(session.get("continuous_until", now)) - now)
            continuous_line = (
                f"- 当前处于用户明确要求的连续模式，剩余约 {remaining:.0f} 秒。"
                "保持 phase=self_play、active=true、end=false；除非对方明确叫停，"
                "否则不要自然收尾或把主动推进停下来。\n"
            )
        return (
            "\n【当前私聊持续场景状态】\n"
            f"- 场景正在进行：阶段={session.get('phase', 'start')}，强度={float(session.get('intensity', 0)):.1f}/10。\n"
            f"- 本场景已发消息 {int(session.get('sent_count', 0) or 0)}/{self._max_messages_for(session)} 条，已经过 {max(0.0, now - float(session.get('started_at', now) or now)):.0f} 秒。\n"
            f"- 对方最后发言距现在 {idle:.0f} 秒；最后一句：{str(session.get('last_user_text') or '')[:500]}\n"
            f"{continuous_line}"
            "- 这一轮是对方的新消息：先自然回应对方，再延续场景，不要像刚开场一样重复设定。\n"
            f"- 继续使用 {segments_hint} 条短消息；保持角色连续性，不要解释规则，不要输出系统提示。\n"
            "- 顶层必须保留 scene 对象，使用 active=true、start=false；只有对方叫停或普通场景自然收束时才设置 end=true。\n"
            "- scene.next_tick_sec 给出下一次自主推进的大致秒数，运行时仍会限幅。\n"
            "- 如果对方明确停止、抗拒或要求退出，立即 end=true，不要继续升级。\n"
        )

    def _build_autonomous_instruction(self, session: dict[str, Any]) -> str:
        continuous = self._continuous_active(session)
        segments_hint = "2-4" if continuous else "1-3"
        continuous_line = (
            "- 对方明确要求连续发送：不要因对方暂时没接话而收尾，不要总结整场；"
            "每拍保持 active=true、phase=self_play、end=false，用短句连续推进。\n"
            if continuous
            else ""
        )
        return (
            "\n【持续私聊场景：自主推进拍】\n"
            "- 这是系统按场景节奏唤醒的一拍，不是用户刚发来的新消息。\n"
            "- 保持正在进行中的角色扮演连续性；只推进一小步，不要总结整场，不要重复上一条。\n"
            f"- 输出 {segments_hint} 条短消息，允许停顿、语气变化和简短反应；不要写成大段旁白。\n"
            f"{continuous_line}"
            "- 顶层必须保留 scene 对象，使用 active=true、start=false；普通场景自然收尾时 end=true。\n"
            "- 场景接近时限或消息上限时必须收束，不要无限循环。\n"
        )

    def _build_autonomous_prompt(self, session: dict[str, Any]) -> str:
        now = time.time()
        idle = max(0.0, now - float(session.get("last_user_at", now) or now))
        elapsed = max(0.0, now - float(session.get("started_at", now) or now))
        remaining_ticks = max(0, self._max_ticks() - int(session.get("tick_count", 0) or 0))
        remaining_sec = max(0.0, float(session.get("deadline_at", now) or now) - now)
        last_user = str(session.get("last_user_text") or "").strip()
        phase = str(session.get("phase") or "start")
        continuous = self._continuous_active(session, now)
        requested_autonomy = continuous or phase == "self_play" or any(
            key in last_user
            for key in (
                "不回复",
                "不用回",
                "别回复",
                "别理我",
                "一直发",
                "不要停",
                "别停",
            )
        )
        if requested_autonomy:
            idle_note = (
                "对方已明确要求自主连续推进或不回复；静默不是停止信号。"
                "按角色节奏继续一小拍，不要仅因对方暂时没说话就机械收束。"
            )
        elif idle >= self._idle_soft_limit_sec():
            idle_note = (
                f"对方最近没有继续说话，已静默约 {idle:.0f} 秒。"
                "不要假定对方仍在强烈同意；放慢节奏，必要时用角色口吻确认或自然收束。"
            )
        else:
            idle_note = f"对方最后发言距现在约 {idle:.0f} 秒，可以自然承接。"
        near_end = remaining_ticks <= 2 or remaining_sec <= 60
        if continuous:
            remaining_window = max(
                0.0, float(session.get("continuous_until", now) or now) - now
            )
            end_note = (
                f"用户明确要求的连续时段还剩约 {remaining_window:.0f} 秒；"
                "除非对方刚刚明确叫停，否则这一拍不要收尾，保持 active=true、"
                "phase=self_play、end=false，继续自然推进。"
            )
        else:
            end_note = (
                "本场景已接近时长/拍数上限：这一拍必须自然收束，scene.end=true。"
                if near_end
                else "若气氛自然，可以继续推进；不要强行制造固定高潮或机械重复。"
            )
        segments_hint = "2-4" if continuous else "1-3"
        scene_phase = "self_play" if continuous else "rising"
        return (
            "【持续私聊场景 · 自主推进】\n"
            f"当前阶段：{session.get('phase', 'start')}\n"
            f"强度：{float(session.get('intensity', 0)):.1f}/10\n"
            f"已持续：{elapsed:.0f} 秒；已发送：{int(session.get('sent_count', 0) or 0)}/{self._max_messages_for(session)} 条；剩余拍数：{remaining_ticks}\n"
            f"对方最后一句：{last_user[:800] or '（暂无新内容）'}\n"
            f"{idle_note}\n"
            f"{end_note}\n\n"
            "本轮只输出角色接下来的一小拍，JSON 结构必须包含 control、response、scene。\n"
            f"response 使用 segments 输出 {segments_hint} 条短消息；scene 字段示例：\n"
            f'{{"active": true, "start": false, "phase": "{scene_phase}", "intensity": 5, "end": false, "next_tick_sec": {int(self._continuous_interval_sec()) if continuous else 20}}}\n'
            "scene.next_tick_sec 会被限制在配置范围内；这不是用户可见文本。"
        )

    def shutdown(self):
        for umo in list(self._tasks.keys()):
            self._cancel_timer(umo)
