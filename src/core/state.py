import hashlib
import random
import time
from collections.abc import Callable
from copy import deepcopy
from enum import Enum


class ChatState(Enum):
    READ_AIR = "read_air"
    FOCUS = "focus"


class MemChatState:
    def __init__(
        self,
        *,
        now: Callable[[], float] | None = None,
        rand: Callable[[], float] | None = None,
    ):
        self._now = now or time.time
        self._rand = rand or random.random
        self.sessions: set[str] = set()
        self.focus_until: dict[str, float] = {}
        self.focus_start: dict[str, float] = {}
        self.last_msg: dict[str, tuple[str, str, str, str]] = {}
        self.last_read_id: dict[str, str] = {}
        self.pending_msgs: dict[str, list[tuple[str, str, str, str, float]]] = {}
        self.last_processed: dict[str, float] = {}
        self.last_cycle_cutoff: dict[str, float] = {}
        self.bot_self_ids: dict[str, str] = {}
        self.bot_self_names: dict[str, str] = {}
        self.interrupt_flags: dict[str, bool] = {}
        self.processing_flags: dict[str, bool] = {}
        self.summary_msg_counts: dict[str, int] = {}
        self.read_air_msg_counts: dict[str, int] = {}
        self.last_persona_hash: str = ""
        self.last_knowledge_hash: str = ""
        self.cycle_tasks: dict[str, str] = {}
        self.current_states: dict[str, ChatState] = {}
        self.heartbeat_start: dict[str, float] = {}
        self.recent_msg_keys: dict[str, dict[str, float]] = {}
        self.focus_delays: dict[str, float] = {}
        self.last_sent_image_ids: dict[str, list[str]] = {}
        self.reply_targets: dict[str, dict[str, str]] = {}
        self.reply_targets_seen_ts: dict[str, dict[str, float]] = {}
        self.reply_summaries: dict[str, dict[str, tuple[str, str, str, float]]] = {}
        self.message_metas: dict[str, dict[str, dict]] = {}
        self.message_meta_seen_ts: dict[str, dict[str, float]] = {}
        self.last_moods: dict[str, str] = {}
        self.last_mood_states: dict[str, str] = {}
        self.mood_state_expires_at: dict[str, float] = {}
        self.last_mood_intensity: dict[str, float] = {}
        self.last_affinity_deltas: dict[str, dict[str, tuple[float, str, float]]] = {}
        self.poke_preapply_msg_ids: dict[str, dict[str, float]] = {}
        self.no_reply_msg_ids: dict[str, dict[str, float]] = {}

    def get_last_mood(self, umo: str) -> str:
        if not umo:
            return ""
        v = self.last_moods.get(umo)
        return str(v or "").strip()

    def set_last_mood(self, umo: str, mood_text: str | None):
        if not umo:
            return
        s = str(mood_text or "").strip()
        if not s:
            return
        self.last_moods[umo] = s[:60]

    def get_last_mood_state(self, umo: str) -> str:
        if not umo:
            return ""
        exp = float(self.mood_state_expires_at.get(umo, 0.0) or 0.0)
        if exp > 0 and self._now() >= exp:
            self.last_mood_states[umo] = "中性"
            self.mood_state_expires_at.pop(umo, None)
        v = self.last_mood_states.get(umo)
        return str(v or "").strip()

    def set_last_mood_state(self, umo: str, mood_state: str | None):
        if not umo:
            return
        s = str(mood_state or "").strip()
        if not s:
            return
        self.last_mood_states[umo] = s[:40]
        if s[:40] == "中性":
            self.mood_state_expires_at.pop(umo, None)

    def clear_last_mood_state(self, umo: str):
        if not umo:
            return
        self.last_mood_states.pop(umo, None)
        self.mood_state_expires_at.pop(umo, None)

    def set_mood_state_ttl(self, umo: str, ttl_sec: int | float | None):
        if not umo:
            return
        try:
            ttl = float(ttl_sec or 0.0)
        except Exception:
            ttl = 0.0
        if ttl <= 0:
            self.mood_state_expires_at.pop(umo, None)
            return
        self.mood_state_expires_at[umo] = self._now() + ttl

    def get_mood_state_ttl_remaining(self, umo: str) -> float:
        if not umo:
            return 0.0
        exp = float(self.mood_state_expires_at.get(umo, 0.0) or 0.0)
        if exp <= 0:
            return 0.0
        remain = exp - self._now()
        if remain <= 0:
            self.mood_state_expires_at.pop(umo, None)
            self.last_mood_states[umo] = "中性"
            return 0.0
        return remain

    def set_last_mood_intensity(self, umo: str, intensity: float | int | None):
        if not umo:
            return
        try:
            v = float(intensity if intensity is not None else 0.0)
        except Exception:
            v = 0.0
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        self.last_mood_intensity[umo] = v

    def get_last_mood_intensity(self, umo: str) -> float:
        if not umo:
            return 0.0
        try:
            return float(self.last_mood_intensity.get(umo, 0.0) or 0.0)
        except Exception:
            return 0.0

    def clear_last_mood(self, umo: str):
        if not umo:
            return
        self.last_moods.pop(umo, None)
        self.last_mood_intensity.pop(umo, None)

    def set_last_affinity_delta(
        self, umo: str, user_id: str, delta: float | int | None, reason: str | None = ""
    ):
        if not umo:
            return
        uid = str(user_id or "").strip()
        if not uid:
            return
        try:
            d = float(delta if delta is not None else 0.0)
        except Exception:
            d = 0.0
        if d > 40.0:
            d = 40.0
        elif d < -40.0:
            d = -40.0
        bucket = self.last_affinity_deltas.get(umo)
        if bucket is None:
            bucket = {}
            self.last_affinity_deltas[umo] = bucket
        bucket[uid] = (d, str(reason or "")[:160], self._now())

    def pop_last_affinity_delta(
        self, umo: str, user_id: str, max_age_sec: float = 600.0
    ) -> tuple[float, str] | None:
        if not umo:
            return None
        uid = str(user_id or "").strip()
        if not uid:
            return None
        bucket = self.last_affinity_deltas.get(umo)
        if not bucket:
            return None
        rec = bucket.pop(uid, None)
        if rec is None:
            return None
        try:
            delta, reason, ts = rec
            age = self._now() - float(ts or 0.0)
        except Exception:
            return None
        if max_age_sec > 0 and age > float(max_age_sec):
            return None
        return float(delta), str(reason or "")

    def mark_poke_preapplied_msg(
        self,
        umo: str,
        msg_id: str,
        *,
        ttl_sec: float = 7200,
        max_size: int = 2048,
    ):
        if not umo:
            return
        mid = str(msg_id or "").strip()
        if not mid:
            return
        now = self._now()
        bucket = self.poke_preapply_msg_ids.get(umo)
        if bucket is None:
            bucket = {}
            self.poke_preapply_msg_ids[umo] = bucket
        if len(bucket) >= max_size:
            expired = [k for k, ts in bucket.items() if now - ts > ttl_sec]
            for k in expired:
                bucket.pop(k, None)
            if len(bucket) >= max_size:
                oldest = sorted(bucket.items(), key=lambda kv: kv[1])[: max(1, max_size // 8)]
                for k, _ in oldest:
                    bucket.pop(k, None)
        bucket[mid] = now

    def is_poke_preapplied_msg(
        self,
        umo: str,
        msg_id: str,
        *,
        ttl_sec: float = 7200,
    ) -> bool:
        if not umo:
            return False
        mid = str(msg_id or "").strip()
        if not mid:
            return False
        bucket = self.poke_preapply_msg_ids.get(umo)
        if not bucket:
            return False
        now = self._now()
        ts = bucket.get(mid)
        if ts is None:
            return False
        if ttl_sec > 0 and now - ts > ttl_sec:
            bucket.pop(mid, None)
            return False
        return True

    def is_duplicate_incoming(
        self,
        umo: str,
        msg_id: str | None,
        msg_text: str | None,
        sender_id: str | None,
        *,
        id_ttl_sec: int = 300,
        fp_ttl_sec: int = 12,
        fp_ttl_sec_at: int = 300,
        is_at: bool = False,
        max_items: int = 512,
    ) -> bool:
        if not umo:
            return False

        now = self._now()
        store = self.recent_msg_keys.get(umo)
        if store is None:
            store = {}
            self.recent_msg_keys[umo] = store

        if len(store) > max_items or (len(store) > 0 and self._rand() < 0.05):
            expired = [k for k, ts in store.items() if now - ts > id_ttl_sec]
            for k in expired:
                store.pop(k, None)

        mid = str(msg_id).strip() if msg_id is not None else ""
        if mid:
            id_key = f"id:{mid}"
            if id_key in store and now - store[id_key] <= id_ttl_sec:
                return True
            store[id_key] = now

        fp_ttl = fp_ttl_sec_at if is_at else fp_ttl_sec
        base = f"{sender_id or ''}|{msg_text or ''}"
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()
        fp_key = f"fp:{digest}"

        if fp_key in store and now - store[fp_key] <= fp_ttl:
            return True
        store[fp_key] = now

        return False

    def get_current_state(self, umo: str) -> ChatState:
        return self.current_states.get(umo, ChatState.READ_AIR)

    def set_current_state(self, umo: str, state: ChatState):
        if umo:
            self.current_states[umo] = state

    def set_cycle_task_type(self, umo: str, task_type: str | None):
        if not umo:
            return
        if task_type == "focus":
            self.set_current_state(umo, ChatState.FOCUS)
        elif task_type == "read_air":
            self.set_current_state(umo, ChatState.READ_AIR)

        if task_type is None:
            self.cycle_tasks.pop(umo, None)
        else:
            self.cycle_tasks[umo] = task_type

    def get_cycle_task_type(self, umo: str) -> str | None:
        return self.cycle_tasks.get(umo)

    def get_last_persona_hash(self) -> str:
        return self.last_persona_hash

    def set_last_persona_hash(self, hash_val: str):
        self.last_persona_hash = hash_val

    def get_last_knowledge_hash(self) -> str:
        return self.last_knowledge_hash

    def set_last_knowledge_hash(self, hash_val: str):
        self.last_knowledge_hash = hash_val

    def increment_msg_count(self, key: str) -> int:
        return self.increment_summary_msg_count(key)

    def reset_msg_count(self, key: str):
        self.reset_summary_msg_count(key)

    def get_msg_count(self, key: str) -> int:
        return self.get_summary_msg_count(key)

    def increment_summary_msg_count(self, key: str) -> int:
        c = self.summary_msg_counts.get(key, 0) + 1
        self.summary_msg_counts[key] = c
        return c

    def reset_summary_msg_count(self, key: str):
        self.summary_msg_counts[key] = 0

    def get_summary_msg_count(self, key: str) -> int:
        return self.summary_msg_counts.get(key, 0)

    def increment_read_air_msg_count(self, key: str) -> int:
        c = self.read_air_msg_counts.get(key, 0) + 1
        self.read_air_msg_counts[key] = c
        return c

    def reset_read_air_msg_count(self, key: str):
        self.read_air_msg_counts[key] = 0

    def get_read_air_msg_count(self, key: str) -> int:
        return self.read_air_msg_counts.get(key, 0)

    def set_processing(self, umo: str, is_processing: bool):
        if umo:
            self.processing_flags[umo] = is_processing

    def is_processing(self, umo: str) -> bool:
        return self.processing_flags.get(umo, False)

    def set_interrupt(self, umo: str):
        if umo:
            self.interrupt_flags[umo] = self._now()

    def check_interrupt(self, umo: str, start_time: float) -> bool:
        if not umo:
            return False
        interrupt_time = self.interrupt_flags.get(umo, 0)
        return interrupt_time > start_time

    def set_self_id(self, umo: str, self_id: str):
        if umo and self_id:
            self.bot_self_ids[umo] = str(self_id)

    def get_self_id(self, umo: str) -> str | None:
        return self.bot_self_ids.get(umo)

    def set_self_name(self, umo: str, self_name: str):
        if umo and self_name:
            self.bot_self_names[umo] = str(self_name)

    def get_self_name(self, umo: str) -> str | None:
        return self.bot_self_names.get(umo)

    def add_session(self, umo: str):
        if umo:
            self.sessions.add(umo)

    def get_sessions(self):
        return self.sessions

    def set_last_processed(self, umo: str, timestamp: float):
        self.last_processed[umo] = timestamp

    def get_last_processed(self, umo: str) -> float:
        return self.last_processed.get(umo, 0.0)

    def set_last_cycle_cutoff(self, umo: str, timestamp: float):
        if not umo:
            return
        self.last_cycle_cutoff[umo] = float(timestamp or 0.0)

    def get_last_cycle_cutoff(self, umo: str) -> float:
        return float(self.last_cycle_cutoff.get(umo, 0.0) or 0.0)

    def set_focus(self, umo: str, duration_sec: int):
        now = self._now()
        t = now + max(1, int(duration_sec))
        self.focus_until[umo] = t
        self.focus_start[umo] = now
        self.set_current_state(umo, ChatState.FOCUS)

    def is_in_focus(self, umo: str) -> bool:
        t = self.focus_until.get(umo)
        now = self._now()
        if not t:
            if self.get_current_state(umo) == ChatState.FOCUS:
                self.set_current_state(umo, ChatState.READ_AIR)
            return False
        if now < t:
            if self.get_current_state(umo) != ChatState.FOCUS:
                self.set_current_state(umo, ChatState.FOCUS)
            return True

        self.focus_until.pop(umo, None)
        if self.get_current_state(umo) == ChatState.FOCUS:
            self.set_current_state(umo, ChatState.READ_AIR)
        return False

    def get_focus_start(self, umo: str) -> float:
        return self.focus_start.get(umo, 0.0)

    def get_focus_remaining_sec(self, umo: str) -> float:
        t = self.focus_until.get(umo)
        if not t:
            return 0.0
        remain = float(t) - self._now()
        if remain < 0:
            return 0.0
        return remain

    def get_focus_delay(self, umo: str, default: float = 15.0) -> float:
        return self.focus_delays.get(umo, default)

    def adjust_focus_delay(
        self, umo: str, delta: float, min_val: float = 3.0, max_val: float = 15.0
    ):
        if not umo:
            return
        curr = self.focus_delays.get(umo, max_val)
        new_val = max(min_val, min(max_val, curr + delta))
        self.focus_delays[umo] = new_val

    def set_focus_delay(self, umo: str, value: float, min_val: float = 1.0):
        if not umo:
            return
        v = float(value)
        if v < min_val:
            v = min_val
        self.focus_delays[umo] = v

    def reset_focus_delay(self, umo: str, default: float = 15.0):
        if umo:
            self.focus_delays[umo] = default

    def set_reply_target(self, umo: str, msg_id: str | None, reply_msg_id: str | None):
        if not umo or not msg_id or not reply_msg_id:
            return
        d = self.reply_targets.get(umo)
        if d is None:
            d = {}
            self.reply_targets[umo] = d
        d[str(msg_id)] = str(reply_msg_id)
        ts_map = self.reply_targets_seen_ts.get(umo)
        if ts_map is None:
            ts_map = {}
            self.reply_targets_seen_ts[umo] = ts_map
        ts_map[str(msg_id)] = self._now()
        self.cleanup_reply_data(umo)

    def get_reply_target(self, umo: str, msg_id: str | None) -> str | None:
        if not umo or not msg_id:
            return None
        d = self.reply_targets.get(umo)
        if not d:
            return None
        return d.get(str(msg_id))

    def set_reply_summary(
        self,
        umo: str,
        reply_msg_id: str | None,
        sender_id: str | None,
        sender_name: str | None,
        text: str | None,
    ):
        if not umo or not reply_msg_id:
            return
        d = self.reply_summaries.get(umo)
        if d is None:
            d = {}
            self.reply_summaries[umo] = d
        d[str(reply_msg_id)] = (
            str(sender_id or ""),
            str(sender_name or ""),
            str(text or ""),
            self._now(),
        )
        self.cleanup_reply_data(umo)

    def get_reply_summary(
        self, umo: str, reply_msg_id: str | None
    ) -> tuple[str, str, str, float] | None:
        if not umo or not reply_msg_id:
            return None
        d = self.reply_summaries.get(umo)
        if not d:
            return None
        return d.get(str(reply_msg_id))

    def cleanup_reply_data(
        self,
        umo: str,
        *,
        reply_targets_ttl_sec: float = 7 * 24 * 3600,
        reply_targets_max_size: int = 4096,
        reply_summaries_ttl_sec: float = 7 * 24 * 3600,
        reply_summaries_max_size: int = 4096,
    ):
        if not umo:
            return
        now = self._now()

        t_map = self.reply_targets_seen_ts.get(umo)
        d = self.reply_targets.get(umo)
        if t_map and d:
            if reply_targets_ttl_sec > 0:
                expired = [
                    k
                    for k, ts in t_map.items()
                    if ts is not None and now - float(ts or 0.0) > reply_targets_ttl_sec
                ]
                for k in expired:
                    t_map.pop(k, None)
                    d.pop(k, None)
            if reply_targets_max_size > 0 and len(t_map) > reply_targets_max_size:
                items = sorted(t_map.items(), key=lambda kv: float(kv[1] or 0.0))
                for k, _ in items[: max(0, len(items) - reply_targets_max_size)]:
                    t_map.pop(k, None)
                    d.pop(k, None)

        s_map = self.reply_summaries.get(umo)
        if s_map:
            if reply_summaries_ttl_sec > 0:
                expired = [
                    k
                    for k, v in s_map.items()
                    if v is not None and now - float(v[3] or 0.0) > reply_summaries_ttl_sec
                ]
                for k in expired:
                    s_map.pop(k, None)
            if reply_summaries_max_size > 0 and len(s_map) > reply_summaries_max_size:
                items = sorted(s_map.items(), key=lambda kv: float(kv[1][3] or 0.0))
                for k, _ in items[: max(0, len(items) - reply_summaries_max_size)]:
                    s_map.pop(k, None)

    def set_message_meta(self, umo: str, msg_id: str | None, meta: dict | None):
        if not umo or not msg_id or not isinstance(meta, dict):
            return
        mid = str(msg_id).strip()
        if not mid:
            return
        d = self.message_metas.get(umo)
        if d is None:
            d = {}
            self.message_metas[umo] = d
        ts_map = self.message_meta_seen_ts.get(umo)
        if ts_map is None:
            ts_map = {}
            self.message_meta_seen_ts[umo] = ts_map
        d[mid] = deepcopy(meta)
        ts_map[mid] = self._now()
        self.cleanup_message_meta(umo)

    def get_message_meta(self, umo: str, msg_id: str | None) -> dict | None:
        if not umo or not msg_id:
            return None
        d = self.message_metas.get(umo)
        if not d:
            return None
        meta = d.get(str(msg_id).strip())
        if not isinstance(meta, dict):
            return None
        return deepcopy(meta)

    def cleanup_message_meta(
        self,
        umo: str,
        *,
        ttl_sec: float = 7 * 24 * 3600,
        max_size: int = 4096,
    ):
        if not umo:
            return
        d = self.message_metas.get(umo)
        ts_map = self.message_meta_seen_ts.get(umo)
        if not d or not ts_map:
            return
        now = self._now()
        if ttl_sec > 0:
            expired = [
                k
                for k, ts in ts_map.items()
                if ts is not None and now - float(ts or 0.0) > ttl_sec
            ]
            for k in expired:
                ts_map.pop(k, None)
                d.pop(k, None)
        if max_size > 0 and len(ts_map) > max_size:
            items = sorted(ts_map.items(), key=lambda kv: float(kv[1] or 0.0))
            for k, _ in items[: max(0, len(items) - max_size)]:
                ts_map.pop(k, None)
                d.pop(k, None)

    def set_heartbeat_start(self, umo: str):
        if umo:
            self.heartbeat_start[umo] = self._now()

    def get_heartbeat_start(self, umo: str) -> float:
        return self.heartbeat_start.get(umo, 0.0)

    def clear_heartbeat_start(self, umo: str):
        if umo:
            self.heartbeat_start.pop(umo, None)

    def clear(self):
        self.sessions.clear()
        self.focus_until.clear()
        self.last_msg.clear()
        self.last_read_id.clear()
        self.pending_msgs.clear()
        self.recent_msg_keys.clear()
        self.reply_targets.clear()
        self.reply_targets_seen_ts.clear()
        self.reply_summaries.clear()
        self.message_metas.clear()
        self.message_meta_seen_ts.clear()
        self.last_moods.clear()
        self.last_mood_states.clear()
        self.last_mood_intensity.clear()
        self.mood_state_expires_at.clear()
        self.last_affinity_deltas.clear()
        self.poke_preapply_msg_ids.clear()
        self.no_reply_msg_ids.clear()

    def mark_no_reply_message_ids(
        self,
        umo: str,
        msg_ids: list[str] | None,
        *,
        ttl_sec: float = 24 * 3600,
        max_size: int = 1024,
    ):
        if not umo or not msg_ids:
            return
        now = self._now()
        bucket = self.no_reply_msg_ids.get(umo)
        if bucket is None:
            bucket = {}
            self.no_reply_msg_ids[umo] = bucket

        for mid in msg_ids:
            s = str(mid or "").strip()
            if not s:
                continue
            bucket[s] = now

        if ttl_sec > 0 and bucket:
            expired = [k for k, ts in bucket.items() if now - float(ts or 0.0) > ttl_sec]
            for k in expired:
                bucket.pop(k, None)

        if max_size > 0 and len(bucket) > max_size:
            items = sorted(bucket.items(), key=lambda kv: float(kv[1] or 0.0))
            for k, _ in items[: max(0, len(bucket) - max_size)]:
                bucket.pop(k, None)

    def is_no_reply_message_id(self, umo: str, msg_id: str | None) -> bool:
        if not umo or msg_id is None:
            return False
        s = str(msg_id).strip()
        if not s:
            return False
        bucket = self.no_reply_msg_ids.get(umo) or {}
        return s in bucket

    def update_last_message(
        self,
        umo: str,
        msg_id: str | None,
        msg_text: str | None,
        sender_name: str | None,
        sender_id: str | None,
    ):
        if not umo:
            return
        mid = msg_id or ""
        mtxt = msg_text or ""
        sname = sender_name or ""
        sid = sender_id or ""
        self.last_msg[umo] = (mid, mtxt, sname, sid)
        now = self._now()
        buf = self.pending_msgs.get(umo)
        if buf is None:
            buf = []
            self.pending_msgs[umo] = buf
        buf.append((mid, mtxt, sname, sid, now))
        if len(buf) > 50:
            buf.pop(0)

    def get_last_message(self, umo: str):
        return self.last_msg.get(umo)

    def clear_last_message(self, umo: str):
        if not umo:
            return
        self.last_msg.pop(umo, None)

    def mark_last_read(self, umo: str, msg_id: str | None, msg_text: str | None = None):
        if not umo:
            return
        self.last_read_id[umo] = msg_id or msg_text or ""

    def has_read(
        self, umo: str, msg_id: str | None, msg_text: str | None = None
    ) -> bool:
        if msg_id is None and not msg_text:
            return False
        return self.last_read_id.get(umo, None) == (msg_id or msg_text)

    def get_pending_messages(self, umo: str) -> list[tuple[str, str, str, str, float]]:
        return list(self.pending_msgs.get(umo, []))

    def get_pending_count(self, umo: str) -> int:
        return len(self.pending_msgs.get(umo, []))

    def clear_pending_messages(self, umo: str):
        self.pending_msgs.pop(umo, None)

    def clear_pending_up_to(self, umo: str, timestamp: float):
        if umo not in self.pending_msgs:
            return

        buf = self.pending_msgs[umo]
        new_buf = [m for m in buf if m[4] > timestamp]

        if not new_buf:
            self.pending_msgs.pop(umo, None)
        else:
            self.pending_msgs[umo] = new_buf

    def clear_pending_first_n(self, umo: str, n: int):
        if not umo or n <= 0:
            return
        buf = self.pending_msgs.get(umo)
        if not buf:
            return
        if n >= len(buf):
            self.pending_msgs.pop(umo, None)
            return
        self.pending_msgs[umo] = buf[n:]

    def prune_pending_messages(
        self, umo: str, cutoff_ts: float, keep_last: int = 1
    ) -> list[tuple[str, str, str, str, float]]:
        if not umo:
            return []
        buf = self.pending_msgs.get(umo)
        if not buf:
            return []
        new_buf = [m for m in buf if m[4] >= cutoff_ts]
        if keep_last > 0 and len(new_buf) < keep_last and len(buf) >= keep_last:
            new_buf = buf[-keep_last:]
        if not new_buf:
            self.pending_msgs.pop(umo, None)
            return []
        self.pending_msgs[umo] = new_buf
        return list(new_buf)


__all__ = ["ChatState", "MemChatState"]
