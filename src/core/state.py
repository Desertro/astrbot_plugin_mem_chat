import hashlib
import random
import time
from collections.abc import Callable
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
