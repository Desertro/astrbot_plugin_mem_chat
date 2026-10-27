import time
from collections.abc import Callable
from typing import Any

from .member_tags import format_sender


class RefItem:
    def __init__(
        self,
        message_id: str,
        text: str,
        sender_name: str,
        sender_id: str,
        ts: float,
        kind: str,
    ):
        self.message_id = str(message_id or "")
        self.text = str(text or "")
        self.sender_name = str(sender_name or "")
        self.sender_id = str(sender_id or "")
        self.ts = float(ts or 0.0)
        self.kind = str(kind or "")


def _sanitize_context_text(text: str) -> tuple[str, bool]:
    s = str(text or "").strip()
    if not s:
        return "", False
    lowered = s.lower()
    if (
        "type=<componenttype.json" in lowered
        or "componenttype.json" in lowered
        or ("data='" in lowered and "componenttype" in lowered)
    ):
        return "[卡片消息]", True
    if any(k in s for k in ("[转发消息]", "聊天记录", "查看") ):
        return "[转发消息]", True
    if len(s) > 800:
        return s[:800].rstrip() + "…", True
    return s, False


def _fmt_ts(ts: float) -> str:
    try:
        if ts <= 0:
            return "--:--"
        return time.strftime("%H:%M", time.localtime(ts))
    except Exception:
        return "--:--"


def _format_time_marker(ts: float, now_ts: float | None = None) -> str:
    try:
        if ts <= 0:
            return "--:--"
        now_val = float(now_ts or time.time())
        dt = time.localtime(ts)
        now_dt = time.localtime(now_val)
        if dt.tm_year == now_dt.tm_year and dt.tm_yday == now_dt.tm_yday:
            return time.strftime("今天 %H:%M", dt)
        if dt.tm_year == now_dt.tm_year and dt.tm_yday == now_dt.tm_yday - 1:
            return time.strftime("昨天 %H:%M", dt)
        if dt.tm_year == now_dt.tm_year:
            return time.strftime("%m-%d %H:%M", dt)
        return time.strftime("%Y-%m-%d %H:%M", dt)
    except Exception:
        return "--:--"


def _coerce_target_info(
    meta: dict[str, Any] | None,
    fallback_sender_name: str,
) -> tuple[str, str, str]:
    m = meta or {}
    target_type = str(m.get("inferred_target_type") or m.get("explicit_target_type") or "").strip()
    target_id = str(m.get("inferred_target_id") or m.get("explicit_target_id") or "").strip()
    target_name = str(m.get("inferred_target_name") or m.get("explicit_target_name") or "").strip()
    if target_type == "bot":
        return "bot", target_id or "bot", target_name or "机器人"
    if target_type in ("group", "group_all"):
        label = "群聊全体" if target_type == "group_all" else "群聊"
        return target_type, target_id or "group", target_name or label
    if target_id or target_name:
        return "user", target_id, target_name or fallback_sender_name
    return "group", "group", "群聊"


def _render_dialogue_line(
    it: RefItem,
    meta: dict[str, Any] | None,
    sender_tags: dict[str, str] | None = None,
) -> str:
    prefix = ""
    kind_norm = (it.kind or "").strip().lower()
    if kind_norm == "referenced":
        prefix = "(引用) "
    opaque = False
    if kind_norm.endswith("_opaque") or it.text in ("[卡片消息]", "[转发消息]"):
        opaque = True
    if opaque:
        prefix = f"{prefix}(不可见) "
    who = format_sender(it.sender_name, it.sender_id, sender_tags)
    _tt, _tid, target_name = _coerce_target_info(meta, it.sender_name or "User")
    if who:
        return f"{prefix}{who} -> {target_name}: {it.text}".strip()
    return f"{prefix}{target_name}: {it.text}".strip()


def _sort_key(ts: float, kind: str, message_id: str) -> tuple[float, int, str]:
    kind_norm = (kind or "").strip().lower()
    kind_rank = 2
    if kind_norm == "referenced":
        kind_rank = 0
    elif kind_norm == "current":
        kind_rank = 1
    return (float(ts or 0.0), kind_rank, str(message_id or ""))


def _topo_sort(
    nodes: dict[str, RefItem],
    edges: list[tuple[str, str]],
) -> list[RefItem]:
    adj: dict[str, set[str]] = {k: set() for k in nodes}
    indeg: dict[str, int] = dict.fromkeys(nodes, 0)

    for src, dst in edges:
        if src not in nodes or dst not in nodes:
            continue
        if dst in adj[src]:
            continue
        adj[src].add(dst)
        indeg[dst] += 1

    ready = sorted(
        [k for k, d in indeg.items() if d == 0],
        key=lambda k: _sort_key(nodes[k].ts, nodes[k].kind, k),
    )
    out: list[str] = []
    while ready:
        k = ready.pop(0)
        out.append(k)
        for nb in sorted(
            adj.get(k, set()),
            key=lambda x: _sort_key(nodes[x].ts, nodes[x].kind, x),
        ):
            indeg[nb] -= 1
            if indeg[nb] == 0:
                ready.append(nb)
        ready.sort(key=lambda x: _sort_key(nodes[x].ts, nodes[x].kind, x))

    if len(out) != len(nodes):
        remaining = [k for k in nodes.keys() if k not in out]
        remaining.sort(key=lambda k: _sort_key(nodes[k].ts, nodes[k].kind, k))
        out.extend(remaining)

    return [nodes[k] for k in out]


def build_ref_context(
    pending_msgs: list[tuple[str, str, str, str, float]] | None,
    current_event: Any | None,
    current_text: str | None,
    get_reply_target: Callable[[str], str | None] | None,
    get_reply_summary: Callable[[str], tuple[str, str, str, float] | None] | None,
    get_message_meta: Callable[[str], dict | None] | None = None,
    max_items: int = 30,
    include_referenced: bool = True,
    time_gap_min_minutes: int = 15,
    time_marker_every_n_messages: int = 4,
    sender_tags: dict[str, str] | None = None,
) -> tuple[list[str], list[tuple[str, str, str, float, str]], dict[int, str]]:
    base: list[RefItem] = []
    seen: set[str] = set()

    if pending_msgs:
        for mid, txt, sname, sid, ts in pending_msgs[-max_items:]:
            mid_str = str(mid or "").strip()
            if not mid_str or mid_str in seen:
                continue
            seen.add(mid_str)
            safe_txt, opaque = _sanitize_context_text(txt or "")
            base.append(
                RefItem(
                    message_id=mid_str,
                    text=safe_txt,
                    sender_name=sname or "",
                    sender_id=sid or "",
                    ts=float(ts or 0.0),
                    kind="recent_opaque" if opaque else "recent",
                )
            )

    try:
        cur_mid = None
        msg_obj = getattr(current_event, "message_obj", None)
        if msg_obj is not None:
            cur_mid = getattr(msg_obj, "message_id", None)
        cur_mid_str = str(cur_mid or "").strip()
        if cur_mid_str and cur_mid_str not in seen:
            seen.add(cur_mid_str)
            safe_txt, opaque = _sanitize_context_text(current_text or "")
            base.append(
                RefItem(
                    message_id=cur_mid_str,
                    text=safe_txt,
                    sender_name=getattr(current_event, "get_sender_name", lambda: "")()
                    or "",
                    sender_id=str(
                        getattr(current_event, "get_sender_id", lambda: "")() or ""
                    ),
                    ts=time.time(),
                    kind="current_opaque" if opaque else "current",
                )
            )
    except Exception:
        pass

    nodes: dict[str, RefItem] = {it.message_id: it for it in base if it.message_id}
    edges: list[tuple[str, str]] = []

    if include_referenced and get_reply_target:
        for it in list(nodes.values()):
            try:
                target = get_reply_target(str(it.message_id))
            except Exception:
                target = None
            target_id = str(target or "").strip()
            if not target_id:
                continue
            edges.append((target_id, it.message_id))
            if target_id in nodes:
                continue
            summary = None
            if get_reply_summary:
                try:
                    summary = get_reply_summary(target_id)
                except Exception:
                    summary = None
            if summary:
                s_sid, s_name, s_txt, s_ts = summary
                nodes[target_id] = RefItem(
                    message_id=target_id,
                    text=s_txt or "",
                    sender_name=s_name or "",
                    sender_id=s_sid or "",
                    ts=float(s_ts or 0.0),
                    kind="referenced",
                )
            else:
                nodes[target_id] = RefItem(
                    message_id=target_id,
                    text="[引用消息]",
                    sender_name="",
                    sender_id="",
                    ts=max(0.0, float(it.ts or 0.0) - 0.001),
                    kind="referenced",
                )

    ordered = _topo_sort(nodes, edges)

    lines: list[str] = []
    candidates: list[tuple[str, str, str, float, str]] = []
    index_to_id: dict[int, str] = {}
    now_ts = time.time()
    last_marker_ts = None
    since_marker = 0
    gap_sec = max(60, int(time_gap_min_minutes or 15) * 60)
    every_n = max(1, int(time_marker_every_n_messages or 4))

    idx = 0
    for it in ordered:
        if not it.message_id:
            continue
        need_marker = False
        if last_marker_ts is None:
            need_marker = True
        elif it.ts > 0 and (it.ts - float(last_marker_ts or 0.0)) >= gap_sec:
            need_marker = True
        elif since_marker >= every_n:
            need_marker = True
        if need_marker:
            lines.append(f"[时间] {_format_time_marker(it.ts, now_ts)}")
            last_marker_ts = it.ts
            since_marker = 0
        idx += 1
        index_to_id[idx] = it.message_id
        meta = None
        if get_message_meta:
            try:
                meta = get_message_meta(it.message_id)
            except Exception:
                meta = None
        body = _render_dialogue_line(it, meta, sender_tags)
        ts_str = _fmt_ts(it.ts)
        lines.append(f"[{ts_str}] [REF:{idx}] {body}".strip())
        candidates.append((it.message_id, it.text, it.sender_id, it.ts, it.kind))
        since_marker += 1

    return lines, candidates, index_to_id
