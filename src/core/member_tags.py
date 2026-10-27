"""Read cached group member roles, levels and titles for prompt labels."""

import sqlite3
from pathlib import Path

TAGS_DB_PATH = (
    Path(__file__).resolve().parents[4] / "plugin_data" / "member_tags" / "tags.db"
)

_ROLE_LABELS = {"owner": "群主", "admin": "管理员"}


def load_sender_tags(group_id: str) -> dict[str, str]:
    """Load sender suffix labels for one group from the member_tags cache.

    Args:
        group_id: Platform group id.

    Returns:
        Mapping of sender id to a suffix such as "（管理员·LV32·老登）".
        Missing or unreadable caches return an empty mapping.
    """
    gid = str(group_id or "").strip()
    if not gid or not TAGS_DB_PATH.exists():
        return {}

    con = None
    try:
        con = sqlite3.connect(TAGS_DB_PATH, timeout=2)
        rows = con.execute(
            "select user_id, role, level, title from member_tags where group_id = ?",
            (gid,),
        ).fetchall()
    except Exception:
        return {}
    finally:
        if con is not None:
            con.close()

    tags: dict[str, str] = {}
    for user_id, role, level, title in rows:
        parts: list[str] = []
        role_label = _ROLE_LABELS.get(str(role or "").strip())
        if role_label:
            parts.append(role_label)
        try:
            lv = int(str(level).strip())
        except (TypeError, ValueError):
            lv = 0
        if lv > 0:
            parts.append(f"LV{lv}")
        title_text = str(title or "").strip()
        if title_text:
            parts.append(title_text)
        if parts:
            tags[str(user_id)] = f"（{'·'.join(parts)}）"
    return tags


def format_sender(
    name: str | None,
    sender_id: str | None,
    sender_tags: dict[str, str] | None = None,
) -> str:
    """Format a sender as "name/QQ（tags）" for prompt lines.

    Args:
        name: Sender display name.
        sender_id: Sender QQ id.
        sender_tags: Mapping produced by :func:`load_sender_tags`.

    Returns:
        The formatted sender label, or an empty string when both name and id
        are missing.
    """
    name_text = str(name or "").strip()
    sid = str(sender_id or "").strip()
    who = f"{name_text}/{sid}".strip("/")
    if not who:
        return ""
    return f"{who}{(sender_tags or {}).get(sid, '')}"
