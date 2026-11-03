"""Input-only name calls and optional model attention controls."""

import math
import re

from astrbot.api.message_components import Plain


def detect_name_call(messages, names) -> bool:
    """Match configured names in the sender's own text components.

    Args:
        messages: Top-level incoming message components.
        names: Configured character names.

    Returns:
        Whether this message is a possible name call.
    """
    if not isinstance(names, list):
        return False
    text = " ".join(part.text for part in messages or [] if isinstance(part, Plain))
    text = re.sub(r"[《〈][^》〉]*[》〉]", "", text)
    text = text.replace("爱丽丝梦游仙境", "")
    for name in names:
        name = str(name or "").strip()
        if not name:
            continue
        pattern = re.escape(name)
        if name.isascii():
            pattern = r"(?<![a-zA-Z0-9_])" + pattern + r"(?![a-zA-Z0-9_])"
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def normalize_focus_control(control, limit: float = 120) -> dict:
    """Validate the optional attention request independently of reply content.

    Args:
        control: Structured control object returned by the model.
        limit: Maximum extension accepted from a single turn.

    Returns:
        Validated action, or an empty object for an invalid request.
    """
    focus = control.get("focus") if isinstance(control, dict) else None
    if not isinstance(focus, dict):
        return {}
    action = focus.get("action", "keep")
    if action not in ("keep", "extend", "release"):
        return {}
    result = {"action": action, "reason": str(focus.get("reason") or "")[:160]}
    if action == "extend":
        seconds = focus.get("seconds", 60)
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            return {}
        if seconds <= 0 or (isinstance(seconds, float) and not math.isfinite(seconds)):
            return {}
        result["seconds"] = float(min(seconds, max(1.0, limit)))
    return result
