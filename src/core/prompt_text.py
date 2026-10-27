"""Turn media placeholders into something the model reads as an action, not speech.

`vision_media` rewrites an image-only message into `[图片内容: …]`, so a line like

    老白/123456789: [图片内容: Q版蓝发女仆装角色抱盆、委屈鼓脸握拳的小动图…]

sits exactly where a person's sentence would sit. The model then quotes the
caption back as if 老白 had said it. Rendering the caption as `[发图]` plus a
one-line legend fixes both the misreading and the token bloat (captions are
often 100+ characters).
"""

import re

_IMAGE_RE = re.compile(r"\[图片内容[:：]\s*(.*?)\]", re.S)
_MARK = "[发图]"

_MEDIA_LEGEND = (
    "【图片消息格式】\n"
    "- 行首或句中的 [发图] 表示这个人发了一张图/表情包，其后的文字是系统对画面的描述，"
    "不是他打的字。引用时请当成“他发了张图”，不要当成他说了那句话。\n"
    "- 纯 [发图] 行意味着这条消息没有文字，只有一个表情包；不要把它当成一句话来回答。"
)


def humanize_media(text: str, max_desc: int = 80) -> str:
    """Rewrite `[图片内容: …]` markers into `[发图]` plus a trimmed caption.

    Args:
        text: Any prompt fragment that may contain image markers.
        max_desc: Longest caption kept, in characters.

    Returns:
        The rewritten text; unchanged when it holds no image marker.
    """
    if not text or "[图片内容" not in text:
        return text

    def _repl(match: re.Match) -> str:
        desc = " ".join(str(match.group(1) or "").split())
        if len(desc) > max_desc:
            desc = desc[:max_desc] + "…"
        return f"{_MARK}{desc}" if desc else _MARK

    out = _IMAGE_RE.sub(_repl, text)
    # A truncated or malformed marker still must not read as speech.
    out = out.replace("[图片内容:", _MARK).replace("[图片内容：", _MARK)
    return out


def with_media_legend(text: str) -> str:
    """Append the one-line legend when the prompt actually contains images."""
    if not text or _MARK not in text:
        return text
    if "【图片消息格式】" in text:
        return text
    return text.rstrip() + "\n\n" + _MEDIA_LEGEND
