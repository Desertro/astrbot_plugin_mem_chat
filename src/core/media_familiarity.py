"""Session-local familiarity metadata for repeated images and memes.

The vision sidecar only describes what an image looks like. Without an explicit
occurrence count, the chat model reasonably treats a frequently repeated meme
as a brand-new image every time. This module builds a small, factual hint from
recent raw messages and a lightweight persistence table. It deliberately does
not generate canned reactions; the model still decides whether and how to
comment on the repetition.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from difflib import SequenceMatcher
from urllib.parse import unquote, urlparse

_IMAGE_RE = re.compile(r"\[图片内容[:：]\s*(.*?)\]", re.S)
_PLAIN_IMAGE_RE = re.compile(r"\[\u56fe\u7247\]")
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\s，。！？、,.!?;；:：（）()【】\[\]{}“”\"'`~…—\-_/\\|]+")
_MAX_HASH_BYTES = 64 * 1024 * 1024
_MAX_HASH_CANDIDATES = 96

_MEDIA_ALIAS_TAG_RE = re.compile(
    r"\[(记图名|改图名|删图名)[:：]\s*(?:(\d{1,2})\s*[:：]\s*)?([^\[\]\r\n]{1,40})\]"
)
_ALIAS_TARGET_RE = r"(?:这(?:个|张)?(?:表情包|动画表情|梗图|图片|图)|它)"
_ALIAS_CAPTURE_RE = r"(?P<alias>[^，。！？!?；;\r\n]{1,30})"
_ALIAS_REJECT_WORDS = {
    "啥",
    "什么",
    "谁",
    "哪个",
    "哪张",
    "啥呀",
    "什么呀",
    "表情包",
    "图片",
    "图",
}


def normalize_media_alias(value, max_len: int = 24) -> str:
    """Normalize one user-facing image alias without inventing content."""
    text = _WS_RE.sub(" ", str(value or "")).strip()
    text = text.strip(" \t\r\n「」『』“”\"'`《》〈〉()（）[]【】")
    text = re.sub(
        r"^(?:叫做?|名为|名字(?:是|叫)|称呼(?:是|叫)|标签(?:是|叫)|是)\s*",
        "",
        text,
    )
    text = text.strip(" \t\r\n，。！？、,.!?;；:：")
    text = re.sub(r"\s+", " ", text)
    if not text or len(text) > max(0, int(max_len or 24)):
        return ""
    if any(ch in text for ch in "[]<>"):
        return ""
    if text in _ALIAS_REJECT_WORDS:
        return ""
    if text.endswith(("吗", "么", "呢", "吧", "？", "?", "的")):
        return ""
    if re.search(r"(?:是谁|是什么|是哪[个张]?)$", text):
        return ""
    if re.search(r"的(?:表情包|动画表情|梗图|图片|图)$", text):
        return ""
    if re.search(r"(?:发的|做的|画的|说的|用的|截的)$", text):
        return ""
    if re.search(r"^(?:很|太|好|真|超|挺|有点|非常)", text):
        return ""
    return text


def parse_media_alias_actions(text: str) -> tuple[str, list[dict]]:
    """Strip hidden alias-action tags and return validated actions.

    Tags are deliberately parsed only after the model has decided that the
    current image is being named. The returned actions carry a 1-based image
    index when the model supplied one.
    """
    actions: list[dict] = []
    action_names = {"记图名": "set", "改图名": "replace", "删图名": "delete"}

    def _replace(match: re.Match) -> str:
        action = action_names.get(match.group(1) or "", "")
        alias = normalize_media_alias(match.group(3) or "")
        if not action or not alias:
            return ""
        index = None
        try:
            raw_index = match.group(2)
            if raw_index:
                index = int(raw_index)
                if index < 1:
                    index = None
        except Exception:
            index = None
        actions.append({"action": action, "alias": alias, "image_index": index})
        return ""

    return _MEDIA_ALIAS_TAG_RE.sub(_replace, str(text or "")), actions


def extract_media_alias_intent(text: str) -> dict | None:
    """Extract only explicit, low-ambiguity image-naming statements.

    This is a safety net for turns where the model chooses no_reply. Ambiguous
    mentions still belong to the model, which can emit a hidden alias action
    after reading the surrounding conversation.
    """
    value = normalize_caption(text, max_len=500)
    if not value:
        return None

    delete_patterns = [
        rf"(?:别|不要)(?:再)?\s*(?:叫|管)\s*(?:{_ALIAS_TARGET_RE})?\s*(?:叫|作)?\s*(?P<alias>[^，。！？!?；;\r\n]{{1,30}}?)\s*(?:了)?\s*$",
        rf"(?:删掉|删除|忘掉)\s*(?:{_ALIAS_TARGET_RE})?\s*(?:的)?\s*(?:标签|名字|称呼)\s*[:：]?\s*{_ALIAS_CAPTURE_RE}",
    ]
    for pattern in delete_patterns:
        match = re.search(pattern, value)
        if match:
            alias = normalize_media_alias(match.group("alias"))
            if alias:
                return {"action": "delete", "alias": alias}

    replace_patterns = [
        rf"(?:把)?\s*{_ALIAS_TARGET_RE}\s*(?:的)?\s*(?:名字|名称|标签|称呼)?\s*(?:改|换)(?:成|为|叫|作)\s*{_ALIAS_CAPTURE_RE}",
        rf"(?:以后|从今以后|之后|往后)\s*(?:就)?\s*(?:叫|管)\s*(?:{_ALIAS_TARGET_RE})?\s*(?:叫|作|为)?\s*{_ALIAS_CAPTURE_RE}",
        rf"{_ALIAS_TARGET_RE}\s*(?:其实|应该|正确(?:的)?(?:名字|称呼|标签)?)\s*(?:是|叫(?:做)?|名为)\s*{_ALIAS_CAPTURE_RE}",
        rf"{_ALIAS_TARGET_RE}\s*不是[^，,。！？!?]{{1,20}}[，,]\s*(?:而是|是)\s*{_ALIAS_CAPTURE_RE}",
    ]
    for pattern in replace_patterns:
        match = re.search(pattern, value)
        if match:
            alias = normalize_media_alias(match.group("alias"))
            if alias:
                return {"action": "replace", "alias": alias}

    set_patterns = [
        rf"{_ALIAS_TARGET_RE}\s*(?:的)?\s*(?:名字|名称|标签|称呼)?\s*(?:就|其实|应该)?\s*(?:是|叫(?:做)?|名为|称作)\s*{_ALIAS_CAPTURE_RE}",
        rf"(?:记住|记一下|记下|存一下)[，,\s]*{_ALIAS_TARGET_RE}\s*(?:的)?\s*(?:名字|标签|称呼)?\s*(?:是|叫(?:做)?)\s*{_ALIAS_CAPTURE_RE}",
        rf"(?:别人|有人|群友)\s*(?:说|讲|认为)[，,\s]*{_ALIAS_TARGET_RE}\s*(?:就|其实|应该)?\s*(?:是|叫(?:做)?)\s*{_ALIAS_CAPTURE_RE}",
    ]
    for pattern in set_patterns:
        match = re.search(pattern, value)
        if match:
            alias = normalize_media_alias(match.group("alias"))
            if alias:
                return {"action": "set", "alias": alias}
    return None


_GENERIC_CAPTIONS = {
    "图片",
    "表情包",
    "动图",
    "gif",
    "黄暴图片",
    "聊天记录",
    "聊天截图",
    "截图",
    "二次元图片",
    "动漫图片",
    "未知图片",
    "无文字",
}


def normalize_caption(text: str, max_len: int = 500) -> str:
    """Collapse a vision caption into one bounded line."""
    value = _WS_RE.sub(" ", str(text or "")).strip()
    if max_len > 0 and len(value) > max_len:
        value = value[:max_len]
    return value


def extract_image_captions(text: str) -> list[str]:
    """Return all vision captions from a rendered message block, in order."""
    return [normalize_caption(m.group(1)) for m in _IMAGE_RE.finditer(str(text or ""))]


def pair_captions_with_images(text: str, image_count: int) -> list[str]:
    """Best-effort alignment between image refs and vision captions."""
    count = max(0, int(image_count or 0))
    captions = extract_image_captions(text)
    if count <= 0:
        return []
    if len(captions) == count:
        return captions
    if count == 1:
        return [captions[0] if captions else ""]
    return [captions[i] if i < len(captions) else "" for i in range(count)]


def fill_plain_image_captions(text: str, captions: list[str]) -> str:
    """Replace ordered plain-image placeholders with durable captions.

    The helper only acts when the number of plain image placeholders exactly
    matches the number of supplied captions. This prevents captions from being
    shifted onto the wrong image in mixed/multi-image messages.
    """
    raw = str(text or "")
    values = [normalize_caption(item, max_len=800) for item in (captions or [])]
    if not raw or not values:
        return raw
    matches = list(_PLAIN_IMAGE_RE.finditer(raw))
    if not matches or len(matches) != len(values):
        return raw
    parts: list[str] = []
    cursor = 0
    for match, caption in zip(matches, values):
        parts.append(raw[cursor : match.start()])
        parts.append(
            f"[\u56fe\u7247\u5185\u5bb9: {caption}]" if caption else match.group(0)
        )
        cursor = match.end()
    parts.append(raw[cursor:])
    return "".join(parts)


def _sha_text(prefix: str, value: str) -> str:
    digest = hashlib.sha256(str(value or "").encode("utf-8", "ignore")).hexdigest()
    return f"{prefix}:{digest}"


def _sha_file(path: str) -> str | None:
    try:
        if not os.path.isfile(path):
            return None
        if os.path.getsize(path) > _MAX_HASH_BYTES:
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"
    except Exception:
        return None


def _local_path(ref: str) -> str:
    value = str(ref or "").strip().strip("`").strip()
    if value.startswith("file://"):
        try:
            parsed = urlparse(value)
            value = unquote(parsed.path or "")
        except Exception:
            value = value[7:]
    return value


def media_reference_fingerprint(ref: str) -> str | None:
    """Build a stable fingerprint for an image reference.

    Local files and inline data are hashed by bytes, so the same meme matches
    even when QQ gives it a fresh temporary filename. URLs retain only their
    path as a weaker fallback.
    """
    value = str(ref or "").strip().strip("`").strip()
    if not value:
        return None
    path = _local_path(value)
    file_fp = _sha_file(path)
    if file_fp:
        return file_fp
    if value.startswith("data:"):
        try:
            payload = value.split(",", 1)[1]
            raw = base64.b64decode(payload, validate=False)
            return f"sha256:{hashlib.sha256(raw).hexdigest()}"
        except Exception:
            return _sha_text("refsha", value)
    if value.startswith(("http://", "https://")):
        try:
            parsed = urlparse(value)
            stable = f"{parsed.scheme}://{parsed.netloc}{unquote(parsed.path or '')}"
            return _sha_text("urlsha", stable)
        except Exception:
            return _sha_text("refsha", value)
    return _sha_text("refsha", path or value)


def media_familiarity_key(ref: str, caption: str = "") -> str | None:
    """Choose a persistence key, preferring image bytes over descriptions."""
    ref_fp = media_reference_fingerprint(ref)
    if ref_fp and ref_fp.startswith("sha256:"):
        return ref_fp
    caption_key = caption_match_key(caption)
    if len(caption_key) >= 8:
        return _sha_text("caption", caption_key)
    return ref_fp


def caption_match_key(text: str) -> str:
    """Normalize a caption for cheap candidate selection."""
    value = normalize_caption(text, max_len=800).lower()
    return _PUNCT_RE.sub("", value)


def _captions_may_match(current: str, previous: str) -> bool:
    """Return True when two captions are plausible versions of one image.

    File hashes make the final decision whenever available. This is only a
    cheap pre-filter to avoid hashing every historical image for every message.
    """
    a = caption_match_key(current)
    b = caption_match_key(previous)
    if not a or not b:
        return False
    if a == b:
        return True
    short = min(len(a), len(b))
    if short < 12:
        return False
    if a in b or b in a:
        return True
    try:
        return SequenceMatcher(None, a, b).ratio() >= 0.76
    except Exception:
        return False


def _caption_is_specific(caption: str) -> bool:
    key = caption_match_key(caption)
    if len(key) < 12:
        return False
    return key not in _GENERIC_CAPTIONS


def parse_image_refs(value) -> list[str]:
    """Parse either a JSON column value or a Python list into image refs."""
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item or "").strip()]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item or "").strip()]
    except Exception:
        pass
    return []


def _row_value(row, key: str, default=None):
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _historical_match(
    *,
    current_ref: str,
    current_caption: str,
    rows: list,
    max_hash_candidates: int = _MAX_HASH_CANDIDATES,
) -> dict:
    """Find confirmed earlier occurrences of one current image."""
    current_content_fp = media_reference_fingerprint(current_ref)
    current_has_content = bool(
        current_content_fp and current_content_fp.startswith("sha256:")
    )
    matched_keys: set[tuple] = set()
    matched_times: list[float] = []
    last_sender_name = ""
    hash_attempts = 0

    for row in rows or []:
        try:
            created_at = float(_row_value(row, "created_at", 0.0) or 0.0)
        except Exception:
            created_at = 0.0
        past_refs = parse_image_refs(_row_value(row, "image_urls_json", "[]"))
        if not past_refs:
            continue
        past_text = str(_row_value(row, "text", "") or "")
        past_captions = pair_captions_with_images(past_text, len(past_refs))
        sender_name = str(_row_value(row, "sender_name", "") or "")
        message_id = str(_row_value(row, "message_id", "") or "")

        for idx, past_ref in enumerate(past_refs):
            past_caption = past_captions[idx] if idx < len(past_captions) else ""
            if not _captions_may_match(current_caption, past_caption):
                continue

            confirmed = False
            if current_has_content:
                if hash_attempts >= max(1, int(max_hash_candidates or 1)):
                    continue
                hash_attempts += 1
                past_fp = media_reference_fingerprint(past_ref)
                confirmed = past_fp == current_content_fp
            elif _caption_is_specific(current_caption):
                # Without a readable current file, only an exact specific
                # description is strong enough to count. Generic labels such
                # as "黄暴图片" would otherwise collapse unrelated images.
                confirmed = caption_match_key(current_caption) == caption_match_key(
                    past_caption
                )

            if not confirmed:
                continue
            key = (message_id, idx, past_ref)
            if key in matched_keys:
                continue
            matched_keys.add(key)
            matched_times.append(created_at)
            if created_at >= max(matched_times or [0.0]) and sender_name:
                last_sender_name = sender_name

    latest_time = max(matched_times) if matched_times else 0.0
    earliest_time = min(matched_times) if matched_times else 0.0
    if latest_time:
        for row in rows or []:
            try:
                if float(_row_value(row, "created_at", 0.0) or 0.0) != latest_time:
                    continue
            except Exception:
                continue
            sender_name = str(_row_value(row, "sender_name", "") or "")
            if sender_name:
                last_sender_name = sender_name
            break
    return {
        "prior_occurrences": len(matched_times),
        "first_seen_at": earliest_time,
        "last_seen_at": latest_time,
        "last_sender_name": last_sender_name,
        "content_fingerprint": current_content_fp,
    }


def analyze_recent_media_occurrences(
    image_refs: list[str],
    captions: list[str],
    rows: list,
    max_hash_candidates: int = _MAX_HASH_CANDIDATES,
) -> list[dict]:
    """Analyze each current image against a bounded recent-message window."""
    results: list[dict] = []
    for idx, ref in enumerate(image_refs or []):
        caption = captions[idx] if idx < len(captions) else ""
        matched = _historical_match(
            current_ref=ref,
            current_caption=caption,
            rows=rows,
            max_hash_candidates=max_hash_candidates,
        )
        results.append(
            {
                "ref": ref,
                "caption": normalize_caption(caption),
                "fingerprint": media_familiarity_key(ref, caption),
                **matched,
            }
        )
    return results


_STICKER_CAPTION_HINTS = (
    "表情包",
    "动画表情",
    "梗图",
    "meme",
    "sticker",
)


def classify_media_kind(sub_type: int = 0, caption: str = "") -> str:
    """Return a coarse media kind without claiming more than we know.

    QQ marks animated stickers with sub_type=1. Static memes may arrive as
    ordinary images, so a few caption hints are used as a fallback; generic
    facial-expression wording is deliberately not enough to call it a sticker.
    """
    try:
        if int(sub_type or 0) == 1:
            return "sticker"
    except Exception:
        pass
    text = normalize_caption(caption, max_len=800).lower()
    if any(hint in text for hint in _STICKER_CAPTION_HINTS):
        return "sticker"
    return "image"


def _entry_aliases(entry: dict) -> list[str]:
    aliases: list[str] = []
    for raw in entry.get("aliases") or []:
        value = normalize_media_alias(raw)
        if value and value not in aliases:
            aliases.append(value)
    return aliases[:4]


def _format_alias_phrase(aliases: list[str]) -> str:
    if not aliases:
        return ""
    quoted = "、".join(f"“{alias}”" for alias in aliases)
    if len(aliases) == 1:
        return f"；本会话里叫它{quoted}"
    return f"；本会话里的称呼有{quoted}"


def format_media_familiarity_block(
    entries: list[dict],
    now: float | None = None,
    *,
    include_current: bool = False,
    alias_learning: bool = False,
) -> str:
    """Render facts and behavioural guidance for the current model turn.

    ``include_current`` is used only by the live turn that carries images. It
    lets the model bind an explicit name to an image that has not repeated yet.
    Historical calls keep the older repeated-image-only behaviour.
    """
    selected: list[dict] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        try:
            seen_count = int(entry.get("seen_count") or 0)
        except Exception:
            seen_count = 0
        aliases = _entry_aliases(entry)
        is_current = bool(entry.get("is_current", False))
        if seen_count >= 2 or aliases or (include_current and is_current):
            selected.append({**entry, "_seen_count": seen_count, "_aliases": aliases})
    if not selected:
        return ""

    lines = ["【表情熟悉度 · 系统观察，不是台词】"]
    has_repeated = any(entry["_seen_count"] >= 2 for entry in selected)
    has_sticker = False
    for entry in selected[:4]:
        seen_count = int(entry["_seen_count"])
        aliases = list(entry["_aliases"])
        caption = normalize_caption(entry.get("caption") or "", max_len=52)
        shown_caption = f"（画面：{caption}）" if caption else ""
        media_kind = str(entry.get("media_kind") or "image").strip().lower()
        is_sticker = media_kind == "sticker" or seen_count >= 2
        has_sticker = has_sticker or is_sticker
        kind_label = "表情包" if is_sticker else "图片"
        alias_phrase = _format_alias_phrase(aliases)

        if seen_count >= 2:
            lines.append(
                f"- 当前{kind_label}{shown_caption}是本会话里熟悉的表情包{alias_phrase}。"
            )
        elif aliases:
            lines.append(f"- 当前{kind_label}{shown_caption}{alias_phrase}。")
        else:
            lines.append(f"- 当前{kind_label}{shown_caption}本会话还没有固定称呼。")

    if has_sticker:
        lines.append(
            "- 表情包是群友表达心情、意境和态度的日常用语，跟图片实际内容关系不大。"
            "结合前后文理解这次在说什么，自然接它的情绪、梗或话题；熟悉感体现在接得懂、接得顺。"
        )
    if has_repeated:
        lines.extend(
            [
                "- 可以围绕这次的意思自然地熟人式吐槽。",
                "- 累计出现次数只帮助认图，不表示今天刷屏，也不代表发送者在重复同一种心情。",
            ]
        )
    if alias_learning:
        lines.extend(
            [
                "- 只有在当前消息明确给图起名或纠正称呼时，才写隐藏动作：加称呼 [记图名:大肥鱼]，替换旧称呼 [改图名:大肥鱼]，删除称呼 [删图名:大肥鱼]。多图时用 [记图名:2:大肥鱼] 指定第几张。",
                "- “这个表情包是大肥鱼”“以后叫它大肥鱼”算明确命名；同一会话刚发出的上一张图被明确点名时也可学习。普通玩梗、疑问、否定、“大肥鱼是谁”、说别的图或别的机器人，都不算；不要盲从含糊说法。",
                "- 已有称呼时自然使用即可，不必每轮都提；隐藏动作会被系统移除，不要在回复正文里解释或复述这些标签。",
            ]
        )
    lines.extend(
        [
            "- 不要复述本段系统观察，也不要声称自己能看见未提供的画面细节。",
        ]
    )
    return "\n".join(lines)
