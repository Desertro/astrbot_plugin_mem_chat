"""
结构化响应模式定义 - 用于 LLM 输出的 JSON 格式

示例:
```json
{
  "response": {
    "text": "直接回复用户的文本内容",
    "emotion": "[happy]",
    "tts": { "enabled": true, "voice": "default", "speed": 1.0 },
    "mentions": [],
    "pokes": [{ "qq": "123456" }]
  },
  "control": {
    "decision": "reply_only",
    "mute": {
      "target_qq": "12345678",
      "duration": 60
    },
    "recall": {
      "message_id": "1234567890",
      "quote_index": 1
    }
  }
}
```
"""

import json
from datetime import datetime
from typing import Any


class StructuredResponse:
    """
    结构化响应基类
    定义 JSON 输出格式，包含普通回复、表情、TTS 等
    """

    @staticmethod
    def create_response(
        text: str,
        emotion: str | None = None,
        tts_enabled: bool = False,
        tts_voice: str = "default",
        tts_speed: float = 1.0,
        mentions: list[dict] | None = None,
        at_position: str | None = None,
        pokes: list[dict] | None = None,
    ) -> dict[str, Any]:
        response = {
            "text": text,
            "tts": {"enabled": tts_enabled, "voice": tts_voice, "speed": tts_speed},
            "mentions": mentions or [],
        }

        if emotion:
            response["emotion"] = emotion
        if at_position:
            response["at_position"] = at_position
        if pokes:
            response["pokes"] = pokes

        return response

    @staticmethod
    def create_full_response(
        response: dict[str, Any],
        control: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        full_response = {
            "response": response,
            "control": control or {"decision": "reply_only"},
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
        }

        return full_response

    @staticmethod
    def validate_response(response_dict: dict[str, Any]) -> bool:
        try:
            if "response" not in response_dict or "control" not in response_dict:
                return False

            response = response_dict["response"]
            if not isinstance(response, dict):
                return False

            control = response_dict["control"]
            if not isinstance(control, dict):
                return False
            decision = control.get("decision")
            if not isinstance(decision, str):
                return False
            decision_norm = decision.strip().lower()
            if decision_norm not in ("no_reply", "reply_only"):
                return False

            if decision_norm in ("reply_only",):
                has_text = isinstance(response.get("text"), str) and bool(
                    response.get("text").strip()
                )
                has_segments = False
                segs = response.get("segments")
                if isinstance(segs, list) and segs:
                    for seg in segs:
                        if (
                            isinstance(seg, dict)
                            and isinstance(seg.get("text"), str)
                            and seg.get("text").strip()
                        ):
                            has_segments = True
                            break
                if not has_text and not has_segments:
                    return False

            if "tts" in response:
                tts = response["tts"]
                if not isinstance(tts.get("enabled", False), bool):
                    return False

            return True

        except (KeyError, TypeError):
            return False

    @staticmethod
    def normalize_response_dict(response_dict: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(response_dict, dict):
            return response_dict

        # 兼容旧/变体协议：content -> response
        if "response" not in response_dict and isinstance(response_dict.get("content"), dict):
            response_dict["response"] = response_dict.get("content") or {}
        response_dict.pop("content", None)

        # 兼容模型偶尔输出的简写：{"decision":"no_reply"}。
        if "control" not in response_dict:
            root_decision = response_dict.get("decision")
            if isinstance(root_decision, str):
                response_dict["control"] = {"decision": root_decision}
        response_dict.pop("decision", None)

        # 兼容 decision 变体
        ctrl = response_dict.get("control")
        if isinstance(ctrl, dict):
            dec = ctrl.get("decision")
            if isinstance(dec, str):
                dec_norm = dec.strip().lower()
                if dec_norm in ("reply", "respond", "chat"):
                    ctrl["decision"] = "reply_only"
                elif dec_norm in ("silent", "ignore"):
                    ctrl["decision"] = "no_reply"
                if dec_norm in ("no_reply", "silent", "ignore") and "response" not in response_dict:
                    response_dict["response"] = {"text": ""}

        allowed_roots = {
            "control",
            "response",
            "mood",
            "affinity",
            "tool_call",
            "impression_delta",
            "session_impression_summary",
            "memory_importance",
            "tools",
            "tool_calls",
        }

        # 兼容 DeepSeek/OAI 格式的 tools 字段 → tool_calls
        if "tools" in response_dict and "tool_calls" not in response_dict:
            raw_tools = response_dict.get("tools")
            if isinstance(raw_tools, list):
                tc_list = []
                for item in raw_tools:
                    if isinstance(item, dict):
                        tc_list.append(item)
                if tc_list:
                    response_dict["tool_calls"] = tc_list
        for key in list(response_dict.keys()):
            if "." not in key:
                continue
            parts = [p for p in key.split(".") if p]
            if len(parts) < 2:
                continue
            root = parts[0]
            if root not in allowed_roots:
                continue
            container = response_dict.get(root)
            if not isinstance(container, dict):
                container = {}
                response_dict[root] = container
            cur = container
            for part in parts[1:-1]:
                nxt = cur.get(part)
                if not isinstance(nxt, dict):
                    nxt = {}
                    cur[part] = nxt
                cur = nxt
            cur[parts[-1]] = response_dict[key]
            response_dict.pop(key, None)

        resp = response_dict.get("response")
        if isinstance(resp, dict):
            mentions = resp.get("mentions")
            if mentions is not None and not isinstance(mentions, list):
                resp["mentions"] = [mentions]
            pokes = resp.get("pokes")
            if pokes is None:
                pokes = resp.get("poke")
            if pokes is not None and not isinstance(pokes, list):
                resp["pokes"] = [pokes]

            segs = resp.get("segments")
            if isinstance(segs, list):
                for seg in segs:
                    if isinstance(seg, dict):
                        s_mentions = seg.get("mentions")
                        if s_mentions is not None and not isinstance(s_mentions, list):
                            seg["mentions"] = [s_mentions]
                        s_pokes = seg.get("pokes")
                        if s_pokes is None:
                            s_pokes = seg.get("poke")
                        if s_pokes is not None and not isinstance(s_pokes, list):
                            seg["pokes"] = [s_pokes]

        response_dict.pop("plan", None)
        response_dict.pop("goal", None)
        response_dict.pop("state_machine", None)

        return response_dict

    @staticmethod
    def _sanitize_plain_text_reply(text: str) -> str:
        import re

        s = str(text or "").strip()
        if not s:
            return ""

        s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*```$", "", s).strip()
        s = re.sub(r"(?im)^\s*(json|output|response)\s*[:：]\s*", "", s).strip()

        lines = [ln.strip() for ln in s.splitlines()]
        lines = [ln for ln in lines if ln]
        if not lines:
            return ""

        # If the model prepends short meta lines like "下面是 JSON" or "答：", keep the last substantial line.
        substantial = [ln for ln in lines if len(ln) >= 4]
        if substantial:
            s = substantial[-1]
        else:
            s = lines[-1]

        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _looks_like_no_reply_text(text: str) -> bool:
        s = str(text or "").strip().lower()
        if not s:
            return False
        markers = (
            "no_reply",
            "[do_not_reply]",
            "无需回复",
            "不需要回复",
            "不用回复",
            "不必回复",
            "不回复",
            "保持安静",
            "不介入",
        )
        return any(m in s for m in markers)

    @staticmethod
    def _fallback_payload_from_text(text: str) -> dict[str, Any] | None:
        clean = StructuredResponse._sanitize_plain_text_reply(text)
        if StructuredResponse._looks_like_no_reply_text(clean):
            return {
                "control": {"decision": "no_reply"},
                "response": {"text": ""},
                "mood": {
                    "text": "本轮输出格式异常，先保持观察。",
                    "state": "中性",
                    "ttl_sec": 0,
                    "intensity": 0.0,
                },
                "affinity": {"delta": 0, "reason": "格式修复回退。"},
            }
        if not clean:
            return None
        return {
            "control": {"decision": "reply_only"},
            "response": {"text": clean},
            "mood": {
                "text": "本轮输出格式异常，已本地整理成回复。",
                "state": "中性",
                "ttl_sec": 0,
                "intensity": 0.0,
            },
            "affinity": {"delta": 0, "reason": "格式修复回退。"},
        }

    @staticmethod
    def _looks_like_structured_meta_text(text: str) -> bool:
        import re

        s = str(text or "").strip()
        if not s:
            return False

        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            return True

        meta_patterns = (
            r'"\s*confidence\s*"\s*:\s*[0-9.]+',
            r'"\s*target_qq\s*"\s*:',
            r'"\s*display_name\s*"\s*:',
            r'"\s*summary\s*"\s*:',
            r'"\s*fact\s*"\s*:',
            r'"\s*impression_delta\s*"\s*:',
            r'"\s*session_impression_summary\s*"\s*:',
            r'"\s*memory_importance\s*"\s*:',
            r'"\s*control\s*"\s*:',
            r'"\s*response\s*"\s*:',
            r'"\s*mood\s*"\s*:',
            r'"\s*affinity\s*"\s*:',
        )
        return any(re.search(p, s, flags=re.IGNORECASE) for p in meta_patterns)

    @staticmethod
    def _fallback_payload_from_dict(response_dict: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(response_dict, dict):
            return None

        normalized = StructuredResponse.normalize_response_dict(dict(response_dict))
        if StructuredResponse.validate_response(normalized):
            return normalized

        response = normalized.get("response")
        if not isinstance(response, dict):
            response = {}

        text_candidates = [
            response.get("text"),
            normalized.get("text"),
            normalized.get("reply"),
            normalized.get("answer"),
            normalized.get("content"),
            normalized.get("message"),
        ]
        text = ""
        for item in text_candidates:
            if isinstance(item, str) and item.strip():
                text = item
                break

        segments = response.get("segments")
        if not text and isinstance(segments, list):
            seg_texts = []
            for seg in segments:
                if isinstance(seg, dict) and isinstance(seg.get("text"), str) and seg.get("text").strip():
                    seg_texts.append(seg.get("text").strip())
            if seg_texts:
                text = "\n".join(seg_texts)

        if not text:
            return None

        fallback = StructuredResponse._fallback_payload_from_text(text)
        if not fallback:
            return None

        # Preserve already-emitted structured fields when they exist.
        if isinstance(response, dict):
            merged_resp = dict(response)
            merged_resp.setdefault("text", fallback["response"].get("text", ""))
            fallback["response"] = merged_resp
        for key in ("mood", "affinity", "tool_calls", "tools"):
            val = normalized.get(key)
            if val is not None:
                fallback[key] = val
        return fallback

    @staticmethod
    def parse_validated(obj: Any) -> dict[str, Any] | None:
        structured: dict[str, Any] | None = None
        if isinstance(obj, dict):
            structured = StructuredResponse.normalize_response_dict(obj)
        elif isinstance(obj, str) and obj.strip():
            try:
                structured = StructuredResponse.extract_json_from_text(obj)
            except Exception:
                structured = None

        if isinstance(structured, dict):
            structured = StructuredResponse.normalize_response_dict(structured)
            if StructuredResponse.validate_response(structured):
                return structured
            repaired = StructuredResponse._fallback_payload_from_dict(structured)
            if isinstance(repaired, dict):
                repaired = StructuredResponse.normalize_response_dict(repaired)
                if StructuredResponse.validate_response(repaired):
                    return repaired

        if isinstance(obj, str) and obj.strip():
            if StructuredResponse._looks_like_structured_meta_text(obj):
                return None
            repaired = StructuredResponse._fallback_payload_from_text(obj)
            if isinstance(repaired, dict):
                repaired = StructuredResponse.normalize_response_dict(repaired)
                if StructuredResponse.validate_response(repaired):
                    return repaired

        return None

    @staticmethod
    def _repair_unescaped_quotes_in_json_strings(text: str) -> str:
        """Escape likely accidental bare quotes inside JSON string values."""
        s = str(text or "")
        out: list[str] = []
        in_string = False
        escaped = False
        i = 0
        n = len(s)
        while i < n:
            ch = s[i]
            if not in_string:
                out.append(ch)
                if ch == '"':
                    in_string = True
                    escaped = False
                i += 1
                continue

            if escaped:
                out.append(ch)
                escaped = False
                i += 1
                continue

            if ch == "\\":
                out.append(ch)
                escaped = True
                i += 1
                continue

            if ch == '"':
                j = i + 1
                while j < n and s[j] in " \t\r\n":
                    j += 1
                nxt = s[j] if j < n else ""
                # A real JSON string terminator is followed by a key colon or a
                # value/list/object delimiter. Otherwise this is probably a
                # human-style quote inside the text, e.g. "这个字母的"深意"哦".
                if nxt in {":", ",", "}", "]", ""}:
                    out.append(ch)
                    in_string = False
                else:
                    out.append('\\"')
                i += 1
                continue

            out.append(ch)
            i += 1
        return "".join(out)

    @staticmethod
    def _loads_json_with_repair(text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            repaired = StructuredResponse._repair_unescaped_quotes_in_json_strings(text)
            if repaired != text:
                return json.loads(repaired)
            raise

    @staticmethod
    def extract_json_from_text(text: str) -> dict[str, Any] | None:
        import re

        text = text.strip()

        try:
            parsed = StructuredResponse._loads_json_with_repair(text)
            if isinstance(parsed, dict):
                return StructuredResponse.normalize_response_dict(parsed)
            if isinstance(parsed, list):
                for item in reversed(parsed):
                    if isinstance(item, dict):
                        return StructuredResponse.normalize_response_dict(item)
        except json.JSONDecodeError:
            pass

        code_patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
        ]

        for pattern in code_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in reversed(matches):
                    content = match.strip()
                    try:
                        parsed = StructuredResponse._loads_json_with_repair(content)
                        if isinstance(parsed, dict):
                            return StructuredResponse.normalize_response_dict(parsed)
                        if isinstance(parsed, list):
                            for item in reversed(parsed):
                                if isinstance(item, dict):
                                    return StructuredResponse.normalize_response_dict(item)
                    except json.JSONDecodeError:
                        continue

        try:
            start_idx = text.rfind("{")
            end_idx = text.rfind("}")

            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                content_to_end = text[: end_idx + 1]
                start_positions = [
                    m.start() for m in re.finditer(r"\{", content_to_end)
                ]

                for pos in reversed(start_positions):
                    try:
                        potential_json = content_to_end[pos:]
                        parsed = StructuredResponse._loads_json_with_repair(potential_json)
                        if isinstance(parsed, dict):
                            if "response" in parsed or "control" in parsed:
                                return StructuredResponse.normalize_response_dict(
                                    parsed
                                )
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

        try:
            match = re.search(r"(\{[\s\S]*\})", text)
            if match:
                try:
                    parsed = StructuredResponse._loads_json_with_repair(match.group(1))
                    if isinstance(parsed, dict):
                        return StructuredResponse.normalize_response_dict(parsed)
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass

        return None


class ErrorTemplates:
    @staticmethod
    def no_images_found(query: str) -> str:
        return f"没有找到关于「{query}」的图片，可能是关键词不太准确或者平台限制。"

    @staticmethod
    def image_404(url: str) -> str:
        return "这个图片链接好像失效了，可能是原图已被删除。"

    @staticmethod
    def execution_error(details: str) -> str:
        return f"处理过程中出了点问题：{details}"

    @staticmethod
    def network_error() -> str:
        return "网络连接有点问题，请稍后再试。"

    @staticmethod
    def api_limit_exceeded() -> str:
        return "API调用次数达到限制，请稍后再试。"


class EmotionFormatter:
    FORMATS = {
        "brackets": "[{emotion}]",
        "parens": "({emotion})",
        "tags": "&&{emotion}&&",
        "none": "{emotion}",
    }

    @staticmethod
    def format_emotion(emotion: str, style: str = "brackets") -> str:
        if not emotion:
            return ""

        if emotion.startswith("[") and emotion.endswith("]"):
            return emotion
        if emotion.startswith("(") and emotion.endswith(")"):
            return emotion
        if emotion.startswith("&&") and emotion.endswith("&&"):
            return emotion

        fmt = EmotionFormatter.FORMATS.get(style, EmotionFormatter.FORMATS["brackets"])
        return fmt.format(emotion=emotion)

    @staticmethod
    def extract_emotion(text: str) -> list[str]:
        import re

        patterns = [
            r"\[([^\[\]]+)\]",
            r"\(([^\(\)]+)\)",
            r"&&([^&]+)&&",
        ]

        emotions = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            emotions.extend(matches)

        return emotions

    @staticmethod
    def remove_emotion_tags(text: str) -> str:
        import re

        patterns = [
            r"\[[^\[\]]+\]",
            r"\([^\(\)]+\)",
            r"&&[^&]+&&",
        ]

        for pattern in patterns:
            text = re.sub(pattern, "", text)

        text = re.sub(r"\s+", " ", text).strip()
        return text
