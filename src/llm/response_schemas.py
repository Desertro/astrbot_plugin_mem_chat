"""
结构化响应模式定义 - 用于 LLM 输出的 JSON 格式

示例:
```json
{
  "response": {
    "text": "直接回复用户的文本内容",
    "emotion": "[happy]",
    "tts": { "enabled": true, "voice": "default", "speed": 1.0 },
    "mentions": []
  },
  "control": {
    "decision": "reply_only",
    "mute": {
      "target_qq": "12345678",
      "duration": 60
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

        resp = response_dict.get("response")
        if isinstance(resp, dict):
            mentions = resp.get("mentions")
            if mentions is not None and not isinstance(mentions, list):
                resp["mentions"] = [mentions]

            segs = resp.get("segments")
            if isinstance(segs, list):
                for seg in segs:
                    if isinstance(seg, dict):
                        s_mentions = seg.get("mentions")
                        if s_mentions is not None and not isinstance(s_mentions, list):
                            seg["mentions"] = [s_mentions]

        response_dict.pop("plan", None)
        response_dict.pop("goal", None)
        response_dict.pop("state_machine", None)

        return response_dict

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

        if not isinstance(structured, dict):
            return None
        if not StructuredResponse.validate_response(structured):
            return None
        return StructuredResponse.normalize_response_dict(structured)

    @staticmethod
    def extract_json_from_text(text: str) -> dict[str, Any] | None:
        import re

        text = text.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return StructuredResponse.normalize_response_dict(parsed)
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
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            return StructuredResponse.normalize_response_dict(parsed)
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
                        parsed = json.loads(potential_json)
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
                    parsed = json.loads(match.group(1))
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
