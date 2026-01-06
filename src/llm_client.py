import asyncio
import copy
import html
import inspect
import json
import logging
import os
import random
import re
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

import aiohttp

from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.message_components import Image, Record, Reply
from astrbot.core import file_token_service
from astrbot.core.agent.tool import ToolSet
from astrbot.core.star.session_llm_manager import SessionServiceManager

from .llm.response_schemas import StructuredResponse
from .llm_client_parts.custom_llm import call_custom_llm, call_custom_llm_google
from .llm_client_parts.image_service import ImageService
from .llm_client_parts.tts_service import TtsService
from .models import DummyEvent

logger = logging.getLogger("astrbot")


class LLMClient:
    def __init__(self, plugin):
        self.plugin = plugin
        self._image_service = ImageService(self._get_conf)
        self._tts_service = TtsService()
        self._user_daily_images = self._image_service._user_daily_images
        self._tts_replacement_audio_cache = self._tts_service._replacement_audio_cache

    def _get_conf(self, key: str, default):
        cfg = getattr(self.plugin, "config", {})
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return default

    def _tool_value_to_text(self, val) -> str:
        if val is None:
            return ""
        if isinstance(val, MessageChain):
            try:
                return (val.get_plain_text() or "").strip()
            except Exception:
                return str(val)
        if isinstance(val, str | int | float | bool):
            return str(val)
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)

    async def _execute_tool_handler(self, handler, event: AstrMessageEvent, args: dict) -> str:
        res = handler(event, **(args or {}))
        if inspect.isasyncgen(res):
            parts: list[str] = []
            async for item in res:
                t = self._tool_value_to_text(item)
                if t:
                    parts.append(t)
            return "\n".join(parts).strip()
        if inspect.isawaitable(res):
            res = await res
        return self._tool_value_to_text(res)

    def _get_task_timeout_sec(self, task_type: str) -> float:
        mapping = {
            "at": "task_timeout_sec_at",
            "focus": "task_timeout_sec_focus",
            "read_air": "task_timeout_sec_read_air",
        }
        k = mapping.get(str(task_type or "").strip(), "task_timeout_sec_at")
        try:
            v = float(self._get_conf(k, 30))
            if v <= 0:
                return 0.0
            return v
        except Exception:
            return 30.0

    def _normalize_image_key(self, url: str) -> str:
        return self._image_service.normalize_image_key(url)

    def _pick_meme_image_file(self, emotion: str) -> str | None:
        return self._image_service.pick_meme_image_file(emotion)

    def _get_user_daily_set(self, event: AstrMessageEvent | None):
        return self._image_service.get_user_daily_set(event)

    def _tts_text_is_speakable(self, text: str) -> bool:
        return self._tts_service.text_is_speakable(text)

    def _tts_cleanup_text(self, text: str) -> str:
        return self._tts_service.cleanup_text(text)

    def _split_tts_text(self, text: str, max_chars: int = 120) -> list[str]:
        return self._tts_service.split_text(text, max_chars=max_chars)

    def _get_tts_replacement_audio_path(self, token: str) -> str | None:
        return self._tts_service.get_replacement_audio_path(token)

    def _build_tts_parts(self, text: str) -> list[tuple[str, str]]:
        return self._tts_service.build_parts(text)

    def _try_concat_wavs(self, wav_paths: list[str]) -> str | None:
        return self._tts_service.try_concat_wavs(wav_paths)

    async def _prepare_llm_image_contents(self, image_urls: list[str]) -> list[dict]:
        return await self._image_service.prepare_llm_image_contents(image_urls)

    async def _call_custom_llm(
        self,
        base_url,
        api_key,
        model,
        system_prompt,
        prompt,
        contexts,
        tools=None,
        image_urls=None,
        temperature=None,
        fmt=None,
        tag: str = "",
    ):
        return await call_custom_llm(
            self,
            base_url,
            api_key,
            model,
            system_prompt,
            prompt,
            contexts,
            tools=tools,
            image_urls=image_urls,
            temperature=temperature,
            fmt=fmt,
            tag=tag,
        )

    async def _call_custom_llm_google(
        self,
        base_url,
        api_key,
        model,
        system_prompt,
        prompt,
        contexts,
        tools=None,
        image_urls=None,
        temperature=None,
        tag: str = "",
    ):
        return await call_custom_llm_google(
            self,
            base_url,
            api_key,
            model,
            system_prompt,
            prompt,
            contexts,
            tools=tools,
            image_urls=image_urls,
            temperature=temperature,
            tag=tag,
        )

    def _get_instruction_prompts(
        self, custom_instruction: str = "", tool_instructions: str = ""
    ) -> str:
        default_instructions = (
            "【回复要求】\n"
            "1. 请根据角色设定、【记忆库上下文】、【人物印象】以及【最近对话】进行简洁回复。\n"
            "2. 这是一个群聊场景，除非有人指名你，不然不是在说你，恶意的言语没at你都不是说你，无需回怼。\n"
            "3. 请注意回复的长度，尽量简短有力，除非需要详细解释。\n"
            "4. 你必须只输出一个结构化 JSON 对象，禁止输出任何自然语言、解释、Markdown 代码块或前后缀。\n"
            "4.1 必须包含字段：control.decision（no_reply/reply_only）与 response.text。\n"
            "4.2 当 control.decision 为 reply_only：response.text 必须为非空字符串。\n"
            "4.3 response.text 中严禁包含思考过程/推理/分析（例如“思考：”“分析：”“<think>...</think>”）。\n"
            "5. 引用：上下文中会用 [REF:序号] 标记可引用的候选；如需引用，请在 JSON 中使用 response.quote_index 指定序号（从 1 开始）。\n"
            "6. 艾特：如需艾特某人，请在 response.text 中使用 `<at:QQ号>` 或 `<at:序号>`（序号见可艾特列表）。系统会自动处理，无需在 response.mentions 中重复填写。\n"
            '6.1 表情：如需发送表情包，将表情标签放在 response.emotion（例如 "[happy]"），不要把 "[happy]" 写进 response.text。\n'
            "6.2 表情纠错：response.text 中严禁出现任何形如 [xxx]、(xxx)、&&xxx&& 的表情标签；如需要，必须改填到 response.emotion，并从 response.text 删除。\n"
            "7. 语音回复：使用 response.tts.enabled 控制是否启用语音；不要在文本中写 `TTS: yes/no`。\n"
            "8. “艾特、引用以及表情功能”和“语音朗读功能”互斥：二选一。\n"
            "8.1 若 response.tts.enabled 为 true：禁止引用/艾特/表情标签（response.quote_index / response.mentions / response.emotion 必须为空或不填），且 response.text 禁止包含任何 URL；否则请将 response.tts.enabled 设为 false。\n"
            "9. 图片发送：如果需要发送图片（如搜图结果），请将图片 URL 放入 response.images 数组（字符串数组）中。系统会自动处理。不要在 response.text 中包含 URL。\n"
        )
        if tool_instructions:
            default_instructions += f"10. 工具调用协议：\n{tool_instructions}\n"

        try:
            meme_plugin = None
            if hasattr(self.plugin.context, "get_registered_star"):
                meme_plugin = self.plugin.context.get_registered_star("meme_manager")

            mapping = None
            if meme_plugin and meme_plugin.activated and meme_plugin.star_cls:
                sender = meme_plugin.star_cls
                mapping = getattr(sender, "category_mapping", None)
                if not mapping and hasattr(sender, "category_manager"):
                    try:
                        mapping = sender.category_manager.get_descriptions()
                    except Exception:
                        pass

            # 即使没有 meme_manager，也注入基础的表情包提示
            if not mapping:
                memes_dir = os.path.join(os.getcwd(), "data", "memes_data", "memes")
                if os.path.exists(memes_dir):
                    dirs = [
                        d
                        for d in os.listdir(memes_dir)
                        if os.path.isdir(os.path.join(memes_dir, d))
                    ]
                    if dirs:
                        mapping = {d: d for d in dirs}

            if mapping:
                meme_prompt = "\n【表情包使用指南】\n你可以使用以下标签来发送表情包（会自动转换为图片）：\n"
                categories = []
                for key, desc in mapping.items():
                    categories.append(f"- [{key}]: {desc}")

                meme_prompt += "\n".join(categories)
                meme_prompt += '\n\n使用说明：\n1. 若要发送表情包，请把标签写进 JSON 的 response.emotion，例如："emotion": "[happy]"。\n2. 不要把 "[happy]" 写进 response.text。\n3. 请根据语境选择合适的表情，不要滥用。\n'
                default_instructions += meme_prompt
        except Exception as e:
            logger.warning(f"[mem_chat] Failed to inject meme_manager prompt: {e}")

        extra = self._get_conf("extra_system_prompt", "")
        if extra:
            default_instructions += f"\n{extra}\n"

        return default_instructions

    def _get_bool_conf(self, name: str, default: bool = False) -> bool:
        if name in ("log_llm_prompt",):
            return True
        try:
            val = self._get_conf(name, default)
        except Exception:
            val = default
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            v = val.strip().lower()
            return v in ("1", "true", "yes", "y", "on")
        if isinstance(val, int | float):
            try:
                return bool(int(val))
            except Exception:
                return bool(val)
        return bool(val)

    async def _fetch_image_from_url(self, url: str) -> str | None:
        path = urlparse(url).path
        if any(
            ext in path.lower()
            for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]
        ):
            return url
        try:
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        content_type = resp.headers.get("Content-Type", "")
                        if "image" in content_type:
                            return url
                        text = await resp.text()
                        match = re.search(
                            r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']',
                            text,
                        )
                        if match:
                            return match.group(1)
                        match = re.search(
                            r'<meta\s+name=["\']twitter:image["\']\s+content=["\']([^"\']+)["\']',
                            text,
                        )
                        if match:
                            return match.group(1)
        except Exception as e:
            logger.warning(f"[mem_chat] Error fetching image from URL {url}: {e}")
            pass
        return None

    async def _resolve_pixiv_artwork_image(self, artwork_url: str) -> str | None:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Referer": "https://www.pixiv.net/",
            }
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.get(
                    artwork_url, headers=headers, timeout=15
                ) as resp:
                    if resp.status != 200:
                        return None
                    text = await resp.text(errors="ignore")
            return self._extract_pixiv_image_from_html(text, artwork_url)
        except Exception as e:
            logger.warning(f"[mem_chat] Failed to resolve Pixiv artwork image: {e}")
        return None

    def _extract_pixiv_image_from_html(
        self, html_text: str, artwork_url: str
    ) -> str | None:
        try:
            text = html_text or ""
            m = re.search(
                r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']',
                text,
            )
            if m:
                return m.group(1)
            m = re.search(
                r'<meta\s+name=["\']twitter:image["\']\s+content=["\']([^"\']+)["\']',
                text,
            )
            if m:
                return m.group(1)

            m = re.search(
                r'<meta[^>]+id=["\']meta-preload-data["\'][^>]+content=["\']([^"\']+)["\']',
                text,
            )
            if not m:
                return None

            content = html.unescape(m.group(1))
            try:
                preload = json.loads(content)
            except Exception:
                return None

            pid_match = re.search(r"pixiv\.net/(?:en/)?artworks/(\d+)", artwork_url)
            pid = pid_match.group(1) if pid_match else None
            illust_block = preload.get("illust") if isinstance(preload, dict) else None
            if not isinstance(illust_block, dict):
                return None

            if pid and isinstance(illust_block.get(pid), dict):
                ill = illust_block.get(pid)
            else:
                ill = next(
                    (v for v in illust_block.values() if isinstance(v, dict)), None
                )
            if not isinstance(ill, dict):
                return None

            urls = ill.get("urls")
            if isinstance(urls, dict):
                for k in ("original", "regular", "large", "small"):
                    u = urls.get(k)
                    if isinstance(u, str) and u.startswith("http"):
                        return u
        except Exception:
            return None
        return None

    async def _resolve_social_media_image(self, url: str) -> str | None:
        pixiv_match = re.search(r"pixiv\.net/(?:en/)?artworks/(\d+)", url)
        if pixiv_match:
            return await self._resolve_pixiv_artwork_image(url)
        if "twitter.com" in url or "x.com" in url:
            if "/status/" in url:
                try:
                    api_url = url.replace("twitter.com", "fxtwitter.com").replace(
                        "x.com", "fxtwitter.com"
                    )
                    headers = {
                        "User-Agent": "Mozilla/5.0 (compatible; Discordbot/2.0; +https://discordapp.com)"
                    }
                    async with aiohttp.ClientSession(trust_env=True) as session:
                        async with session.get(
                            api_url, headers=headers, timeout=10
                        ) as resp:
                            if resp.status == 200:
                                text = await resp.text()
                                m = re.search(
                                    r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']',
                                    text,
                                )
                                if m:
                                    img_url = m.group(1)
                                    logger.info(
                                        f"[mem_chat] Resolved X/Twitter image via fxtwitter: {img_url}"
                                    )
                                    return img_url
                except Exception as e:
                    logger.warning(f"[mem_chat] Failed to resolve X/Twitter image: {e}")
        return None

    def _strip_search_listing_text(self, text: str) -> str:
        try:
            lines = text.splitlines()
            cleaned = []
            in_kb_block = False
            for ln in lines:
                s = ln.strip()
                if s.lower().startswith("here are the images/artworks found"):
                    continue
                if s.startswith("---") and (
                    "Pixiv API Results" in s or "Web Search Results" in s
                ):
                    continue
                if s.startswith("System:") and "Please select the best URL" in s:
                    continue
                if re.match(r"^\[\d+\]\s.*\(URL:\s*[^)]+\)\s*$", s):
                    continue
                if s.startswith("###PLANPAD_START###"):
                    in_kb_block = True
                    continue
                if s.startswith("###PLANPAD_END###"):
                    in_kb_block = False
                    continue
                if in_kb_block:
                    continue
                if s.startswith("用户的原始问题是："):
                    continue
                if re.match(r"^[!！]+$", s):
                    continue
                cleaned.append(ln)
            return "\n".join(cleaned)
        except Exception:
            return text

    def _sanitize_final_reply(self, text: str) -> str:
        try:
            s = text or ""
            try:

                def _extract_text_from_json_obj(obj) -> str:
                    extracted = ""
                    if isinstance(obj, dict):
                        for k in [
                            "final",
                            "answer",
                            "reply",
                            "text",
                            "content",
                            "message",
                            "output",
                        ]:
                            v = obj.get(k)
                            if isinstance(v, str) and v.strip():
                                extracted = v.strip()
                                break
                        if not extracted:
                            v = obj.get("data")
                            if isinstance(v, str) and v.strip():
                                extracted = v.strip()
                    elif isinstance(obj, list):
                        for it in obj:
                            if isinstance(it, dict):
                                v = (
                                    it.get("text")
                                    or it.get("content")
                                    or it.get("message")
                                    or it.get("answer")
                                )
                                if isinstance(v, str) and v.strip():
                                    extracted = (extracted + "\n" + v.strip()).strip()
                            elif isinstance(it, str) and it.strip():
                                extracted = (extracted + "\n" + it.strip()).strip()
                    return extracted

                def _replace_fenced_json_blocks(src: str) -> str:
                    def _repl(m: re.Match) -> str:
                        inner = (m.group(1) or "").strip()
                        if not inner:
                            return ""
                        try:
                            obj = json.loads(inner)
                        except Exception:
                            return ""
                        extracted = _extract_text_from_json_obj(obj)
                        return extracted

                    src = re.sub(
                        r"```(?:toolcall-json|json)\s*([\s\S]*?)```", _repl, src
                    )
                    return src

                s = _replace_fenced_json_blocks(s).strip()
            except Exception:
                pass
            try:
                s_strip = (s or "").strip()
                if (s_strip.startswith("{") and s_strip.endswith("}")) or (
                    s_strip.startswith("[") and s_strip.endswith("]")
                ):
                    obj = json.loads(s_strip)
                    extracted = _extract_text_from_json_obj(obj)
                    if extracted:
                        s = extracted
                    else:
                        s = ""
            except Exception:
                pass
            try:
                s = re.sub(r"(?is)<think>.*?</think>", "", s).strip()
                s = re.sub(
                    r"(?is)```(?:think|thinking|analysis)\\s*[\\s\\S]*?```", "", s
                ).strip()
                s = re.sub(
                    r"(?is)^\s*(analysis|reasoning|thoughts)\s*:\s*.*?(?=\n\s*\n|$)",
                    "",
                    s,
                ).strip()
                s = re.sub(
                    r"(?is)^\s*(?:思考过程|思考|推理|分析)\s*[:：]\s*.*?(?=\n\s*\n|$)",
                    "",
                    s,
                ).strip()
                s = re.sub(
                    r"(?s)【(?:思考过程|思考|推理|分析)】[\s\S]*?(?=\n\s*\n|$)", "", s
                ).strip()
            except Exception:
                pass
            s = re.sub(r"(?im)^\s*USE_TOOLS\s*:\s*(yes|no|true|false)\s*$", "", s)
            s = re.sub(r"(?im)^\s*GOAL\s*:\s*.*$", "", s)
            s = re.sub(r"(?im)^\s*(?:OBJECTIVE|目的)\s*[:：]\s*.*$", "", s)
            s = re.sub(r"(?im)^\s*SUBJECT\s*:\s*.*$", "", s)
            s = re.sub(r"(?im)^\s*SEARCH_QUERY\s*:\s*.*$", "", s)
            s = re.sub(r"(?i)\s*\bTTS\s*[:：]\s*(yes|no)\b", "", s)
            s = re.sub(r"[ \t]{2,}", " ", s)
            s = re.sub(r"[ \t]+\n", "\n", s)
            s = re.sub(r"(?im)^\s*PLAN\s*:\s*(?:\n(?:\s*-\s.*\n?)*)?", "", s)
            s = re.sub(r"(?s)【(?:计划|PLAN)】[\s\S]*?(?=\n\s*\n|$)", "", s)
            s = re.sub(r"(?s)【决策】[\s\S]*?(?=\n\s*\n|$)", "", s)
            s = re.sub(r"(?im)^\s*【(?:专注开始|心跳开始)】.*$", "", s)
            s = re.sub(
                r"(?im)^\s*(?:-\s*)?步骤\s*[0-9一二三四五六七八九十]+\s*[:：].*$", "", s
            )
            s = re.sub(r"(?im)^\s*(?:-\s*)?step\s*\d+\s*[:：].*$", "", s)
            s = re.sub(r"(?s)【任务目标】[\s\S]*?(?=\n\s*\n|$)", "", s)
            s = re.sub(r"(?s)【RELAY】[\s\S]*?(?=【OBSERVATIONS】|$)", "", s)
            s = re.sub(r"(?s)【OBSERVATIONS】[\s\S]*$", "", s)
            s = re.sub(r"(?im)^\s*SUBTASKS\s*:\s*$", "", s)
            s = re.sub(r"(?im)^\s*ALIASES\s*:\s*$", "", s)
            s = re.sub(r"(?im)^\s*WEB_SUMMARY\s*:\s*$", "", s)
            s = re.sub(r"(?im)^\s*CANDIDATE_URLS\s*:\s*$", "", s)
            try:
                s2 = (s or "").lstrip()
                m_cjk = re.search(r"[\u4e00-\u9fff]", s2)
                if m_cjk and m_cjk.start() >= 80:
                    prefix = s2[: m_cjk.start()]
                    if "[" not in prefix and "@" not in prefix:
                        asciiish = sum(1 for ch in prefix if ord(ch) < 128)
                        if len(prefix) > 0 and (asciiish / len(prefix)) >= 0.88:
                            s = s2[m_cjk.start() :].lstrip()
            except Exception:
                pass
            s = re.sub(r"\n{3,}", "\n\n", s)
            return s
        except Exception:
            return text

    async def _recover_images_from_search_results(
        self,
        llm_contexts: list,
        segment_text: str,
        chain: MessageChain,
        temp_files_to_clean: list,
        umo: str,
        topic_keywords: list[str],
    ) -> bool:
        try:
            for ctx in llm_contexts:
                if ctx.get("role") == "tool" and ctx.get("name") in ["search_images"]:
                    content = ctx.get("content", "")
                    potential_urls = []
                    try:
                        data = json.loads(content)

                        def extract_urls(obj):
                            urls = []
                            if isinstance(obj, dict):
                                for k, v in obj.items():
                                    if (
                                        k
                                        in [
                                            "image",
                                            "link",
                                            "thumbnail",
                                            "url",
                                            "content",
                                        ]
                                        and isinstance(v, str)
                                        and v.startswith("http")
                                    ):
                                        urls.append(v)
                                    else:
                                        urls.extend(extract_urls(v))
                            elif isinstance(obj, list):
                                for item in obj:
                                    urls.extend(extract_urls(item))
                            return urls

                        potential_urls.extend(extract_urls(data))
                    except Exception:
                        pass
                    regex_matches = re.findall(r"(https?://[^\s\"'<>\];]+)", content)
                    potential_urls.extend(regex_matches)
                    potential_urls = list(set(potential_urls))

                    valid_candidates = []
                    for url in potential_urls:
                        if any(
                            ext in url.lower()
                            for ext in [
                                ".jpg",
                                ".jpeg",
                                ".png",
                                ".gif",
                                ".webp",
                                ".bmp",
                            ]
                        ):
                            valid_candidates.append(url)
                        elif any(
                            d in url
                            for d in [
                                "encrypted-tbn0.gstatic.com",
                                "images.google.com",
                                "bing.net",
                                "googleusercontent.com",
                                "sinaimg.cn",
                                "slimg.cn",
                                "twimg.com",
                                "pbs.twimg.com",
                                "pixiv.net",
                                "x.com",
                                "twitter.com",
                            ]
                        ):
                            valid_candidates.append(url)
                        elif "format=jpg" in url or "format=png" in url:
                            valid_candidates.append(url)
                    uds = self._get_user_daily_set(
                        None
                    )  # Event is not available here easily, assume None or pass it
                    if uds:
                        valid_candidates = [
                            u
                            for u in valid_candidates
                            if self._normalize_image_key(u) not in uds
                        ]
                    random.shuffle(valid_candidates)
                    for img_url in valid_candidates[:5]:
                        if img_url in segment_text:
                            continue
                        logger.info(f"[mem_chat] Trying recovery download: {img_url}")
                        success, is_404 = await self._download_and_append_image(
                            img_url, chain, temp_files_to_clean, umo
                        )
                        if success:
                            # 只要成功下载了一张，就继续尝试下一张，直到达到上限或列表结束
                            pass
                    return any(
                        p.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"))
                        for p in temp_files_to_clean
                    )
        except Exception as e:
            logger.error(f"[mem_chat] Recover images failed: {e}")
        return False

    async def _download_and_append_image(
        self, url_to_download: str, chain_obj, temp_list: list, umo: str
    ) -> tuple[bool, bool]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        url_to_download = url_to_download.strip("`\"' ")

        # Pixiv i.pximg.net
        if "i.pximg.net" in url_to_download:
            cfg = self.plugin.context.get_config(umo=umo)
            prov_settings = cfg.get("provider_settings", {})
            refresh_token = prov_settings.get(
                "pixiv_refresh_token"
            ) or self.plugin.config.get("pixiv_refresh_token")
            if not refresh_token:
                logger.warning(
                    "[mem_chat] Pixiv Refresh Token is missing. Skipping i.pximg.net."
                )
            else:
                try:
                    from pixivpy3 import AppPixivAPI

                    api_kwargs = {"timeout": 30}
                    api = AppPixivAPI(**api_kwargs)
                    api.auth(refresh_token=refresh_token)

                    file_ext = "jpg"
                    try:
                        path = urlparse(url_to_download).path
                        if "." in path:
                            ext = path.split(".")[-1].lower()
                            if ext in ["jpg", "jpeg", "png", "gif", "webp", "bmp"]:
                                file_ext = ext
                    except Exception:
                        pass

                    file_name = f"mem_chat_{int(time.time())}_{random.randint(1000, 9999)}.{file_ext}"
                    temp_dir = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "..", "temp")
                    )
                    os.makedirs(temp_dir, exist_ok=True)
                    await asyncio.to_thread(
                        api.download, url_to_download, path=temp_dir, name=file_name
                    )

                    temp_path = os.path.join(temp_dir, file_name)
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        chain_obj.file_image(temp_path)
                        temp_list.append(temp_path)
                        return True, False
                except Exception as e:
                    logger.error(f"[mem_chat] Pixiv API download failed: {e}")
                    pass

        # Pixiv Artwork URL
        pixiv_match = re.search(r"pixiv\.net/(?:en/)?artworks/(\d+)", url_to_download)
        if pixiv_match:
            pid = pixiv_match.group(1)
            cfg = self.plugin.context.get_config(umo=umo)
            prov_settings = cfg.get("provider_settings", {})
            refresh_token = prov_settings.get(
                "pixiv_refresh_token"
            ) or self.plugin.config.get("pixiv_refresh_token")
            if refresh_token:
                try:
                    from pixivpy3 import AppPixivAPI

                    api_kwargs = {"timeout": 30}
                    api = AppPixivAPI(**api_kwargs)
                    api.auth(refresh_token=refresh_token)
                    json_result = api.illust_detail(pid)
                    illust = json_result.get("illust") if json_result else None
                    if not illust:
                        raise RuntimeError("Pixiv illust_detail returned empty illust")

                    original_url = illust.get("meta_single_page", {}).get(
                        "original_image_url"
                    ) or illust.get("image_urls", {}).get("large")
                    if not original_url:
                        raise RuntimeError(
                            "Pixiv illust_detail missing original_image_url"
                        )

                    file_ext = "jpg"
                    if "." in original_url:
                        file_ext = original_url.split(".")[-1].lower()
                    file_name = f"mem_chat_{int(time.time())}_{random.randint(1000, 9999)}.{file_ext}"
                    temp_dir = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "..", "temp")
                    )
                    os.makedirs(temp_dir, exist_ok=True)
                    await asyncio.to_thread(
                        api.download, original_url, path=temp_dir, name=file_name
                    )
                    temp_path = os.path.join(temp_dir, file_name)
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        chain_obj.file_image(temp_path)
                        temp_list.append(temp_path)
                        return True, False
                    raise RuntimeError("Pixiv api.download produced empty file")
                except Exception as e:
                    logger.error(f"[mem_chat] Pixiv API resolution failed: {e}")
                    pass

            resolved = await self._resolve_pixiv_artwork_image(url_to_download)
            if resolved:
                url_to_download = resolved
            else:
                return False, False

        # Twitter/X
        if "twitter.com" in url_to_download or "x.com" in url_to_download:
            resolved = await self._resolve_social_media_image(url_to_download)
            if resolved:
                url_to_download = resolved

        # Normal Download
        file_ext = "jpg"
        try:
            path = urlparse(url_to_download).path
            if "." in path:
                ext = path.split(".")[-1].lower()
                if ext in ["jpg", "jpeg", "png", "gif", "webp", "bmp"]:
                    file_ext = ext
        except Exception:
            pass

        file_name = (
            f"mem_chat_{int(time.time())}_{random.randint(1000, 9999)}.{file_ext}"
        )
        temp_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "temp")
        )
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file_name)

        try:
            if "i.pximg.net" in url_to_download:
                headers["Referer"] = "https://www.pixiv.net"

            # 增加通用 User-Agent，解决 Reddit 等网站的 403 问题
            if "redd.it" in url_to_download or "reddit.com" in url_to_download:
                headers["User-Agent"] = (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            elif "headers" not in headers:
                headers["User-Agent"] = (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )

            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.get(
                    url_to_download, headers=headers, timeout=30
                ) as resp:
                    if resp.status == 200:
                        content_type = resp.headers.get("Content-Type", "").lower()
                        if "text/html" in content_type:
                            return False, False
                        data = await resp.read()
                        with open(temp_path, "wb") as f:
                            f.write(data)
                        if os.path.getsize(temp_path) == 0:
                            return False, False
                        chain_obj.file_image(temp_path)
                        temp_list.append(temp_path)
                        return True, False
                    elif resp.status == 404:
                        return False, True
        except Exception as e:
            logger.error(f"[mem_chat] Download error: {e}")
        return False, False

    async def generate_text(
        self, prompt: str, system_prompt: str = "", umo: str = ""
    ) -> str:
        custom_base = self._get_conf("custom_llm_base_url", "")
        custom_key = self._get_conf("custom_llm_api_key", "")
        custom_model = self._get_conf("custom_llm_model", "")
        use_custom_llm = bool(custom_base and custom_key and custom_model)

        if use_custom_llm:
            response = await self._call_custom_llm(
                base_url=custom_base,
                api_key=custom_key,
                model=custom_model,
                system_prompt=system_prompt,
                prompt=prompt,
                contexts=[],
                tools=None,
                tag="generate_text",
            )
            return response.completion_text if response else ""
        else:
            provider = self.plugin.context.get_using_provider(umo=umo)
            if not provider:
                return ""
            try:
                response = await provider.text_chat(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    session_id=umo,
                    contexts=[],
                    func_tool=None,
                )
                return response.completion_text if response else ""
            except Exception as e:
                logger.error(f"[mem_chat] generate_text error: {e}")
                return ""

    # =========================================================================
    # Refactored Helper Methods
    # =========================================================================

    def _prepare_llm_config(self, umo: str, event):
        llm_base = self._get_conf(
            "llm_base_url", self._get_conf("custom_llm_base_url", "")
        )
        llm_key = self._get_conf(
            "llm_api_key", self._get_conf("custom_llm_api_key", "")
        )
        llm_model = self._get_conf("llm_model", self._get_conf("custom_llm_model", ""))
        llm_fmt = self._get_conf(
            "llm_format", self._get_conf("custom_llm_format", "openai")
        )
        try:
            llm_temp = float(self._get_conf("llm_temperature", 1.0))
        except Exception:
            llm_temp = 1.0

        use_custom_llm = bool(llm_base and llm_key and llm_model)
        provider = self.plugin.context.get_using_provider(umo=umo)
        if self._get_bool_conf("disable_provider_llm", True):
            provider = None

        return {
            "llm": {
                "base": llm_base,
                "key": llm_key,
                "model": llm_model,
                "temp": llm_temp,
                "fmt": llm_fmt,
            },
            "use_custom_llm": use_custom_llm,
            "provider": provider,
        }

    async def _prepare_conversation_context(
        self, umo: str, memory_context: str, user_msgs_to_save: list, event
    ):
        conv_mgr = self.plugin.context.conversation_manager
        cid = await conv_mgr.get_curr_conversation_id(umo)
        conversation = None
        original_contexts = []
        if not cid:
            cid = await conv_mgr.new_conversation(umo)
            conversation = await conv_mgr.get_conversation(umo, cid)
        else:
            conversation = await conv_mgr.get_conversation(umo, cid)
            if conversation and conversation.history:
                try:
                    import json

                    raw_history = json.loads(conversation.history)
                    for msg in raw_history:
                        content = str(msg.get("content", ""))
                        if content.strip().startswith("{") and content.strip().endswith(
                            "}"
                        ):
                            try:
                                data = json.loads(content)
                                if "response" in data and "text" in data["response"]:
                                    msg["content"] = data["response"]["text"]
                                elif "text" in data:
                                    msg["content"] = data["text"]
                            except Exception:
                                pass
                    original_contexts = raw_history
                except Exception:
                    pass

        if user_msgs_to_save:
            for msg in user_msgs_to_save:
                if event:
                    msg["sender_id"] = event.get_sender_id()
                try:
                    content = str(msg.get("content", ""))
                    # 移除可能的 JSON 结构，只保留文本
                    if content.strip().startswith("{") and content.strip().endswith(
                        "}"
                    ):
                        try:
                            import json

                            data = json.loads(content)
                            if "response" in data and "text" in data["response"]:
                                content = data["response"]["text"]
                            elif "text" in data:
                                content = data["text"]
                        except Exception:
                            pass
                    msg["content"] = self._strip_search_listing_text(content)
                except Exception:
                    pass
                original_contexts.append(msg)

        history_count = 0
        try:
            history_count = int(self._get_conf("history_message_count", 12))
        except Exception:
            history_count = 12
        if history_count <= 0:
            history_count = 0

        trimmed_contexts = original_contexts
        if history_count and len(original_contexts) > history_count:
            trimmed_contexts = original_contexts[-history_count:]
        elif history_count == 0:
            trimmed_contexts = []

        if conversation and (
            user_msgs_to_save or len(trimmed_contexts) != len(original_contexts)
        ):
            await conv_mgr.update_conversation(umo, conversation.cid, trimmed_contexts)

        # Persona logic
        persona = None
        if conversation and conversation.persona_id:
            persona = next(
                (
                    p
                    for p in self.plugin.context.persona_manager.personas_v3
                    if p["name"] == conversation.persona_id
                ),
                None,
            )

        if persona:
            if persona.get("_begin_dialogs_processed"):
                llm_contexts = copy.deepcopy(persona["_begin_dialogs_processed"])
                llm_contexts.extend(trimmed_contexts)
                return {
                    "conversation": conversation,
                    "persona": persona,
                    "history": llm_contexts,
                }

        return {
            "conversation": conversation,
            "persona": persona,
            "history": trimmed_contexts,
        }

    async def _translate_for_pixiv(self, query: str, umo: str) -> list[str]:
        """Use LLM to translate query into Japanese and Romanized keywords for better Pixiv search."""
        try:
            cfg = self.plugin.context.get_config(umo=umo)
            prov_settings = cfg.get("provider_settings", {})
            base_url = prov_settings.get("custom_llm_url") or self._get_conf(
                "custom_llm_url", ""
            )
            api_key = prov_settings.get("custom_llm_key") or self._get_conf(
                "custom_llm_key", ""
            )
            model = prov_settings.get("custom_llm_model") or self._get_conf(
                "custom_llm_model", ""
            )

            if not base_url or not api_key or not model:
                return []

            system_prompt = "You are a translation assistant. Translate the keyword into Japanese Pixiv tags and Romanized Japanese. Return only the translations separated by commas. No explanation."
            user_prompt = f"Keyword: {query}"

            # Use a short timeout and no history for speed
            resp_obj = await self._call_custom_llm(
                base_url,
                api_key,
                model,
                system_prompt,
                user_prompt,
                [],
                tag="pixiv_trans",
            )
            if (
                resp_obj
                and hasattr(resp_obj, "completion_text")
                and resp_obj.completion_text
            ):
                keywords = [
                    k.strip() for k in resp_obj.completion_text.split(",") if k.strip()
                ]
                return keywords
            return []
        except Exception as e:
            logger.error(f"[mem_chat] Pixiv translation failed: {e}")
            return []

    async def _proactive_pixiv_search(
        self, query: str, umo: str, max_results: int = 5
    ) -> list[str]:
        """Proactively search Pixiv for artworks based on keywords."""
        try:
            from pixivpy3 import AppPixivAPI
        except ImportError:
            logger.info(
                "[mem_chat] Proactive Pixiv search skipped: pixivpy3 not installed."
            )
            return []

        cfg = self.plugin.context.get_config(umo=umo)
        prov_settings = cfg.get("provider_settings", {})
        refresh_token = prov_settings.get(
            "pixiv_refresh_token"
        ) or self.plugin.config.get("pixiv_refresh_token")
        if not refresh_token:
            logger.info(
                "[mem_chat] Proactive Pixiv search skipped: pixiv_refresh_token not configured."
            )
            return []

        # R-18 筛选设置
        allow_r18 = bool(
            cfg.get("pixiv_allow_r18", False)
            or prov_settings.get("pixiv_allow_r18", False)
        )
        allow_r18g = bool(
            cfg.get("pixiv_allow_r18g", False)
            or prov_settings.get("pixiv_allow_r18g", False)
        )

        # 清洗查询词
        noise = {
            "イラスト",
            "插画",
            "fanart",
            "アート",
            "艺术",
            "art",
            "画",
            "图",
            "图片",
        }
        try:
            parts = [
                p for p in [s.strip() for s in query.split()] if p and p not in noise
            ]
        except Exception:
            parts = [query.strip()] if query else []
        q_clean = " ".join(parts) if parts else (query or "")
        q_clean = self._clean_image_search_query(q_clean)

        # 多语言扩展
        search_keywords = [q_clean]
        is_japanese = any(
            "\u3040" <= c <= "\u309f" or "\u30a0" <= c <= "\u30ff" for c in q_clean
        )
        if not is_japanese and q_clean:
            translated = await self._translate_for_pixiv(q_clean, umo)
            for tk in translated:
                if tk.lower() != q_clean.lower() and tk not in search_keywords:
                    search_keywords.append(tk)

        strategies = []
        for kw in search_keywords:
            kw_parts = [p.strip() for p in kw.split() if p.strip()]
            if len(kw_parts) == 1:
                strategies.append((kw_parts[0], "exact_match_for_tags"))
            strategies.append((kw, "title_and_caption"))
            strategies.append((kw, "partial_match_for_tags"))

        timeout_sec = 25
        try:
            timeout_sec = int(
                prov_settings.get(
                    "pixiv_timeout_sec", cfg.get("pixiv_timeout_sec", timeout_sec)
                )
            )
        except Exception:
            timeout_sec = 25
        timeout_sec = max(5, min(60, timeout_sec))

        def run_search() -> list[str]:
            import socket

            socket.setdefaulttimeout(timeout_sec)

            api_kwargs = {"timeout": timeout_sec}
            api = AppPixivAPI(**api_kwargs)
            api.auth(refresh_token=refresh_token)

            recent_days = 90
            try:
                recent_days = int(
                    prov_settings.get(
                        "pixiv_recent_days", cfg.get("pixiv_recent_days", recent_days)
                    )
                )
            except Exception:
                recent_days = 90

            recent_items = []
            older_items = []
            seen = set()

            for q, target in strategies:
                if len(recent_items) >= max_results:
                    break
                try:
                    jr = api.search_illust(q, search_target=target, sort="date_desc")
                except Exception:
                    continue
                illusts = jr.get("illusts", []) if isinstance(jr, dict) else []

                next_url = jr.get("next_url") if isinstance(jr, dict) else None
                while True:
                    for illust in illusts:
                        if len(recent_items) >= max_results:
                            break
                        if illust.get("type") != "illust":
                            continue
                        pid = illust.get("id")
                        if not pid or pid in seen:
                            continue

                        xr = int(illust.get("x_restrict", 0) or 0)
                        if (xr == 1 and not allow_r18) or (xr == 2 and not allow_r18g):
                            continue
                        seen.add(pid)
                        title = illust.get("title") or ""
                        user = illust.get("user", {}).get("name") or ""
                        url = f"https://www.pixiv.net/artworks/{pid}"

                        is_recent = False
                        try:
                            cd = illust.get("create_date") or illust.get(
                                "create_datetime"
                            )
                            if isinstance(cd, str) and cd:
                                if cd.endswith("Z"):
                                    cd = cd.replace("Z", "+00:00")
                                dt = datetime.fromisoformat(cd)
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=timezone.utc)
                                age_days = (
                                    datetime.now(timezone.utc)
                                    - dt.astimezone(timezone.utc)
                                ).days
                                is_recent = age_days <= recent_days
                        except Exception:
                            is_recent = False

                        if is_recent:
                            recent_items.append((title, user, url))
                        else:
                            older_items.append((title, user, url))

                    if len(recent_items) >= max_results or not next_url:
                        break
                    try:
                        params = api.parse_qs(next_url)
                        jr = api.search_illust(**params)
                        illusts = jr.get("illusts", []) if isinstance(jr, dict) else []
                        next_url = jr.get("next_url") if isinstance(jr, dict) else None
                    except Exception:
                        break

            ret = []
            idx = 1
            for title, user, url in recent_items[:max_results]:
                ret.append(f"[{idx}] {title} - Author: {user} (URL: {url})")
                idx += 1
            if len(ret) < max_results:
                for title, user, url in older_items:
                    if len(ret) >= max_results:
                        break
                    ret.append(f"[{idx}] {title} - Author: {user} (URL: {url})")
                    idx += 1

            return ret

        try:
            logger.info(
                f"[mem_chat] Proactive Pixiv search start query={q_clean} timeout={timeout_sec}s"
            )
            ret = await asyncio.wait_for(
                asyncio.to_thread(run_search), timeout=timeout_sec + 5
            )
            if not ret:
                logger.info(
                    f"[mem_chat] Proactive Pixiv search got 0 results for query: {q_clean}"
                )
            return ret
        except asyncio.TimeoutError:
            logger.error(
                f"[mem_chat] Proactive Pixiv search timed out for query: {q_clean}"
            )
            return []
        except Exception as e:
            logger.error(f"[mem_chat] Proactive Pixiv search failed: {e}")
            return []

    def _clean_image_search_query(self, text: str) -> str:
        if not text:
            return ""
        t = str(text)
        t = re.sub(r"\[At:\d+\]", "", t).strip()
        t = re.sub(r"\[回复[^\]]*\]", "", t).strip()
        t = re.sub(r"@\d+", "", t).strip()
        t = t.replace("@全体成员", "").strip()

        t = re.sub(
            r"(请|麻烦|帮我|给我|发我|发给我|再来|再给|来|给|发|换|找|搜|搜索|查|查询|想看|想要|我要)",
            " ",
            t,
        )
        t = re.sub(r"[一二两三四五六七八九十几多0-9]+\s*张", " ", t)
        t = re.sub(r"(一张|两张|几张|多张|张)", " ", t)
        t = re.sub(r"(别的|不同的|换个|换张|另一张)", " ", t)
        t = re.sub(r"(图片|插图|插画|图|立绘|壁纸|头像|表情包)", " ", t)
        t = re.sub(r"\s+", " ", t).strip()

        if t:
            cjk = re.findall(r"[\u4e00-\u9fa5]{2,}", t)
            if cjk:
                return cjk[-1].strip()
        return t

    async def _extract_pixiv_search_plan(
        self, text: str, contexts: list[dict], llm_config: dict, umo: str
    ) -> tuple[list[str], int]:
        system_prompt = (
            "你是一个搜图助手。你的任务是根据对话内容，提炼用于 Pixiv 搜图的关键词，适当补全关键词，比如补全罗马音、日文等pixiv常用关键词，并判断用户想要的图片数量。\n"
            "只输出一个 JSON 对象，不要输出任何解释文字。\n"
            'JSON 格式：{"keywords":["关键词1","关键词2"],"count":N}\n'
            '要求：keywords 为 1-3 个字符串；count 为 1-5 的整数。无法提炼则返回 {"keywords":[],"count":1}。'
        )
        user_prompt = f"当前请求：{text}"

        raw = ""
        try:
            if llm_config.get("use_custom_llm"):
                cfg = llm_config.get("llm") or {}
                resp_obj = await self._call_custom_llm(
                    base_url=cfg.get("base"),
                    api_key=cfg.get("key"),
                    model=cfg.get("model"),
                    system_prompt=system_prompt,
                    prompt=user_prompt,
                    contexts=contexts or [],
                    tools=None,
                    image_urls=[],
                    temperature=cfg.get("temp"),
                    tag="img_plan",
                )
                raw = getattr(resp_obj, "completion_text", "") or ""
            else:
                provider = llm_config.get("provider")
                if not provider:
                    return [], 1
                resp_obj = await provider.text_chat(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    session_id=f"{umo}:img_plan",
                    contexts=contexts or [],
                    func_tool=None,
                )
                raw = getattr(resp_obj, "completion_text", "") or ""
        except Exception as e:
            logger.error(f"[mem_chat] Pixiv search plan extraction failed: {e}")
            return [], 1

        data = None
        try:
            data = StructuredResponse.extract_json_from_text(raw)
        except Exception:
            data = None
        if not isinstance(data, dict):
            return [], 1

        kws = data.get("keywords")
        if isinstance(kws, str):
            kws = [kws]
        if not isinstance(kws, list):
            kws = []
        cleaned = []
        seen = set()
        for k in kws[:3]:
            if not isinstance(k, str):
                continue
            kk = self._clean_image_search_query(k)
            if not kk:
                continue
            key = kk.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(kk)

        cnt = data.get("count")
        try:
            cnt = int(cnt)
        except Exception:
            cnt = 1
        cnt = max(1, min(5, cnt))
        return cleaned, cnt

    async def _extract_keyword_with_llm(self, text: str, umo: str) -> str:
        """Use LLM to extract the core search keyword from user message."""
        try:
            cfg = self.plugin.context.get_config(umo=umo)
            prov_settings = cfg.get("provider_settings", {})
            base_url = prov_settings.get("custom_llm_url") or self._get_conf(
                "custom_llm_url", ""
            )
            api_key = prov_settings.get("custom_llm_key") or self._get_conf(
                "custom_llm_key", ""
            )
            model = prov_settings.get("custom_llm_model") or self._get_conf(
                "custom_llm_model", ""
            )

            if not base_url or not api_key or not model:
                return ""

            system_prompt = "你是一个关键词提取助手。请从用户的搜图请求中提取核心对象名称。不要包含'搜'、'图'、'一张'等词。只返回关键词本身，不要任何解释。如果无法提取，返回空。"
            user_prompt = f"请求内容: {text}"

            resp_obj = await self._call_custom_llm(
                base_url,
                api_key,
                model,
                system_prompt,
                user_prompt,
                [],
                tag="extract_kw",
            )
            if (
                resp_obj
                and hasattr(resp_obj, "completion_text")
                and resp_obj.completion_text
            ):
                return self._clean_image_search_query(resp_obj.completion_text.strip())
            return ""
        except Exception as e:
            logger.error(f"[mem_chat] Keyword extraction failed: {e}")
            return ""

    def _strip_reply_blocks(self, text: str) -> str:
        if not text:
            return ""
        s = str(text)
        try:
            s = re.sub(r"\[CQ:reply,[^\]]*\]", "", s, flags=re.IGNORECASE)
        except Exception:
            pass

        out: list[str] = []
        i = 0
        while i < len(s):
            if s.startswith("[回复", i):
                depth = 0
                while i < len(s):
                    ch = s[i]
                    if ch == "[":
                        depth += 1
                    elif ch == "]":
                        depth -= 1
                        if depth <= 0:
                            i += 1
                            break
                    i += 1
                continue
            out.append(s[i])
            i += 1
        return "".join(out)

    def _detect_intents(
        self,
        umo: str,
        prompt: str,
        user_msgs: list,
        task_type: str,
        event,
        enable_search: bool = True,
    ):
        last_user_text = ""
        full_block_text = ""
        if user_msgs:
            self_id = None
            try:
                if event and hasattr(event, "get_self_id"):
                    self_id = event.get_self_id()
            except Exception:
                self_id = None
            if not self_id:
                try:
                    self_id = self.plugin.state.get_self_id(umo)
                except Exception:
                    self_id = None

            user_contents = []
            for m in user_msgs:
                if m.get("role") != "user":
                    continue
                if self_id is not None and str(m.get("sender_id") or "") == str(
                    self_id
                ):
                    continue
                user_contents.append(str(m.get("content", "")))
            full_block_text = " ".join(user_contents)
            last_user_text = user_contents[-1] if user_contents else ""
        elif prompt:
            full_block_text = str(prompt)
            last_user_text = str(prompt)

        # Alias Intent
        alias_intent = False
        try:
            # 别名意图识别：优先看最后一条，但也参考整个块
            target_text = last_user_text or full_block_text
            toks = re.findall(
                r"[\u4e00-\u9fa5]+|[A-Za-z]+|[\u3040-\u30ff]+", target_text or ""
            )
            toks = [t.strip().lower() for t in toks if len(t) >= 2]
            if toks and len(toks) <= 3:
                alias_intent = True
            if re.search(
                r"(原名|本名|真名|日文名|全称|昵称|别名)", full_block_text or ""
            ):
                alias_intent = True
        except Exception:
            pass

        # Image Intent
        image_intent = False
        search_keyword = ""
        at_message_content = ""  # 精准匹配到的 AT 消息内容
        try:
            # 1. 识别消息块中是否包含显式的 AT 机器人指令
            # 我们在 user_contents 中寻找包含 AT 标签或机器人 ID 的行
            if user_msgs:
                for content in reversed(user_contents):
                    # 匹配 [At:self_id] 或 @self_id 或 机器人昵称等（简单正则）
                    # 注意：self_id 在前面已经获取
                    if self_id and (
                        f"[At:{self_id}]" in content or f"@{self_id}" in content
                    ):
                        # 提取 AT 之后的内容作为搜图指令的来源
                        at_text = re.sub(
                            rf"\[At:{self_id}\]|@{self_id}", "", content
                        ).strip()
                        if at_text:
                            at_message_content = at_text
                            break

            # 2. 意图识别基础文本：优先使用精准匹配到的 AT 内容，否则用最后一条用户消息
            text_for_intent_raw = at_message_content or last_user_text
            text_for_intent = self._strip_reply_blocks(
                text_for_intent_raw or ""
            ).strip()
            text_for_intent = re.sub(r"\[At:\d+\]", "", text_for_intent).strip()
            text_for_intent = re.sub(r"@\d+", "", text_for_intent).strip()
            text_for_intent = text_for_intent.replace("@全体成员", "").strip()

            msg_l = text_for_intent.lower()
            kw_list = [
                "图",
                "图片",
                "插图",
                "找图",
                "搜图",
                "立绘",
                "壁纸",
                "头像",
                "插画",
                "pixiv",
                "twitter",
                "x.com",
                "artworks",
                "illustration",
                "image",
                "picture",
            ]

            # 强化匹配：如果是“再来一张”，即便没有明显的搜图词，也视为图片意图
            if any(kw in msg_l for kw in kw_list):
                image_intent = True
            if re.search(r"(来|再来|换|发|找)[一二两三四五几多]?\s*张", msg_l):
                image_intent = True

            # Guard: Explicitly exclude "表情包" to prevent false positives in normal chat
            if "表情包" in msg_l:
                image_intent = False

            if image_intent:
                # 尝试从识别出的文本中提取关键词
                # 1. 移除常见前缀
                clean_text = re.sub(
                    r"^(请|帮我|想看|给我|来|再来|找|搜|搜索|查|查询|看|看看|一张|两张|几张|多张|张|别的|换个|换张|换)+",
                    "",
                    text_for_intent,
                ).strip()
                # 2. 移除常见后缀
                clean_text = re.sub(
                    r"(的)?(图|图片|插图|插画|立绘|壁纸|头像|表情包|样子|模样|照片|画|作品|artworks|illustration|image|picture|pixiv|twitter|x.com)+$",
                    "",
                    clean_text,
                ).strip()

                if clean_text:
                    search_keyword = self._clean_image_search_query(clean_text)
                if not search_keyword:
                    image_intent = False
        except Exception:
            pass

        allow_search = enable_search

        # Override: If explicit command/at, allow search (if not already enabled)
        # But generally enable_search passed from invoke_llm should be the source of truth based on State.
        if event and (
            getattr(event, "is_at_or_wake_command", False)
            or getattr(event, "is_command", False)
        ):
            allow_search = True

        return {
            "alias": alias_intent,
            "image": image_intent,
            "search_keyword": search_keyword,
            "allow_search": allow_search,
            "last_text": last_user_text,
            "full_text": full_block_text,
            "at_text": at_message_content,  # 传回精准匹配到的 AT 消息
        }

    def _setup_tools(self, umo: str, intents: dict, persona):
        if not self._get_bool_conf("enable_llm_tools", False):
            return None, False

        tmgr = self.plugin.context.get_llm_tool_manager()
        tools = None
        persona_explicit_tools = set()
        if persona:
            if persona.get("tools") is None:
                tools = tmgr.get_full_tool_set()
            else:
                tools = ToolSet()
                for tool_name in persona["tools"]:
                    t = tmgr.get_func(tool_name)
                    if t and t.active:
                        tools.add_tool(t)
                        persona_explicit_tools.add(tool_name)
                for t_name in ["web_search", "web_search_tavily"]:
                    t = tmgr.get_func(t_name)
                    if t:
                        tools.add_tool(t)
        else:
            tools = tmgr.get_full_tool_set()

        if tools and hasattr(tools, "tools"):
            to_remove = [t.name for t in tools.tools if not t.active]
            # 显式移除 search_images 工具，因为我们现在改用主动检索注入
            to_remove.append("search_images")

            if hasattr(tools, "remove_tool"):
                for name in set(to_remove):
                    tools.remove_tool(name)

        allow_search = intents["allow_search"]
        if not allow_search:
            # 对于搜索类工具，如果不是显式允许搜索，则移除。
            search_tools = [
                "web_search",
                "web_search_tavily",
                "tavily_extract_web_page",
                "AIsearch",
                "google_search",
            ]
            if tools and hasattr(tools, "remove_tool"):
                for tname in search_tools:
                    # 如果是 Persona 显式要求的工具，不移除
                    if tname in persona_explicit_tools:
                        continue
                    tools.remove_tool(tname)

                # 对于 search_images，由于我们现在改用主动注入逻辑，它已经被移除了。
                pass
        elif allow_search:
            cfg = self.plugin.context.get_config(umo=umo)
            prov_settings = cfg.get("provider_settings", {})
            ws_provider = prov_settings.get("websearch_provider", "default")

            if ws_provider == "tavily":
                t_tavily = tmgr.get_func("web_search_tavily")
                t_extract = tmgr.get_func("tavily_extract_web_page")
                if tools:
                    if t_tavily:
                        tools.add_tool(t_tavily)
                    if t_extract:
                        tools.add_tool(t_extract)
                    if hasattr(tools, "remove_tool"):
                        tools.remove_tool("web_search")
                        tools.remove_tool("fetch_url")
                        tools.remove_tool("AIsearch")
            else:
                t_search = tmgr.get_func("web_search")
                t_fetch = tmgr.get_func("fetch_url")
                if tools:
                    if t_search:
                        tools.add_tool(t_search)
                    if t_fetch:
                        tools.add_tool(t_fetch)
                    if hasattr(tools, "remove_tool"):
                        tools.remove_tool("web_search_tavily")
                        tools.remove_tool("tavily_extract_web_page")

        has_search_tools = False
        if tools and getattr(tools, "tools", None):
            tool_names = [t.name for t in tools.tools]
            if any(
                n in tool_names
                for n in [
                    "search_images",
                    "web_search",
                    "web_search_tavily",
                    "tavily_extract_web_page",
                    "fetch_url",
                ]
            ):
                has_search_tools = True
        return tools, has_search_tools

    def _build_instructions(
        self,
        instruction_text,
        quote_candidates,
        at_candidates,
        image_urls,
        intents,
        tools,
    ):
        tool_instrs = []
        tool_instr = "\n".join(tool_instrs) if tool_instrs else ""
        instruction_suffix = self._get_instruction_prompts(
            instruction_text, tool_instructions=tool_instr
        )
        at_instruction = ""

        if at_candidates:
            at_list = []
            for i, c in enumerate(at_candidates):
                uid = c[0]
                uname = c[1] if len(c) > 1 else "未知"
                at_list.append(f"{i + 1}. {uname} (ID: {uid})")
            at_instruction = "\n【可艾特的人员列表】\n" + "\n".join(at_list)
            instruction_suffix += at_instruction

        if image_urls:
            instruction_suffix += "\n\n【系统提示】\n用户发送了图片，你能够看到这张图片。请在回复中简要描述图片内容，并根据内容进行回复。"

        return {
            "suffix": instruction_suffix,
            "at_instr": at_instruction,
        }

    def _build_system_prompts(self, persona, memory_context, umo):
        system_prompt_base = ""
        if persona and "prompt" in persona:
            p_text = str(persona.get("prompt") or "").strip()
            if p_text:
                if "【人物印象/设定】" in p_text:
                    system_prompt_base = p_text
                else:
                    system_prompt_base = "【人物印象/设定】\n" + p_text
        full_system_prompt = (
            f"{system_prompt_base}\n\n{memory_context}"
            if system_prompt_base
            else memory_context
        )

        self_id = self.plugin.state.get_self_id(umo)
        if self_id:
            self_name = self.plugin.state.get_self_name(umo)
            if self_name:
                full_system_prompt += (
                    f"\n\n【系统信息】\n你的ID是: {self_id}/{self_name}"
                )
            else:
                full_system_prompt += f"\n\n【系统信息】\n你的ID是: {self_id}"

        guard = (
            "\n\n【群聊规则】\n"
            "- 仅在被@、被明确点名（含你的ID/昵称），或当前处于专注模式/被要求回应时进行直接回应。以【当前最新对话】为最高优先级：优先回答用户刚发/正在讨论的内容；【历史对话】仅作背景参考，可能过期或无关，不得据此继续回复旧的@/旧话题。\n"
            "- 未被点名时可以进行简短、克制的第三人称吐槽/旁白；请明确你不是被指向的对象，避免使用第一人称代入。大家一般叫你爱丽丝，不会有什么别的称呼。\n"
            '- 对两人互相对骂的场景，即使包含模糊提及，也应选择不回复（control.decision="no_reply"）；不要参与或加剧争吵。\n'
            '- 若消息与机器人完全无关，或对方明确不希望你介入，应选择不回复（control.decision="no_reply"）。\n'
            '- 不确定是否需要回复，也应选择不回复（control.decision="no_reply"）。'
        )
        full_system_prompt += guard
        return full_system_prompt

    async def _process_and_send_final_reply(
        self,
        final_reply,
        event,
        task_type,
        quote_candidates,
        at_candidates,
        llm_contexts,
        umo,
        user_msgs_to_save,
    ):
        structured = StructuredResponse.parse_validated(final_reply)
        if structured:
            resp = structured.get("response") or {}
            control = structured.get("control") or {}

            # 处理禁言逻辑 (Mute)
            mute_info = control.get("mute")
            if (
                isinstance(mute_info, dict)
                and event
                and hasattr(event, "message_obj")
                and hasattr(event.message_obj, "group_id")
            ):
                target_qq = str(mute_info.get("target_qq") or "")
                duration = mute_info.get("duration")

                if target_qq and duration:
                    try:
                        # 转换时长为整数，限制在 1-100s
                        duration_int = int(duration)
                        duration_int = max(1, min(100, duration_int))

                        logger.info(
                            f"[mem_chat] Attempting to mute {target_qq} in group {event.message_obj.group_id} for {duration_int}s"
                        )

                        # 尝试多种可能的禁言 API (参考 animewifex)
                        success = False
                        for method_name in ["set_group_ban", "group_ban"]:
                            if hasattr(event.bot, method_name):
                                try:
                                    method = getattr(event.bot, method_name)
                                    await method(
                                        group_id=int(event.message_obj.group_id),
                                        user_id=int(target_qq),
                                        duration=duration_int,
                                    )
                                    success = True
                                    logger.info(
                                        f"[mem_chat] Mute successful using event.bot.{method_name}"
                                    )
                                    break
                                except Exception as e:
                                    logger.error(
                                        f"[mem_chat] Mute failed using event.bot.{method_name}: {e}"
                                    )

                        if not success:
                            # 尝试 adapter
                            adapter = getattr(event, "adapter", None)
                            if adapter:
                                for method_name in ["set_group_ban", "group_ban"]:
                                    if hasattr(adapter, method_name):
                                        try:
                                            method = getattr(adapter, method_name)
                                            await method(
                                                group_id=int(
                                                    event.message_obj.group_id
                                                ),
                                                user_id=int(target_qq),
                                                duration=duration_int,
                                            )
                                            success = True
                                            logger.info(
                                                f"[mem_chat] Mute successful using adapter.{method_name}"
                                            )
                                            break
                                        except Exception as e:
                                            logger.error(
                                                f"[mem_chat] Mute failed using adapter.{method_name}: {e}"
                                            )

                    except Exception as e:
                        logger.error(f"[mem_chat] Mute operation failed: {e}")

            # 处理群头衔逻辑 (Set Group Title)
            title_info = control.get("set_group_title")
            if (
                isinstance(title_info, dict)
                and event
                and hasattr(event, "message_obj")
                and hasattr(event.message_obj, "group_id")
            ):
                target_qq = str(title_info.get("target_qq") or "")
                new_title = str(title_info.get("new_title") or "")

                if target_qq and new_title:
                    try:
                        logger.info(
                            f"[mem_chat] Attempting to set group title for {target_qq} in group {event.message_obj.group_id} to {new_title}"
                        )

                        success = False
                        # 尝试多种可能的 API
                        for method_name in [
                            "set_group_special_title",
                            "set_group_card",
                        ]:
                            if hasattr(event.bot, method_name):
                                try:
                                    method = getattr(event.bot, method_name)
                                    params = {
                                        "group_id": int(event.message_obj.group_id),
                                        "user_id": int(target_qq),
                                    }
                                    if method_name == "set_group_special_title":
                                        params["special_title"] = new_title
                                    else:
                                        params["card"] = new_title

                                    await method(**params)
                                    success = True
                                    logger.info(
                                        f"[mem_chat] Set title successful using event.bot.{method_name}"
                                    )
                                    break
                                except Exception as e:
                                    logger.error(
                                        f"[mem_chat] Set title failed using event.bot.{method_name}: {e}"
                                    )

                        if not success:
                            # 尝试 call_action 兜底
                            try:
                                await event.bot.call_action(
                                    "set_group_special_title",
                                    group_id=int(event.message_obj.group_id),
                                    user_id=int(target_qq),
                                    special_title=new_title,
                                )
                                success = True
                                logger.info(
                                    "[mem_chat] Set title successful using event.bot.call_action"
                                )
                            except Exception as e:
                                logger.error(
                                    f"[mem_chat] Set title failed using event.bot.call_action: {e}"
                                )

                    except Exception as e:
                        logger.error(f"[mem_chat] Set title operation failed: {e}")

            base_emotion = resp.get("emotion")
            send_event = event if event else DummyEvent(umo, self.plugin.context)

            segs = resp.get("segments")
            segments = []
            if isinstance(segs, list) and segs:
                segments = [s for s in segs if isinstance(s, dict)]
            if not segments:
                segments = [resp]

            for seg in segments:
                temp_files_to_clean: list[str] = []
                segment_text = (
                    seg.get("text") if isinstance(seg.get("text"), str) else ""
                )
                segment_text = self._sanitize_final_reply(segment_text)
                segment_text = self._strip_search_listing_text(segment_text)

                emotions_to_send: list[str] = []
                raw_emotion = seg.get("emotion", base_emotion)
                if isinstance(raw_emotion, str) and raw_emotion.strip():
                    s = raw_emotion.strip()
                    bracketed = re.findall(r"\[([^\[\]]+)\]", s)
                    if bracketed:
                        for b in bracketed:
                            t = (b or "").strip()
                            if t:
                                emotions_to_send.append(t)
                    else:
                        parts = re.split(r"[,，\s]+", s)
                        for p in parts:
                            t = (p or "").strip()
                            if t.startswith("&&") and t.endswith("&&") and len(t) >= 4:
                                t = t[2:-2].strip()
                            if t.startswith("[") and t.endswith("]") and len(t) >= 2:
                                t = t[1:-1].strip()
                            if t.startswith("(") and t.endswith(")") and len(t) >= 2:
                                t = t[1:-1].strip()
                            if t:
                                emotions_to_send.append(t)

                mentions = seg.get("mentions")
                if not isinstance(mentions, list):
                    mentions = (
                        resp.get("mentions")
                        if isinstance(resp.get("mentions"), list)
                        else []
                    )
                at_position = seg.get("at_position")
                if not isinstance(at_position, str) or not at_position.strip():
                    at_position = resp.get("at_position")
                at_position_norm = (at_position or "prefix").strip().lower()

                quote_index = seg.get("quote_index", resp.get("quote_index"))
                quote_id = None
                quote_sid = None
                if (
                    isinstance(quote_index, int)
                    and quote_candidates
                    and 1 <= quote_index <= len(quote_candidates)
                ):
                    quote_item = quote_candidates[quote_index - 1]
                    quote_id = quote_item[0]
                    quote_sid = quote_item[2] if len(quote_item) > 2 else None

                chain = MessageChain()

                explicit_images = []
                images_field = seg.get("images")
                if not isinstance(images_field, list):
                    images_field = (
                        resp.get("images")
                        if isinstance(resp.get("images"), list)
                        else []
                    )
                if isinstance(images_field, list):
                    for it in images_field:
                        if isinstance(it, str) and it.strip():
                            explicit_images.append(it.strip())
                image_urls_field = seg.get("image_urls")
                if not isinstance(image_urls_field, list):
                    image_urls_field = (
                        resp.get("image_urls")
                        if isinstance(resp.get("image_urls"), list)
                        else []
                    )
                if isinstance(image_urls_field, list):
                    for it in image_urls_field:
                        if isinstance(it, str) and it.strip():
                            explicit_images.append(it.strip())
                segment_text = re.sub(r"`(https?://[^`]+)`", r" \1 ", segment_text)
                for m in re.finditer(
                    r'!\s*["\'`]?\s*(https?://[^\s"\'<>`]+)["\'`]?', segment_text
                ):
                    url = m.group(1).strip("`\"'.;,)]} !?")
                    explicit_images.append(url)
                segment_text = re.sub(
                    r'!\s*["\'`]?\s*https?://[^\s"\'<>`]+["\'`]?', "", segment_text
                )

                for m in re.finditer(r"!\[(.*?)\]\((.*?)\)", segment_text):
                    url = (m.group(2) or "").strip("`\"' ")
                    if url:
                        explicit_images.append(url)
                segment_text = re.sub(r"!\[(.*?)\]\((.*?)\)", "", segment_text)

                # 独立实现表情解析和文本清理逻辑
                valid_emoticons = []
                memes_dir = os.path.join(os.getcwd(), "data", "memes_data", "memes")
                if os.path.exists(memes_dir):
                    valid_emoticons = [
                        d
                        for d in os.listdir(memes_dir)
                        if os.path.isdir(os.path.join(memes_dir, d))
                    ]

                # 1. &&emotion&& 格式
                hex_matches = list(re.finditer(r"&&([^&]+)&&", segment_text))
                for m in hex_matches:
                    emo = m.group(1).strip()
                    if emo in valid_emoticons:
                        emotions_to_send.append(emo)
                        segment_text = segment_text.replace(m.group(0), "", 1)
                    else:
                        segment_text = segment_text.replace(
                            m.group(0), "", 1
                        )  # 静默移除无效标签

                # 2. [emotion] 格式
                bracket_matches = list(re.finditer(r"\[([^\[\]]+)\]", segment_text))
                for m in bracket_matches:
                    emo = m.group(1).strip()
                    if emo in valid_emoticons:
                        emotions_to_send.append(emo)
                        segment_text = segment_text.replace(m.group(0), "", 1)

                # 3. (emotion) 格式
                paren_matches = list(re.finditer(r"\(([^()]+)\)", segment_text))
                for m in paren_matches:
                    emo = m.group(1).strip()
                    if emo in valid_emoticons:
                        emotions_to_send.append(emo)
                        segment_text = segment_text.replace(m.group(0), "", 1)

                segment_text_for_url_scan = segment_text
                segment_text_for_msg = segment_text.strip()
                try:
                    urls_in_text = re.findall(
                        r"(https?://[^\s]+)", segment_text_for_msg
                    )
                    non_url = re.sub(
                        r"(https?://[^\s]+)", "", segment_text_for_msg
                    ).strip()
                    if urls_in_text and not non_url:
                        segment_text_for_msg = ""
                except Exception:
                    pass

                mentioned_ids = set()
                mention_ids: list[str] = []
                for m in mentions:
                    qq = None
                    if isinstance(m, dict):
                        qq = (
                            m.get("qq")
                            or m.get("id")
                            or m.get("user_id")
                            or m.get("sender_id")
                        )
                    elif isinstance(m, str | int):
                        qq = str(m)

                    if qq is None:
                        continue
                    qq_str = str(qq).strip()
                    if not qq_str or qq_str in mentioned_ids:
                        continue

                    mentioned_ids.add(qq_str)
                    mention_ids.append(qq_str)

                if (
                    task_type == "at"
                    and quote_sid
                    and str(quote_sid) not in mentioned_ids
                ):
                    mention_ids.append(str(quote_sid))
                    mentioned_ids.add(str(quote_sid))

                used_mention_ids = (
                    set()
                )  # 严格记录已发送过的 QQ，防止任何形式的重复艾特

                def append_mentions() -> bool:
                    appended = False
                    for qq_str in mention_ids:
                        if qq_str in used_mention_ids:
                            continue
                        try:
                            chain.at(name="", qq=qq_str)
                            used_mention_ids.add(qq_str)  # 标记为已使用
                            appended = True
                        except Exception:
                            pass
                    return appended

                seg_tts = seg.get("tts")
                if not isinstance(seg_tts, dict):
                    seg_tts = (
                        resp.get("tts") if isinstance(resp.get("tts"), dict) else {}
                    )
                want_tts = (
                    bool(seg_tts.get("enabled"))
                    if isinstance(seg_tts.get("enabled"), bool)
                    else False
                )
                tts_provider = None
                provider_tts_enable = False
                dual_output = False
                use_file_service = False
                callback_api_base = None
                can_process_tts = False
                try:
                    cfg = self.plugin.context.get_config(umo=umo)
                    provider_tts_settings = (
                        cfg.get("provider_tts_settings", {})
                        if isinstance(cfg, dict)
                        else {}
                    )
                    provider_tts_enable = bool(provider_tts_settings.get("enable"))
                    dual_output = bool(provider_tts_settings.get("dual_output"))
                    use_file_service = bool(
                        provider_tts_settings.get("use_file_service")
                    )
                    callback_api_base = (
                        cfg.get("callback_api_base") if isinstance(cfg, dict) else None
                    )
                except Exception:
                    pass
                if want_tts and provider_tts_enable:
                    try:
                        tts_provider = self.plugin.context.get_using_tts_provider(
                            umo=umo
                        )
                    except Exception:
                        tts_provider = None
                    can_process_tts = bool(
                        tts_provider
                    ) and SessionServiceManager.should_process_tts_request(send_event)
                has_image_url = bool(explicit_images)
                has_any_url = False
                try:
                    has_any_url = bool(
                        re.search(r"https?://", segment_text_for_url_scan)
                    )
                except Exception:
                    has_any_url = False
                if want_tts and has_any_url:
                    want_tts = False
                    can_process_tts = False
                if not has_image_url:
                    try:
                        candidates = re.findall(
                            r"(https?://[^\s]+)", segment_text_for_url_scan
                        )
                        for url in candidates:
                            clean_url = (url or "").strip("`\"'.,;)]} !?")
                            if not clean_url:
                                continue
                            path = urlparse(clean_url).path
                            if any(
                                ext in path.lower()
                                for ext in [
                                    ".jpg",
                                    ".jpeg",
                                    ".png",
                                    ".gif",
                                    ".webp",
                                    ".bmp",
                                ]
                            ) or any(
                                d in clean_url
                                for d in [
                                    "encrypted-tbn0.gstatic.com",
                                    "images.google.com",
                                    "bing.net",
                                    "pximg.net",
                                    "pixiv.net/artworks",
                                ]
                            ):
                                has_image_url = True
                                break
                    except Exception:
                        pass
                if want_tts and has_image_url:
                    want_tts = False
                    can_process_tts = False
                tts_priority_mode = (
                    want_tts
                    and can_process_tts
                    and (not has_image_url)
                    and (not has_any_url)
                )

                async def append_content(prefix: str = ""):
                    if not segment_text_for_msg:
                        return

                    text_to_send = prefix + segment_text_for_msg

                    # 检查是否包含内联艾特占位符，例如 <at:123456> 或 <at:1> (索引)
                    # 我们支持两种格式: <at:QQ> 和 <at:index>
                    inline_at_pattern = r"<at:(\w+)>"
                    parts = re.split(inline_at_pattern, text_to_send)

                    # 如果没有匹配到任何占位符，走原来的逻辑
                    if len(parts) <= 1:
                        if want_tts and can_process_tts and len(text_to_send) > 1:
                            try:
                                tts_input = self._tts_cleanup_text(text_to_send)
                                parts_tts = self._build_tts_parts(tts_input)
                                audio_paths: list[str] = []
                                for kind, value in parts_tts:
                                    if kind == "audio":
                                        audio_paths.append(value)
                                        continue
                                    for chunk in self._split_tts_text(value):
                                        audio_path = await tts_provider.get_audio(chunk)
                                        if audio_path:
                                            audio_paths.append(audio_path)
                                combined_path = (
                                    self._try_concat_wavs(audio_paths)
                                    if audio_paths
                                    else None
                                )
                                record_paths = (
                                    [combined_path] if combined_path else audio_paths
                                )
                                if record_paths:
                                    for record_path in record_paths:
                                        url = None
                                        if use_file_service and callback_api_base:
                                            token = (
                                                await file_token_service.register_file(
                                                    record_path
                                                )
                                            )
                                            url = (
                                                f"{callback_api_base}/api/file/{token}"
                                            )
                                        chain.chain.append(
                                            Record(
                                                file=url or record_path,
                                                url=url or record_path,
                                            )
                                        )
                                    if dual_output:
                                        chain.message(text_to_send)
                                    return
                            except Exception as e:
                                logger.error(f"[mem_chat] TTS failed: {e!s}")
                        chain.message(text_to_send)
                        return

                    # 包含内联艾特，我们需要逐段处理
                    for match in re.finditer(inline_at_pattern, text_to_send):
                        at_val = match.group(1).strip()
                        target_qq = None
                        if at_val.isdigit():
                            idx = int(at_val)
                            if 1 <= idx <= len(mention_ids):
                                target_qq = mention_ids[idx - 1]
                            elif at_candidates and 1 <= idx <= len(at_candidates):
                                target_qq = str(at_candidates[idx - 1][0])
                            elif len(at_val) >= 5:
                                target_qq = at_val
                        else:
                            target_qq = at_val
                        if target_qq and target_qq not in mention_ids:
                            mention_ids.append(target_qq)

                    # 注意：包含内联艾特时，TTS 可能会变得复杂，暂时对带内联艾特的消息回退到纯文本模式
                    for i, part in enumerate(parts):
                        if i % 2 == 0:
                            # 文本部分
                            if part:
                                chain.message(part)
                        else:
                            # 占位符内容 (part 是括号里的内容)
                            at_val = part.strip()
                            target_qq = None

                            # 1. 检查是否已经在 used_mention_ids 中（如果 LLM 在文本中写了两次同一个占位符）
                            if at_val in used_mention_ids:
                                continue

                            # 2. 尝试作为索引 (1-based)
                            if at_val.isdigit():
                                idx = int(at_val)
                                if 1 <= idx <= len(mention_ids):
                                    target_qq = mention_ids[idx - 1]
                                elif at_candidates and 1 <= idx <= len(at_candidates):
                                    # 如果 mention_ids 为空，尝试从系统提供的候选列表中获取
                                    target_qq = str(at_candidates[idx - 1][0])
                                else:
                                    target_qq = at_val  # 视为 QQ
                            else:
                                target_qq = at_val

                            if target_qq:
                                if target_qq in used_mention_ids:
                                    continue  # 再次检查，防止索引指向的 QQ 已经通过 ID 艾特过了
                                try:
                                    chain.at(name="", qq=target_qq)
                                    used_mention_ids.add(target_qq)
                                except Exception:
                                    chain.message(f"@{at_val}")  # 回退

                if quote_id and not tts_priority_mode:
                    reply_kwargs = {"id": str(quote_id)}
                    if quote_sid:
                        reply_kwargs["sender_id"] = quote_sid
                    chain.chain.append(Reply(**reply_kwargs))

                if tts_priority_mode:
                    mention_ids = []
                    emotions_to_send = []

                valid_img_urls = []
                for url in explicit_images:
                    if url and url not in valid_img_urls:
                        valid_img_urls.append(url)

                current_urls = re.findall(
                    r"(https?://[^\s]+)", segment_text_for_url_scan
                )
                for url_match in current_urls:
                    clean_url = url_match.strip("`\"'.,;)]} !?")
                    resolved_url = await self._resolve_social_media_image(clean_url)
                    if resolved_url:
                        clean_url = resolved_url
                    is_valid_img = False
                    try:
                        path = urlparse(clean_url).path
                        if any(
                            ext in path.lower()
                            for ext in [
                                ".jpg",
                                ".jpeg",
                                ".png",
                                ".gif",
                                ".webp",
                                ".bmp",
                            ]
                        ):
                            is_valid_img = True
                        elif any(
                            d in clean_url
                            for d in [
                                "encrypted-tbn0.gstatic.com",
                                "images.google.com",
                                "bing.net",
                                "pximg.net",
                                "pixiv.net/artworks",
                            ]
                        ):
                            is_valid_img = True
                    except Exception:
                        pass
                    if is_valid_img and clean_url not in valid_img_urls:
                        valid_img_urls.append(clean_url)

                downloaded_images = []
                if valid_img_urls:
                    uds = self._get_user_daily_set(event)
                    if uds:
                        valid_img_urls = [
                            u
                            for u in valid_img_urls
                            if self._normalize_image_key(u) not in uds
                        ]
                    random.shuffle(valid_img_urls)

                    # 创建一个临时的 MessageChain 用于下载
                    temp_chain = MessageChain()
                    desired_count = 3
                    try:
                        desired_count = int(
                            seg.get("image_count") or resp.get("image_count") or 0
                        )
                    except Exception:
                        desired_count = 0
                    if desired_count <= 0:
                        desired_count = max(1, len(explicit_images) or 3)
                    desired_count = max(1, min(6, desired_count))

                    for url in valid_img_urls[:desired_count]:
                        success, _is_404 = await self._download_and_append_image(
                            url, temp_chain, temp_files_to_clean, umo
                        )
                        if success:
                            # 如果下载成功，从文本中移除该 URL，避免重复显示
                            try:
                                segment_text_for_msg = segment_text_for_msg.replace(
                                    url, ""
                                ).strip()
                            except Exception:
                                pass
                            # 收集下载成功的图片（从 temp_chain 中提取最后一个 Image 段）
                            if temp_chain.chain and isinstance(
                                temp_chain.chain[-1], Image
                            ):
                                downloaded_images.append(temp_chain.chain[-1])

                # 如果没搜到显式图，尝试从上下文恢复
                if not downloaded_images:
                    temp_chain = MessageChain()
                    if await self._recover_images_from_search_results(
                        llm_contexts,
                        segment_text_for_url_scan,
                        temp_chain,
                        temp_files_to_clean,
                        umo,
                        [],
                    ):
                        for seg in temp_chain.chain:
                            if isinstance(seg, Image):
                                downloaded_images.append(seg)

                if at_position_norm == "suffix":
                    await append_content()
                    append_mentions()
                else:
                    has_mentions = append_mentions()
                    await append_content("\u200e" if has_mentions else "")

                # 最后添加下载好的图片
                for img_seg in downloaded_images:
                    chain.chain.append(img_seg)

                if not tts_priority_mode:
                    unique_emotions = []
                    seen_emo = set()
                    for emo in emotions_to_send:
                        e = (emo or "").strip()
                        if not e:
                            continue
                        if e not in seen_emo:
                            seen_emo.add(e)
                            unique_emotions.append(e)
                    if unique_emotions:
                        meme_file = self._pick_meme_image_file(unique_emotions[0])
                        if meme_file:
                            try:
                                chain.chain.append(Image.fromFileSystem(meme_file))
                            except Exception:
                                pass

                if chain.chain:
                    try:
                        await send_event.send(chain)
                    finally:
                        for p in temp_files_to_clean:
                            try:
                                if p and os.path.exists(p):
                                    os.remove(p)
                            except Exception:
                                pass

            return True

        text = self._sanitize_final_reply(final_reply or "")
        text = self._strip_search_listing_text(text)
        text = (text or "").strip()
        if not text:
            logger.info(
                f"[mem_chat] Sanitized final reply is empty for {umo}. Skipping send but marking as processed."
            )
            return True
        send_event = event if event else DummyEvent(umo, self.plugin.context)
        chain = MessageChain()
        chain.message(text)
        await send_event.send(chain)
        return True

    def _compress_log(self, text: str, length: int = 200) -> str:
        if not text:
            return ""
        s = str(text)
        if length and len(s) > length:
            return s[:length] + "...(truncated)"
        return s

    def _filter_log_lines(self, text: str) -> str:
        if not text:
            return ""
        try:
            raw = str(text).replace("\r\n", "\n").replace("\r", "\n")
            out = []
            for ln in raw.split("\n"):
                s = ln.rstrip()
                if not s.strip():
                    out.append("")
                    continue
                if re.match(
                    r"^\[\d{2}:\d{2}:\d{2}\]\s+\[[^\]]+\]\s+\[[A-Z]+\]", s.strip()
                ):
                    continue
                if s.strip().startswith("[mem_chat]"):
                    continue
                if "src.llm_client:" in s or "src.scheduler:" in s:
                    continue
                out.append(s)
            return "\n".join(out)
        except Exception:
            return str(text)

    def _dedupe_section_blocks(self, text: str) -> str:
        if not text:
            return ""
        try:
            raw = str(text).replace("\r\n", "\n").replace("\r", "\n")
            lines = raw.split("\n")
            blocks: list[tuple[str | None, list[str]]] = []
            curr_h: str | None = None
            curr: list[str] = []
            for ln in lines:
                m = re.match(r"^\s*(【[^】]{1,40}】)\s*$", ln)
                if m:
                    if curr_h is not None or curr:
                        blocks.append((curr_h, curr))
                    curr_h = m.group(1)
                    curr = [curr_h]
                else:
                    curr.append(ln)
            if curr_h is not None or curr:
                blocks.append((curr_h, curr))

            merged_blocks: dict[str, list[str]] = {}
            no_header_blocks: list[list[str]] = []

            for h, b in blocks:
                if h is None:
                    no_header_blocks.append(b)
                else:
                    if h not in merged_blocks:
                        merged_blocks[h] = []
                    # 只有当内容不重复时才添加
                    for line in b:
                        if line != h:  # 标题本身已经在 key 里了
                            merged_blocks[h].append(line)

            out_lines: list[str] = []
            # 先放没有标题的
            for b in no_header_blocks:
                out_lines.extend(b)
            # 再放有标题的
            for h, content in merged_blocks.items():
                out_lines.append(h)
                out_lines.extend(content)
                out_lines.append("")  # 块之间留空

            return "\n".join(out_lines).strip()
        except Exception:
            return str(text)

    def _redact_inline_image_data(self, text: str) -> str:
        if not text:
            return ""
        try:
            return re.sub(
                r"(data:image/[^;]+;base64,)[A-Za-z0-9+/=\s]{200,}",
                lambda m: m.group(1)
                + f"<omitted {len(m.group(0)) - len(m.group(1))} chars>",
                str(text),
                flags=re.IGNORECASE,
            )
        except Exception:
            return str(text)

    def _redact_inline_image_data_obj(self, obj):
        if obj is None:
            return None
        if isinstance(obj, str):
            return self._redact_inline_image_data(obj)
        if isinstance(obj, list):
            return [self._redact_inline_image_data_obj(i) for i in obj]
        if isinstance(obj, tuple):
            return [self._redact_inline_image_data_obj(i) for i in obj]
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                out[k] = self._redact_inline_image_data_obj(v)
            return out
        return obj

    async def invoke_llm(
        self,
        umo: str,
        prompt: str,
        memory_context: str,
        event: AstrMessageEvent | None = None,
        task_type: str = "at",
        instruction_text: str = "",
        user_msgs_to_save: list[dict] = None,
        image_urls: list[str] = None,
        quote_candidates: list = None,
        at_candidates: list = None,
        created_at: float = 0,
    ) -> bool:
        """Invoke LLM manually and send reply. Returns True if a reply was sent."""

        # 1. Timeout Check
        timeout_sec = self._get_task_timeout_sec(task_type)
        if (
            created_at > 0
            and timeout_sec > 0
            and time.time() - created_at > timeout_sec
        ):
            logger.warning(
                f"[mem_chat] Dropping LLM request for {umo} due to timeout before invocation (waited {time.time() - created_at:.2f}s, timeout={timeout_sec:.0f}s).",
            )
            return False

        # 2. Config
        llm_config = self._prepare_llm_config(umo, event)
        if not llm_config["provider"] and not llm_config["use_custom_llm"]:
            logger.warning(
                f"[mem_chat] No LLM provider found for {umo} and no custom LLM configured."
            )
            return False

        # 3. Context
        conv_info = await self._prepare_conversation_context(
            umo, memory_context, user_msgs_to_save, event
        )
        intents = self._detect_intents(umo, prompt, user_msgs_to_save, task_type, event)

        plan_keywords: list[str] = []
        plan_count = 1
        if task_type == "at" and intents.get("image"):
            # Use the whole message block (full_text) for extraction, but prioritize showing the AT part context
            # By passing the full_block_text, the LLM has more context to understand what kind of image to search.
            text_for_plan = intents.get("full_text") or intents.get("last_text")
            if text_for_plan:
                try:
                    plan_keywords, plan_count = await self._extract_pixiv_search_plan(
                        text_for_plan,
                        conv_info.get("history") or [],
                        llm_config,
                        umo,
                    )
                except Exception:
                    plan_keywords, plan_count = [], 1
                if plan_keywords:
                    intents["search_keyword"] = plan_keywords[0]
                    logger.info(
                        f"[mem_chat] Pixiv search plan keywords={plan_keywords} count={plan_count} (source: message block)"
                    )

        # Removed refined keyword extraction for non-AT tasks to keep it simple.
        # Rule-based search_keyword from _detect_intents will be used below in Plan A.

        tools, has_search_tools = self._setup_tools(umo, intents, conv_info["persona"])

        if task_type == "at" and intents.get("image") and plan_keywords:
            all_lines: list[str] = []
            for kw in plan_keywords:
                try:
                    results = await self._proactive_pixiv_search(
                        kw, umo, max_results=max(6, plan_count * 3)
                    )
                except Exception:
                    results = []
                if results:
                    all_lines.append(f"【关键词：{kw}】")
                    all_lines.extend(results[: max(6, plan_count * 3)])
                    all_lines.append("")
            if all_lines:
                pixiv_ctx = (
                    "\n\n【Pixiv 实时搜图结果】\n" + "\n".join(all_lines).strip()
                )
                memory_context += pixiv_ctx
                if not instruction_text:
                    instruction_text = ""
                instruction_text += (
                    f"\n系统已自动为你搜索了相关图片。请从【Pixiv 实时搜图结果】中选择最合适的 {plan_count} 张图片 URL。"
                    "将 URL 放入 response.images 数组（必须是字符串数组），并在 response.text 中写一句简短回复。"
                    "不要在 response.text 中粘贴多余的 URL。"
                )
        elif intents.get("image") and intents.get("search_keyword"):
            # Plan A: Rule-based Proactive Search
            # Allowed in FOCUS mode (Lightweight)
            # Not allowed in READ_AIR

            should_search = False
            if task_type == "focus":
                should_search = True
            elif task_type == "at":
                # Should have been handled by Plan B above, but if Plan B failed or condition missed,
                # fallback to Plan A is acceptable for AT tasks too.
                should_search = True

            if should_search:
                keyword_raw = intents["search_keyword"]
                keyword = self._clean_image_search_query(keyword_raw)
                if keyword != (keyword_raw or ""):
                    logger.info(
                        f"[mem_chat] Proactive search keyword cleaned: {keyword_raw} -> {keyword}"
                    )
                pixiv_results = []
                if keyword:
                    logger.info(
                        f"[mem_chat] Proactive search triggered for keyword: {keyword}"
                    )
                    pixiv_results = await self._proactive_pixiv_search(keyword, umo)
                if pixiv_results:
                    pixiv_ctx = "\n\n【Pixiv 实时搜图结果】\n" + "\n".join(
                        pixiv_results
                    )
                    memory_context += pixiv_ctx
                    if not instruction_text:
                        instruction_text = ""
                    instruction_text += (
                        "\n系统已自动为你搜索了相关图片。请从【Pixiv 实时搜图结果】中选择最合适的 1-2 张图片 URL。"
                        "将 URL 放入 response.images 数组（必须是字符串数组），并在 response.text 中写一句简短回复。"
                        "不要在 response.text 中粘贴多余的 URL。"
                    )
            else:
                logger.debug(
                    f"[mem_chat] Proactive search (Plan A) skipped for task_type={task_type}."
                )

        # 4. Instructions
        instructions = self._build_instructions(
            "", quote_candidates, at_candidates, image_urls, intents, tools
        )
        system_prompt = self._build_system_prompts(
            conv_info["persona"], memory_context, umo
        )

        # 注入动态指令到 prompt 中（如搜图结果处理说明）
        if instruction_text:
            prompt += f"\n\n{instruction_text}"

        # 5. Execution (Simplified)
        llm_contexts = []  # 保持为空，依赖 memory_context 中的【最近对话】和【人物设定】

        start_time = time.time()

        # Logging Config
        log_llm = self._get_bool_conf("log_llm_prompt", True)
        log_llm_full = self._get_bool_conf("log_llm_prompt_full", False)
        log_llm_ctx_full = self._get_bool_conf("log_llm_contexts_full", False)

        if self.plugin.state.check_interrupt(umo, start_time):
            return False

        # Determine Prompt & LLM
        if task_type == "at":
            decision_instructions = (
                "【决策（AT 专用）】\n"
                "本轮为显式 @ 消息，请使用结构化 JSON 输出，并在 control.decision 中填入：\n"
                '1) 直接回复：control.decision = "reply_only"，response.text 写给用户看的回复。\n'
                "请始终返回一个包含 response 和 control 字段的 JSON 对象。"
            )
        else:
            decision_instructions = (
                "【决策】\n"
                "请在对话阶段进行二选一，并使用结构化 JSON 输出：\n"
                '1) 不回复：将 control.decision 设为 "no_reply"，response.text 可留空或写给自己看的备注。\n'
                '2) 直接回复：将 control.decision 设为 "reply_only"，response.text 写给用户看的最终回复内容。\n'
                "请始终返回一个包含 response 和 control 字段的 JSON 对象。"
            )

        prompt_to_use = (
            f"{prompt}\n\n{instructions['suffix']}\n\n{decision_instructions}"
        )
        sys_prompt_to_use = system_prompt
        tools_to_use = tools if (tools and getattr(tools, "tools", None)) else None
        if tools_to_use and hasattr(tools_to_use, "tools") and not tools_to_use.tools:
            tools_to_use = None

        prompt_to_use = self._dedupe_section_blocks(
            self._filter_log_lines(prompt_to_use)
        )
        sys_prompt_to_use = self._dedupe_section_blocks(
            self._filter_log_lines(sys_prompt_to_use)
        )

        # Call LLM
        # Select Config
        cfg = llm_config["llm"] if llm_config["use_custom_llm"] else None

        if log_llm:
            if log_llm_full:
                logger.info(
                    f"[mem_chat] LLM PromptFull: {self._redact_inline_image_data(prompt_to_use)}"
                )
                logger.info(
                    f"[mem_chat] LLM SystemFull: {self._redact_inline_image_data(sys_prompt_to_use)}"
                )
            if log_llm_ctx_full:
                try:
                    safe_ctx = self._redact_inline_image_data_obj(llm_contexts)
                    logger.info(
                        f"[mem_chat] LLM ContextsFull: {json.dumps(safe_ctx, ensure_ascii=False)}"
                    )
                except Exception:
                    logger.info("[mem_chat] LLM ContextsFull: <unserializable>")

        response = None
        step_count = 0
        max_steps = 3
        final_payload_to_send = None
        control_decision = None
        final_reply = ""

        while step_count < max_steps:
            # Check interrupt before each LLM step
            if self.plugin.state.check_interrupt(umo, start_time):
                logger.info(
                    f"[mem_chat] LLM invocation interrupted for {umo} at step {step_count}."
                )
                return False

            response = None
            step_prompt = (
                prompt_to_use
                if step_count == 0
                else ("请基于以上工具返回结果组织最终回复，并使用结构化 JSON 输出。")
            )
            if llm_config["use_custom_llm"]:
                llm_task = asyncio.create_task(
                    self._call_custom_llm(
                        base_url=cfg["base"],
                        api_key=cfg["key"],
                        model=cfg["model"],
                        system_prompt=sys_prompt_to_use,
                        prompt=step_prompt,
                        contexts=llm_contexts,
                        tools=tools_to_use,
                        image_urls=image_urls if step_count == 0 else [],
                        temperature=cfg["temp"],
                        fmt=cfg.get("fmt"),
                        tag="llm",
                    )
                )
                try:
                    response = await asyncio.wait_for(
                        asyncio.shield(llm_task), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[mem_chat] LLM timeout. Type: {task_type}")
                    return False
            else:
                if not llm_config["provider"]:
                    return False
                try:
                    provider = llm_config["provider"]
                    response = await provider.text_chat(
                        prompt=step_prompt,
                        system_prompt=sys_prompt_to_use,
                        session_id=umo,
                        contexts=llm_contexts,
                        func_tool=tools_to_use,
                    )
                except Exception as e:
                    logger.error(f"[mem_chat] Provider error: {e}")
                    break

            if not response:
                break

            final_reply = response.completion_text or ""
            if log_llm:
                logger.info(
                    f"[mem_chat] LLM Step {step_count} RawCompletion: {self._compress_log(final_reply, length=4000)}",
                )

            structured = None
            if final_reply:
                try:
                    structured = StructuredResponse.extract_json_from_text(final_reply)
                except Exception:
                    structured = None

            # Detect tool calls
            tool_calls = []
            # 1. Native tool calls
            if response.tools_call_name:
                for name, args, tid in zip(
                    response.tools_call_name,
                    response.tools_call_args,
                    response.tools_call_ids,
                ):
                    tool_calls.append({"name": name, "args": args, "id": tid})

            # 2. JSON tool calls (Talker protocol)
            if structured and isinstance(structured, dict):
                resp_block = structured.get("response")
                tc = structured.get("tool_call")
                if not tc and isinstance(resp_block, dict):
                    tc = resp_block.get("tool_call")
                if isinstance(tc, dict) and tc.get("name"):
                    name = tc["name"]
                    if not any(t["name"] == name for t in tool_calls):
                        tool_calls.append(
                            {
                                "name": name,
                                "args": tc.get("args") or {},
                                "id": "json_" + uuid.uuid4().hex[:8],
                            }
                        )

            if not tool_calls:
                # No more tools, this is the final response
                structured_valid = StructuredResponse.parse_validated(
                    structured or final_reply
                )
                if structured_valid:
                    final_payload_to_send = structured_valid
                    resp_block = structured_valid.get("response") or {}
                    text_field = resp_block.get("text")
                    if isinstance(text_field, str):
                        final_reply = text_field.strip()
                    else:
                        segs = resp_block.get("segments")
                        if isinstance(segs, list) and segs:
                            parts_text = []
                            for seg in segs:
                                if (
                                    isinstance(seg, dict)
                                    and isinstance(seg.get("text"), str)
                                    and seg.get("text").strip()
                                ):
                                    parts_text.append(seg.get("text").strip())
                            final_reply = "\n".join(parts_text).strip()
                        else:
                            final_reply = ""
                    control_block = structured_valid.get("control") or {}
                    control_decision = control_block.get("decision")
                else:
                    final_payload_to_send = final_reply
                    control_decision = "reply_only"  # Fallback
                break

            if not tools_to_use:
                tool_calls = []
                structured_valid = StructuredResponse.parse_validated(
                    structured or final_reply
                )
                if structured_valid:
                    final_payload_to_send = structured_valid
                    resp_block = structured_valid.get("response") or {}
                    text_field = resp_block.get("text")
                    if isinstance(text_field, str):
                        final_reply = text_field.strip()
                    else:
                        segs = resp_block.get("segments")
                        if isinstance(segs, list) and segs:
                            parts_text = []
                            for seg in segs:
                                if isinstance(seg, dict) and seg.get("text"):
                                    parts_text.append(seg.get("text").strip())
                            final_reply = "\n".join(parts_text).strip()
                        else:
                            final_reply = ""
                    control_block = structured_valid.get("control") or {}
                    control_decision = control_block.get("decision")
                else:
                    final_payload_to_send = final_reply
                    control_decision = "reply_only"
                break

            # Execute tools
            tool_results = []
            allowed_tool_names: set[str] = set()
            if tools_to_use and getattr(tools_to_use, "tools", None):
                allowed_tool_names = {
                    t.name for t in tools_to_use.tools if getattr(t, "name", None)
                }

            for tc in tool_calls:
                name = tc["name"]
                args = tc["args"]
                tid = tc["id"]

                logger.info(f"[mem_chat] Executing tool: {name} with args: {args}")

                tmgr = self.plugin.context.get_llm_tool_manager()
                func = tmgr.get_func(name)

                result_text = ""
                if not allowed_tool_names:
                    result_text = "Error: No tools enabled."
                elif name not in allowed_tool_names:
                    result_text = f"Error: Tool {name} is not allowed."
                elif func and func.handler:
                    try:
                        tool_event = event if event else DummyEvent(umo, self.plugin.context)
                        result_text = await self._execute_tool_handler(
                            func.handler, tool_event, args
                        )
                    except Exception as e:
                        logger.error(f"[mem_chat] Tool {name} execution error: {e}")
                        result_text = f"Error: {str(e)}"
                else:
                    result_text = f"Error: Tool {name} not found or no handler."

                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tid,
                        "name": name,
                        "content": result_text,
                    }
                )

            # Add assistant message and tool results to contexts
            ctx_assistant = {"role": "assistant", "content": final_reply}
            if response.raw_completion:
                if (
                    isinstance(response.raw_completion, list)
                    and len(response.raw_completion) > 0
                ):
                    ctx_assistant["tool_calls"] = response.raw_completion

            llm_contexts.append(ctx_assistant)
            llm_contexts.extend(tool_results)
            step_count += 1

        if not response:
            logger.warning(f"[mem_chat] LLM invocation failed (no response) for {umo}.")
            return False

        if log_llm:
            log_reply = re.sub(
                r"(?i)\s*\bTTS\s*[:：]\s*(yes|no)\b",
                "",
                final_reply or "",
            ).strip()
            logger.info(f"[mem_chat] LLM Final Response: {log_reply}")

        ctx_reply_for_history = re.sub(
            r"(?i)\s*\bTTS\s*[:：]\s*(yes|no)\b",
            "",
            final_reply or "",
        ).strip()
        ctx_entry = {"role": "assistant", "content": ctx_reply_for_history}
        llm_contexts.append(ctx_entry)

        if not control_decision:
            logger.warning(
                f"[mem_chat] LLM response is not valid structured JSON (strict mode). Raw: {final_reply[:200]}"
            )
            return False
        if control_decision == "no_reply":
            logger.info(f"[mem_chat] LLM decided not to reply (no_reply) for {umo}.")
            return True  # Return True so scheduler clears the processed messages

        # Check interrupt before sending final reply
        if self.plugin.state.check_interrupt(umo, start_time):
            logger.info(
                f"[mem_chat] LLM invocation interrupted for {umo} before sending final reply."
            )
            return False

        # 7. Final Reply Processing
        send_ok = await self._process_and_send_final_reply(
            final_payload_to_send if final_payload_to_send is not None else final_reply,
            event,
            task_type,
            quote_candidates,
            at_candidates,
            llm_contexts,
            umo,
            user_msgs_to_save,
        )

        if send_ok and ctx_reply_for_history:
            try:
                conv_mgr = self.plugin.context.conversation_manager
                cid = None
                try:
                    conversation = (
                        conv_info.get("conversation")
                        if isinstance(conv_info, dict)
                        else None
                    )
                    cid = getattr(conversation, "cid", None)
                except Exception:
                    cid = None

                if not cid:
                    cid = await conv_mgr.get_curr_conversation_id(umo)

                if cid:
                    conv = await conv_mgr.get_conversation(umo, cid)
                    if conv:
                        try:
                            history = json.loads(conv.history) if conv.history else []
                        except Exception:
                            history = []

                        self_id = ""
                        try:
                            if event:
                                self_id = str(event.get_self_id() or "")
                        except Exception:
                            self_id = ""
                        if not self_id:
                            try:
                                self_id = str(self.plugin.state.get_self_id(umo) or "")
                            except Exception:
                                self_id = ""

                        self_name = ""
                        try:
                            self_name = str(
                                self.plugin.state.get_self_name(umo) or ""
                            ).strip()
                        except Exception:
                            self_name = ""
                        sender_name = self_name or "Assistant"

                        history.append(
                            {
                                "role": "assistant",
                                "content": ctx_reply_for_history,
                                "sender_name": sender_name,
                                "sender_id": self_id,
                            },
                        )

                        try:
                            history_count = int(
                                self._get_conf("history_message_count", 12)
                            )
                        except Exception:
                            history_count = 12
                        if history_count > 0 and len(history) > history_count:
                            history = history[-history_count:]

                        await conv_mgr.update_conversation(umo, cid, history)
                        logger.info(
                            f"[mem_chat] Appended assistant reply to conversation history. cid={cid} len={len(history)}",
                        )

                        try:
                            gid = ""
                            if event:
                                gid = event.get_group_id() or event.get_sender_id()
                            if not gid:
                                gid = umo.split(":")[-1] if ":" in umo else umo
                            sender_info = sender_name
                            if self_id and sender_name:
                                sender_info = f"{sender_name}/{self_id}"
                            asyncio.create_task(
                                self.plugin.db_mgr.ensure_message_embedded(
                                    ctx_reply_for_history,
                                    gid,
                                    sender_info,
                                ),
                            )
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(
                    f"[mem_chat] Failed to append assistant reply to history: {e}"
                )

        return send_ok
