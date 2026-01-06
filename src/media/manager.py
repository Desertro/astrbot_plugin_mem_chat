import asyncio
import base64
import io
import logging
import os
import re
import tempfile
import ipaddress
import socket
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import aiohttp
from PIL import Image as PILImage

from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import At, AtAll, Face, Image, Plain, Reply

from .image_vision_cache import ImageVisionCache

logger = logging.getLogger("astrbot")


class MediaManager:
    def __init__(self, config: dict, *, db_path: str | None = None, get_conf=None):
        self.config = config
        self._get_conf = get_conf or (lambda k, d: self.config.get(k, d))
        self._image_cache: ImageVisionCache | None = None
        if db_path:
            try:
                self._image_cache = ImageVisionCache(db_path, self._get_conf)
            except Exception as e:
                logger.error(f"[mem_chat] Failed to init image cache: {e}")
                self._image_cache = None

    def close(self):
        if self._image_cache:
            try:
                self._image_cache.close()
            except Exception:
                pass
            self._image_cache = None

    def _is_safe_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return False
            
            # Resolve hostname to IP
            try:
                ip_list = socket.getaddrinfo(hostname, None)
                ips = [ip[4][0] for ip in ip_list]
            except socket.gaierror:
                return False

            for ip_str in ips:
                try:
                    ip = ipaddress.ip_address(ip_str)
                    if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                        logger.warning(f"[mem_chat] Blocked unsafe IP: {ip_str} for URL: {url}")
                        return False
                except ValueError:
                    continue
            return True
        except Exception as e:
            logger.error(f"[mem_chat] Failed to validate URL {url}: {e}")
            return False

    async def _compress_image(self, data: bytes, max_size_kb: int = 500) -> tuple[bytes, str]:
        """Compress image data to be under max_size_kb. Returns (compressed_data, mime_type)."""
        return await asyncio.to_thread(self._compress_image_sync, data, max_size_kb)

    def _compress_image_sync(self, data: bytes, max_size_kb: int = 500) -> tuple[bytes, str]:
        """Synchronous implementation of image compression."""
        mime_type = "image/jpeg"
        if data.startswith(b"\x89PNG"):
            mime_type = "image/png"
        elif data.startswith(b"GIF8"):
            mime_type = "image/gif"
        elif data.startswith(b"RIFF") and b"WEBP" in data[:12]:
            mime_type = "image/webp"

        if len(data) <= max_size_kb * 1024:
            return data, mime_type

        try:
            img_io = io.BytesIO(data)
            img = PILImage.open(img_io)

            img_format = img.format if img.format else "JPEG"

            if img_format == "GIF":
                logger.info(
                    f"[mem_chat] GIF detected ({len(data)} bytes). Converting to JPEG for compression."
                )
                img.seek(0)
                img = img.convert("RGB")
                img_format = "JPEG"
                mime_type = "image/jpeg"

            quality = 85
            output = io.BytesIO()

            img.save(output, format=img_format, quality=quality)

            while output.tell() > max_size_kb * 1024 and quality > 30:
                quality -= 10
                output = io.BytesIO()
                img.save(output, format=img_format, quality=quality)

            while output.tell() > max_size_kb * 1024:
                width, height = img.size
                if width < 200 or height < 200:
                    break
                img = img.resize(
                    (int(width * 0.7), int(height * 0.7)), PILImage.Resampling.LANCZOS
                )
                output = io.BytesIO()
                img.save(output, format=img_format, quality=quality)

            new_data = output.getvalue()
            if img_format == "JPEG":
                mime_type = "image/jpeg"
            elif img_format == "PNG":
                mime_type = "image/png"
            elif img_format == "WEBP":
                mime_type = "image/webp"

            logger.info(
                f"[mem_chat] Compressed image from {len(data)} to {len(new_data)} bytes (quality={quality}, size={img.size}, mime={mime_type})"
            )
            return new_data, mime_type
        except Exception as e:
            logger.error(f"[mem_chat] Failed to compress image: {e}")
            return data, mime_type

    @asynccontextmanager
    async def download_image_context(self, url: str):
        """Context manager to download image and automatically clean up temp file."""
        path = await self.download_image(url)
        try:
            yield path
        finally:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.debug(f"[mem_chat] Cleaned up temp image: {path}")
                except Exception as e:
                    logger.warning(f"[mem_chat] Failed to cleanup temp image {path}: {e}")

    async def download_image(self, url: str) -> str | None:
        """Download image to temp file and return path. Caller is responsible for cleanup."""
        try:
            url = url.strip("`\"' .,;:)")
            
            if not self._is_safe_url(url):
                logger.warning(f"[mem_chat] Blocked unsafe URL: {url}")
                return None

            logger.info(
                f"[mem_chat] Downloading image: {url} | System environment proxy enabled"
            )

            parsed_uri = urlparse(url)
            domain = f"{parsed_uri.scheme}://{parsed_uri.netloc}/"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Referer": domain,
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            }

            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.get(url, headers=headers, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        data, mime_type = await self._compress_image(data)
                    else:
                        logger.warning(
                            f"[mem_chat] Failed to download image {url}: {resp.status}"
                        )
                        return None

                    suffix = ".jpg"
                    if "png" in mime_type:
                        suffix = ".png"
                    elif "gif" in mime_type:
                        suffix = ".gif"
                    elif "webp" in mime_type:
                        suffix = ".webp"

                    fd, path = tempfile.mkstemp(suffix=suffix)

                    with os.fdopen(fd, "wb") as f:
                        f.write(data)

                    logger.info(f"[mem_chat] Downloaded image to {path}")
                    return path
        except Exception as e:
            logger.error(f"[mem_chat] Error downloading image {url}: {e}")
            return None

    async def image_to_base64(self, file_path_or_url: str) -> str | None:
        file_path_or_url = file_path_or_url.strip()
        if file_path_or_url.startswith("`") and file_path_or_url.endswith("`"):
            file_path_or_url = file_path_or_url.strip("`").strip()

        if file_path_or_url.startswith("http"):
            try:
                async with aiohttp.ClientSession(trust_env=True) as session:
                    async with session.get(file_path_or_url, timeout=20) as resp:
                        if resp.status == 200:
                            content = await resp.read()
                            content, mime = self._compress_image(content)
                            b64 = base64.b64encode(content).decode("utf-8")
                            return f"data:{mime};base64,{b64}"
                        logger.warning(
                            f"[mem_chat] Failed to download image: {resp.status}"
                        )
                        return file_path_or_url
            except Exception as e:
                logger.error(f"[mem_chat] Error downloading image: {e}")
                return file_path_or_url

        if os.path.exists(file_path_or_url):
            try:
                with open(file_path_or_url, "rb") as f:
                    content = f.read()
                    content, mime = self._compress_image(content)
                    return f"data:{mime};base64,{base64.b64encode(content).decode('utf-8')}"
            except Exception as e:
                logger.error(f"[mem_chat] Failed to encode image: {e}")
                return None
        return None

    async def call_vision_llm(self, image_data: str) -> str:
        v_conf = self.config.get("vision_llm", {})
        if not v_conf.get("enable", False):
            return ""

        api_key = v_conf.get("api_key", "")
        base_url = v_conf.get("base_url", "https://api.openai.com/v1").rstrip("/")
        format_type = v_conf.get("format", "openai")
        model = v_conf.get("model", "gpt-4o-mini")
        prompt_text = v_conf.get("prompt", "请简要描述这张图片的内容，50字以内。")

        if format_type == "google":
            if "/v1/chat/completions" in base_url:
                base_url = base_url.replace("/v1/chat/completions", "")

            mime_type = "image/jpeg"
            b64_data = image_data
            if "," in image_data:
                header, b64_data = image_data.split(",", 1)
                mime_type = header.split(":")[1].split(";")[0]

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt_text},
                            {"inline_data": {"mime_type": mime_type, "data": b64_data}},
                        ]
                    }
                ],
                "generationConfig": {"maxOutputTokens": 4096, "temperature": 0.4},
            }
            url = f"{base_url}/v1beta/models/{model}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
        else:
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": image_data}},
                        ],
                    }
                ],
                "max_tokens": 4096,
            }
            url = f"{base_url}/chat/completions"

        try:
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.post(
                    url, headers=headers, json=payload, timeout=120
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if format_type == "google":
                            candidates = data.get("candidates", [])
                            if candidates and candidates[0].get("content", {}).get(
                                "parts"
                            ):
                                content = candidates[0]["content"]["parts"][0].get(
                                    "text", ""
                                )
                            else:
                                logger.error(
                                    f"[mem_chat] Vision Google response error: {data}"
                                )
                                return ""
                        else:
                            choices = data.get("choices", [])
                            if not choices:
                                logger.error(
                                    f"[mem_chat] Vision LLM response has no choices: {data}"
                                )
                                return ""
                            content = choices[0]["message"].get("content", "")

                        if not content:
                            logger.warning(
                                f"[mem_chat] Vision LLM returned empty content. Data: {data}"
                            )
                            content = ""
                        logger.info(f"[mem_chat] Vision LLM success: {content}")
                        return f"[图片内容: {content}]"
                    text = await resp.text()
                    logger.error(
                        f"[mem_chat] Vision LLM failed: {resp.status} - {text}"
                    )
        except Exception as e:
            logger.error(f"[mem_chat] Vision LLM error: {e}")

        return ""

    async def extract_images_from_event(self, event: AstrMessageEvent) -> list[str]:
        """Extract images from event message chain and return as URLs or file paths."""
        images = []
        try:
            if (
                hasattr(event, "message_obj")
                and event.message_obj
                and hasattr(event.message_obj, "message")
            ):
                chain = event.message_obj.message
                for i in chain:
                    if isinstance(i, Image):
                        img_src = i.url if i.url else i.file
                        if img_src:
                            if img_src.startswith("`") and img_src.endswith("`"):
                                img_src = img_src.strip("`").strip()
                            images.append(img_src)
        except Exception as e:
            logger.error(f"[mem_chat] Error extracting images: {e}")
        return images

    async def get_text_with_images(self, event: AstrMessageEvent) -> str:
        """Helper to extract text and replace images with [图片] or description"""
        try:
            if (
                hasattr(event, "message_obj")
                and event.message_obj
                and hasattr(event.message_obj, "message")
            ):
                chain = event.message_obj.message
                if not chain:
                    return event.message_str or ""

                parts = []
                is_at = bool(getattr(event, "is_at_or_wake_command", False))
                group_id = str(event.get_group_id() or "").strip()
                sender_id = str(event.get_sender_id() or "").strip() or None
                msg_id = None
                try:
                    msg_obj = getattr(event, "message_obj", None)
                    if msg_obj:
                        msg_id = getattr(msg_obj, "message_id", None)
                except Exception:
                    msg_id = None

                for i in chain:
                    if isinstance(i, Plain):
                        parts.append(i.text)
                    elif isinstance(i, Image):
                        v_conf = self.config.get("vision_llm", {})

                        img_src = i.url if i.url else i.file
                        url = i.url
                        file_token = i.file

                        is_command = False
                        plain_text_content = "".join(
                            [x.text for x in chain if isinstance(x, Plain)]
                        ).strip()
                        ignore_prefixes = (
                            "#",
                            "/",
                            "＃",
                            "bnn",
                            "手办化",
                            "贴纸化",
                            "棉花娃娃化",
                            "服装分解",
                            "模型套件",
                            "角色界面",
                            "角色设定",
                            "钥匙扣",
                            "姿势表",
                            "转身动画",
                            "多机位",
                            "电影分镜",
                            "动漫分镜",
                            "真人化",
                            "场景真人化",
                            "半真人半动漫",
                            "元素融合",
                            "猫娘化",
                            "coser化",
                            "痛房化",
                            "痛车化",
                            "地雷女化",
                        )
                        if plain_text_content.startswith(ignore_prefixes):
                            is_command = True

                        chat_model_supports_vision = self.config.get(
                            "chat_model_supports_vision", False
                        )
                        should_run_vision = False

                        if v_conf.get("enable", False) and not is_command and img_src:
                            if is_at:
                                if not chat_model_supports_vision:
                                    should_run_vision = True
                            else:
                                should_run_vision = True

                        image_key = None
                        cached_caption = None
                        claimed_caption_gen = False
                        if self._image_cache and group_id:
                            try:
                                image_key, _mime = self._image_cache.record_seen(
                                    group_id=group_id,
                                    url=url,
                                    file_token=file_token,
                                    sender_id=sender_id,
                                    msg_id=str(msg_id) if msg_id is not None else None,
                                    is_at=is_at,
                                )
                                if not is_at:
                                    cached_caption = self._image_cache.get_caption(
                                        group_id=group_id,
                                        image_key=image_key,
                                    )
                            except Exception:
                                image_key = None
                                cached_caption = None

                        if cached_caption:
                            parts.append(cached_caption)
                        elif should_run_vision:
                            if self._image_cache and group_id and image_key and not is_at:
                                try:
                                    claimed_caption_gen = (
                                        self._image_cache.claim_caption_generation(
                                            group_id=group_id,
                                            image_key=image_key,
                                        )
                                    )
                                    if not claimed_caption_gen:
                                        waited = self._image_cache.wait_caption_ready(
                                            group_id=group_id,
                                            image_key=image_key,
                                            timeout_sec=2.5,
                                            interval_sec=0.2,
                                        )
                                        if waited:
                                            parts.append(waited)
                                            continue
                                except Exception:
                                    claimed_caption_gen = False
                            img_data = await self.image_to_base64(img_src)
                            if img_data:
                                desc = await self.call_vision_llm(img_data)
                                if desc:
                                    parts.append(desc)
                                    if (
                                        self._image_cache
                                        and group_id
                                        and image_key
                                        and not is_at
                                    ):
                                        try:
                                            self._image_cache.set_caption(
                                                group_id=group_id,
                                                image_key=image_key,
                                                caption=desc,
                                                model=str(v_conf.get("model") or "").strip()
                                                or None,
                                            )
                                        except Exception:
                                            pass
                                else:
                                    if (
                                        claimed_caption_gen
                                        and self._image_cache
                                        and group_id
                                        and image_key
                                        and not is_at
                                    ):
                                        try:
                                            self._image_cache.set_caption(
                                                group_id=group_id,
                                                image_key=image_key,
                                                caption="",
                                                model=str(v_conf.get("model") or "").strip()
                                                or None,
                                                status="ready",
                                                err="vision_empty",
                                            )
                                        except Exception:
                                            pass
                                    parts.append("[图片]")
                            else:
                                if (
                                    claimed_caption_gen
                                    and self._image_cache
                                    and group_id
                                    and image_key
                                    and not is_at
                                ):
                                    try:
                                        self._image_cache.set_caption(
                                            group_id=group_id,
                                            image_key=image_key,
                                            caption="",
                                            model=str(v_conf.get("model") or "").strip()
                                            or None,
                                            status="ready",
                                            err="image_to_base64_failed",
                                        )
                                    except Exception:
                                        pass
                                parts.append("[图片]")
                        else:
                            parts.append("[图片]")

                    elif isinstance(i, At):
                        parts.append(f"@{i.qq}")
                    elif isinstance(i, AtAll):
                        parts.append("@全体成员")
                    elif isinstance(i, Face):
                        parts.append("[表情]")
                    elif isinstance(i, Reply):
                        reply_text = i.message_str or "[引用消息]"
                        parts.append(
                            f"[回复 {i.sender_nickname or '未知'}: {reply_text}] "
                        )
                    else:
                        pass
                result_text = "".join(parts).strip()
                result_text = re.sub(r"\[图片\]", "", result_text)
                return result_text.strip()
            return re.sub(r"\[图片\]", "", event.message_str or "").strip()
        except Exception as e:
            logger.error(f"[mem_chat] Error parsing text with images: {e}")
            return re.sub(r"\[图片\]", "", event.message_str or "").strip()


__all__ = ["MediaManager"]
