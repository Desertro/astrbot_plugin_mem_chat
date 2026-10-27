import asyncio
import json
import os
import re
import sys
import time

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import (
    At,
    AtAll,
    Image,
    Json,
    Node,
    Nodes,
    Plain,
    Poke,
    Reply,
)
from astrbot.core.message.message_event_result import MessageEventResult

BLOCKED_KNOWLEDGE_SOURCE_TOKENS = (
    "bbs.colg.cn",
    "colg",
)
BLOCKED_PATCH_TOKENS = (
    "补丁",
    "patch",
    "想定制补丁",
    "vx",
    "qq3287944904",
)

try:
    from .src.core.models import DummyEvent
    from .src.core.scheduler import Scheduler
    from .src.core.state import MemChatState
    from .src.db.database import DatabaseManager
    from .src.llm_client import LLMClient
except ImportError:
    _base_dir = os.path.dirname(__file__)
    if _base_dir and _base_dir not in sys.path:
        sys.path.insert(0, _base_dir)
    from src.core.models import DummyEvent
    from src.core.scheduler import Scheduler
    from src.core.state import MemChatState
    from src.db.database import DatabaseManager
    from src.llm_client import LLMClient


def _should_ignore_sender(self_id, cached_self_id, sender_id) -> bool:
    if not sender_id:
        return True
    if self_id and str(self_id) == str(sender_id):
        return True
    if cached_self_id and str(cached_self_id) == str(sender_id):
        return True
    return False


@register("mem_chat", "author", "一个简单的 mem_chat 插件", "0.1.0", "")
class MemChat(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}

        self.state = MemChatState()

        self._mem_db_path = os.path.join(
            os.path.dirname(__file__), "mem_chat_embeddings.db"
        )

        # Initialize Managers
        self.db_mgr = DatabaseManager(self._mem_db_path, self.config)
        self.llm_client = LLMClient(self)
        self.scheduler = Scheduler(
            self, self.context, self.config, self.state, self.db_mgr, self.llm_client
        )

        asyncio.create_task(self._startup_init())

        logger.info(f"[mem_chat] Plugin initialized. DB path: {self._mem_db_path}")

    def _normalize_target_label(
        self,
        target_type: str,
        target_id: str | None = None,
        target_name: str | None = None,
    ) -> tuple[str, str, str]:
        t = str(target_type or "").strip()
        tid = str(target_id or "").strip()
        tname = str(target_name or "").strip()
        if t == "bot":
            return t, tid or "bot", tname or "机器人"
        if t in ("group", "group_all"):
            return (
                t,
                tid or "group",
                tname or ("群聊全体" if t == "group_all" else "群聊"),
            )
        if t in ("user", "reply_sender", "mentioned_user"):
            return "user", tid, tname or tid or "某人"
        return "group", "group", "群聊"

    def _extract_poke_meta(
        self,
        event: AstrMessageEvent,
        *,
        self_id: str,
        sender_id: str | None,
        sender_name: str | None,
    ) -> dict:
        raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
        action_text = "戳了戳"
        suffix_text = ""
        try:
            raw_info = raw.get("raw_info", []) if hasattr(raw, "get") else []
            for item in raw_info or []:
                if not isinstance(item, dict):
                    continue
                txt = str(item.get("txt") or "").strip()
                if not txt:
                    continue
                if txt in ("戳了戳", "拍了拍", "抱了抱"):
                    action_text = txt
                elif action_text:
                    suffix_text += txt
        except Exception:
            pass

        poke_operator_id = ""
        poke_target_id = ""
        for message in event.get_messages() or []:
            if not isinstance(message, Poke):
                continue
            poke_operator_id = str(
                getattr(message, "id", None) or sender_id or ""
            ).strip()
            poke_target_id = str(getattr(message, "qq", None) or "").strip()
            break

        sender_id_str = str(sender_id or "").strip()
        sender_name_str = str(sender_name or "").strip()
        operator_name = (
            sender_name_str
            if poke_operator_id and sender_id_str and poke_operator_id == sender_id_str
            else ""
        )

        target_name = ""
        if poke_target_id and self_id and poke_target_id == str(self_id).strip():
            target_name = "机器人"

        return {
            "poke_operator_id": poke_operator_id,
            "poke_operator_name": operator_name,
            "poke_target_id": poke_target_id,
            "poke_target_name": target_name,
            "poke_action_text": action_text,
            "poke_suffix_text": suffix_text,
        }

    def _build_poke_texts(self, event: AstrMessageEvent) -> list[str]:
        self_id = str(event.get_self_id() or "").strip()
        sender_id = str(event.get_sender_id() or "").strip()
        sender_name = str(event.get_sender_name() or "").strip()
        poke_meta = self._extract_poke_meta(
            event,
            self_id=self_id,
            sender_id=sender_id,
            sender_name=sender_name,
        )
        action_text = str(poke_meta.get("poke_action_text") or "戳了戳").strip()
        suffix_text = str(poke_meta.get("poke_suffix_text") or "").strip()
        texts: list[str] = []
        for message in event.get_messages() or []:
            if not isinstance(message, Poke):
                continue
            operator_id = str(getattr(message, "id", None) or sender_id or "").strip()
            target_id = str(getattr(message, "qq", None) or "").strip()
            operator_label = operator_id or "有人"
            if sender_name and sender_id and operator_id and sender_id == operator_id:
                operator_label = f"{sender_name}/{operator_id}"
            if target_id and self_id and target_id == self_id:
                texts.append(f"{operator_label} {action_text}你{suffix_text}".strip())
            elif target_id:
                texts.append(
                    f"{operator_label} {action_text} {target_id}{suffix_text}".strip()
                )
            else:
                texts.append(f"{operator_label} 触发了戳一戳".strip())
        return texts

    def _enrich_text_with_poke_details(
        self,
        event: AstrMessageEvent,
        text: str | None,
    ) -> str:
        base = str(text or "").strip()
        poke_texts = self._build_poke_texts(event)
        if not poke_texts:
            return base
        detail = "；".join([t for t in poke_texts if str(t or "").strip()]).strip()
        if not detail:
            return base
        if not base:
            return detail
        if base in ("[戳一戳]", "[其他消息]"):
            return detail
        merged = base.replace("[戳一戳]", detail).replace("[其他消息]", detail)
        return merged or detail

    def _infer_message_target(
        self,
        *,
        self_id: str,
        sender_id: str | None,
        sender_name: str | None,
        reply_target_id: str | None,
        reply_sender_id: str | None,
        reply_sender_name: str | None,
        at_targets: list[dict],
        at_all: bool,
        poke_meta: dict | None = None,
    ) -> dict:
        explicit_type = "group"
        explicit_id = "group"
        explicit_name = "群聊"
        reason = "no_explicit_target"
        confidence = "low"
        poke = poke_meta if isinstance(poke_meta, dict) else {}
        poke_target_id = str(poke.get("poke_target_id") or "").strip()
        poke_target_name = str(poke.get("poke_target_name") or "").strip()

        if poke_target_id and self_id and poke_target_id == str(self_id):
            explicit_type, explicit_id, explicit_name = self._normalize_target_label(
                "bot", self_id, poke_target_name or "机器人"
            )
            reason = "poke_bot"
            confidence = "high"
        elif poke_target_id:
            explicit_type, explicit_id, explicit_name = self._normalize_target_label(
                "user", poke_target_id, poke_target_name
            )
            reason = "poke_user"
            confidence = "high"
        elif reply_sender_id and self_id and str(reply_sender_id) == str(self_id):
            explicit_type, explicit_id, explicit_name = self._normalize_target_label(
                "bot", self_id, "机器人"
            )
            reason = "reply_to_bot"
            confidence = "high"
        elif at_targets:
            first = at_targets[0]
            first_id = str(first.get("id") or "").strip()
            first_name = str(first.get("name") or "").strip()
            if self_id and first_id == str(self_id):
                explicit_type, explicit_id, explicit_name = (
                    self._normalize_target_label("bot", self_id, "机器人")
                )
                reason = "at_bot"
                confidence = "high"
            else:
                explicit_type, explicit_id, explicit_name = (
                    self._normalize_target_label("user", first_id, first_name)
                )
                reason = "at_user"
                confidence = "high"
        elif reply_sender_id or reply_target_id:
            explicit_type, explicit_id, explicit_name = self._normalize_target_label(
                "user",
                str(reply_sender_id or reply_target_id),
                str(reply_sender_name or ""),
            )
            reason = "reply_to_user"
            confidence = "high"
        elif at_all:
            explicit_type, explicit_id, explicit_name = self._normalize_target_label(
                "group_all", "group_all", "群聊全体"
            )
            reason = "at_all"
            confidence = "medium"

        inferred_type, inferred_id, inferred_name = (
            explicit_type,
            explicit_id,
            explicit_name,
        )
        inferred_reason = reason
        if explicit_type == "group":
            inferred_type, inferred_id, inferred_name = self._normalize_target_label(
                "group", "group", "群聊"
            )
            inferred_reason = "default_group"

        return {
            "msg_id": "",
            "sender_id": str(sender_id or "").strip(),
            "sender_name": str(sender_name or "").strip(),
            "reply_to_msg_id": str(reply_target_id or "").strip(),
            "reply_to_sender_id": str(reply_sender_id or "").strip(),
            "reply_to_sender_name": str(reply_sender_name or "").strip(),
            "at_targets": at_targets,
            "at_all": bool(at_all),
            "poke_operator_id": str(poke.get("poke_operator_id") or "").strip(),
            "poke_operator_name": str(poke.get("poke_operator_name") or "").strip(),
            "poke_target_id": poke_target_id,
            "poke_target_name": poke_target_name,
            "poke_action_text": str(poke.get("poke_action_text") or "戳了戳").strip(),
            "poke_suffix_text": str(poke.get("poke_suffix_text") or "").strip(),
            "explicit_target_type": explicit_type,
            "explicit_target_id": explicit_id,
            "explicit_target_name": explicit_name,
            "inferred_target_type": inferred_type,
            "inferred_target_id": inferred_id,
            "inferred_target_name": inferred_name,
            "target_confidence": confidence,
            "target_reason": inferred_reason,
        }

    def _is_blocked_knowledge_item(
        self,
        text: str,
        *,
        source: str = "",
        url: str = "",
        file_path: str = "",
    ) -> bool:
        raw = "\n".join(
            [
                str(text or ""),
                str(source or ""),
                str(url or ""),
                str(file_path or ""),
            ]
        ).lower()
        if not raw.strip():
            return True
        # Hard block COLG来源，避免补丁/广告类知识混入。
        if any(tok in raw for tok in BLOCKED_KNOWLEDGE_SOURCE_TOKENS):
            return True
        if any(tok in raw for tok in BLOCKED_PATCH_TOKENS):
            return True
        return False

    async def _clean_colg_patch_knowledge(self) -> int:
        patterns = list(BLOCKED_KNOWLEDGE_SOURCE_TOKENS + BLOCKED_PATCH_TOKENS)
        cleaned = await self.db_mgr.delete_knowledge_by_like_patterns(patterns)
        try:
            colg_path = os.path.join(
                os.path.dirname(__file__), "knowledge", "colg_hourly.md"
            )
            if os.path.exists(colg_path):
                os.remove(colg_path)
        except Exception:
            pass
        if cleaned > 0:
            logger.info(
                f"[mem_chat] Cleaned blocked COLG/Patch knowledge entries: {cleaned}"
            )
        return int(cleaned)

    async def upsert_knowledge_items(
        self, items: list[dict], group_id: str = "global"
    ) -> dict:
        model_name = str(
            self.config.get("embedding_openai_model", "text-embedding-3-small")
            or "text-embedding-3-small"
        )
        added = 0
        skipped = 0
        for it in items or []:
            mid = str(it.get("id") or "")
            text = str(it.get("text") or "")
            src = str(it.get("source") or "")
            url = str(it.get("url") or "")
            if not mid or not text:
                continue
            if self._is_blocked_knowledge_item(text, source=src, url=url):
                skipped += 1
                continue
            try:
                await self.db_mgr.get_or_compute_embedding(
                    mid, text, group_id, model_name, type="knowledge"
                )
                added += 1
            except Exception:
                pass
        return {"added": added, "skipped": skipped}

    async def persona_share(
        self,
        target_umo: str,
        items: list[dict],
        *,
        intro: str | None = None,
        query: str | None = None,
    ) -> bool:
        if not target_umo:
            return False
        if not items:
            return False

        send_event = DummyEvent(target_umo, self.context)
        group_id = ""
        try:
            group_id = str(send_event.get_group_id() or "")
        except Exception:
            group_id = ""
        if not group_id:
            try:
                group_id = str(target_umo.split(":")[2])
            except Exception:
                group_id = "global"

        lines: list[str] = []
        image_urls: list[str] = []
        query_parts: list[str] = []
        for it in items or []:
            title = str(it.get("title") or "").strip()
            author = str(it.get("author") or "").strip()
            url = str(it.get("url") or it.get("artwork_url") or "").strip()
            img = str(it.get("image_url") or it.get("image") or "").strip()
            score = it.get("score")
            score_s = ""
            if score is not None and str(score).strip():
                score_s = str(score).strip()

            if title:
                query_parts.append(title)
            if author:
                query_parts.append(author)
            if url:
                query_parts.append(url)
            if img:
                image_urls.append(img)

            seg = f"- {title}" if title else "- (untitled)"
            if author:
                seg += f" | author={author}"
            if score_s:
                seg += f" | score={score_s}"
            if url:
                seg += f" | url={url}"
            if img:
                seg += f" | image={img}"
            lines.append(seg)

        if not image_urls:
            return False

        q = (query or " ".join(query_parts)).strip()
        if not q:
            q = "图片分享"

        (
            memory_context,
            admin_protocols_text,
        ) = await self.scheduler.build_memory_context(
            query=q,
            group_id=group_id,
            sender_name="System",
        )

        prompt = (
            "【场景描述】\n"
            "你正潜伏在群聊中观察大家的讨论。现在，你发现了几张非常热门/有趣的图片（来自 X/Twitter），你觉得很适合分享给群友。\n\n"
            "【你的任务】\n"
            "1. 观察氛围：阅读【最近对话】，判断现在的聊天气氛（是在热烈讨论、划水摸鱼还是冷场）。\n"
            "2. 组织语言：以你的【人物设定】口吻，组织一段 1-2 句的点评。\n"
            "   - 如果【最近对话】中有相关话题，请尝试“接梗”或顺着话题分享。\n"
            "   - 如果话题不相关，请用一种“打破僵局”或“插播分享”的自然方式引入。\n"
            "3. 严禁复读：不要像复读机一样机械地列出标题和作者，要像真实人类一样聊天。不要编造未提供的信息。\n"
            "4. 发送图片：必须将图片链接放入 JSON 的 response.images 数组中，不要在 response.text 中粘贴图片链接。\n"
            "5. 附带链接：可以在 response.text 中附带 0-1 个作品原链接（url 字段），不要堆砌链接。\n\n"
            "【待分享内容】\n" + "\n".join(lines).strip()
        )
        if intro and str(intro).strip():
            prompt = str(intro).strip() + "\n\n" + prompt

        # Record last sent images for feedback
        if target_umo not in self.state.last_sent_image_ids:
            self.state.last_sent_image_ids[target_umo] = []
        for it in items:
            iid = str(it.get("id") or "")
            if iid:
                self.state.last_sent_image_ids[target_umo].append(iid)

        # Keep only last 10
        if len(self.state.last_sent_image_ids[target_umo]) > 10:
            self.state.last_sent_image_ids[target_umo] = self.state.last_sent_image_ids[
                target_umo
            ][-10:]

        return await self.llm_client.invoke_llm(
            umo=target_umo,
            prompt=prompt,
            memory_context=memory_context,
            event=send_event,
            task_type="focus",
            admin_protocols_text=admin_protocols_text,
            image_urls=image_urls,
        )

    async def share_text_to_session(self, target_umo: str, text: str) -> bool:
        if not target_umo:
            return False
        if not text:
            return False
        send_event = DummyEvent(target_umo, self.context)
        await send_event.send(MessageEventResult().message(text))
        return True

    def _extract_json_obj(self, text: str) -> dict | None:
        s = str(text or "").strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", s)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None

    def is_group_enabled_for_target(self, target_umo: str) -> tuple[bool, str]:
        """Check whether a target group is enabled for mem_chat handling."""
        umo = str(target_umo or "").strip()
        if not umo:
            return False, "invalid_target"
        try:
            parts = umo.split(":")
            if len(parts) >= 3:
                group_id = str(parts[2] or "").strip()
            else:
                group_id = ""
        except Exception:
            group_id = ""
        if not group_id:
            return False, "invalid_group"
        try:
            group_whitelist = [
                str(g).strip()
                for g in (self.config.get("group_whitelist", []) or [])
                if str(g).strip()
            ]
        except Exception:
            group_whitelist = []
        if group_whitelist and group_id not in group_whitelist:
            return False, f"group_not_enabled({group_id})"
        return True, "ok"

    async def _resolve_bot_for_group(self, group_id: str, target_umo: str):
        """尝试获取该群的 bot 实例和 self_id，返回 (bot, self_id)。"""
        try:
            mem_scheduler_meta = self.context.get_registered_star("mem_scheduler")
            if mem_scheduler_meta and getattr(mem_scheduler_meta, "star_cls", None):
                ms = mem_scheduler_meta.star_cls
                grt = getattr(ms, "_group_runtime", {}).get(str(group_id))
                if grt:
                    bot = grt.get("bot")
                    sid = str(grt.get("self_id") or "").strip()
                    if bot:
                        return bot, sid
        except Exception:
            pass

        inferred_self_id = str(self.state.get_self_id(target_umo) or "").strip()

        try:
            insts = list(self.context.platform_manager.get_insts() or [])
        except Exception:
            insts = []

        platform_hint = str(target_umo).split(":")[0] if ":" in str(target_umo) else ""
        insts.sort(
            key=lambda x: (
                0
                if str(getattr(getattr(x, "meta", lambda: None)(), "id", "") or "")
                == platform_hint
                else 1
            )
        )

        for inst in insts:
            bot = getattr(inst, "bot", None)
            if not bot:
                continue
            try:
                if inferred_self_id:
                    return bot, inferred_self_id
                info = await bot.get_login_info()
                data = info.get("data", {}) if isinstance(info, dict) else {}
                sid = str(data.get("user_id") or "").strip()
                if sid:
                    return bot, sid
                sid = str(getattr(inst, "client_self_id", "") or "").strip()
                if sid.isdigit() and sid not in ("bot", "unknown", "unknown_selfid"):
                    return bot, sid
            except Exception:
                try:
                    sid = str(getattr(inst, "client_self_id", "") or "").strip()
                    if sid.isdigit() and sid not in (
                        "bot",
                        "unknown",
                        "unknown_selfid",
                    ):
                        return bot, sid
                except Exception:
                    pass
                continue
        return None, ""

    async def suggest_self_group_card(
        self,
        target_umo: str,
        topic: str = "",
        max_len: int = 18,
    ) -> dict:
        """使用 mem_chat 当前人格/上下文，生成“是否改群名片”的决策 JSON。"""
        umo = str(target_umo or "").strip()
        if not umo:
            return {"should_change": False, "reason": "invalid_target"}
        enabled, enabled_reason = self.is_group_enabled_for_target(umo)
        if not enabled:
            return {"should_change": False, "reason": enabled_reason}

        provider_id = await self.context.get_current_chat_provider_id(umo)
        provider_id = str(provider_id or "").strip()
        if not provider_id:
            provider = self.context.get_using_provider()
            provider_id = str(getattr(provider, "id", "") or "").strip()
        if not provider_id:
            return {"should_change": False, "reason": "no_provider"}

        group_id = ""
        try:
            group_id = str(umo.split(":")[2])
        except Exception:
            group_id = "global"

        personas = await self.db_mgr.get_latest_personas(limit=2)
        persona_text = "\n\n".join(
            [str(x or "").strip() for x in personas if str(x or "").strip()]
        )[:3000]
        memory_context, _ = await self.scheduler.build_memory_context(
            query=(str(topic or "").strip() or "群名片调整"),
            group_id=group_id,
            sender_name="System",
            task_type="focus",
        )

        prompt = (
            "你要为当前机器人决定是否更改自己在群内的群名片。"
            "请只输出 JSON，不要输出解释。\n"
            f"目标会话: {umo}\n"
            f"最大长度: {int(max(1, max_len))}\n"
            f"主题偏好: {str(topic or '无').strip()}\n\n"
            "输出格式:\n"
            '{"should_change": true/false, "new_card": "...", "reason": "..."}\n\n'
            "约束:\n"
            "1) 需要改名时 should_change=true，new_card 不能为空。\n"
            "2) 不需要改名时 should_change=false，new_card 置空。\n"
            "3) 名字自然、简洁、可读，不要营销感，不要符号堆砌。\n"
        )

        system_prompt = (
            "你是 mem_chat 机器人本体，必须严格遵守角色设定和最近对话氛围做昵称决策。"
            "只输出 JSON。"
        )
        if persona_text:
            system_prompt += "\n\n【当前人格设定片段】\n" + persona_text

        try:
            resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt + "\n\n【最近上下文】\n" + (memory_context or ""),
                system_prompt=system_prompt,
            )
            obj = self._extract_json_obj(getattr(resp, "completion_text", ""))
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            logger.warning(f"[mem_chat] suggest_self_group_card failed: {e}")
        return {"should_change": False, "reason": "llm_failed"}

    async def submit_nickname_refresh_task(
        self,
        target_umo: str,
        topic: str = "",
        max_len: int = 18,
        runtime: dict | None = None,
        must_change: bool = False,
    ) -> dict:
        """像常规推送一样投递一次“改名任务”给 mem_chat，由其自行决策并执行工具。"""
        if not target_umo:
            return {
                "success": False,
                "outcome": "failed",
                "error": "missing_target_umo",
            }
        enabled, enabled_reason = self.is_group_enabled_for_target(target_umo)
        if not enabled:
            return {
                "success": False,
                "outcome": "failed",
                "error": enabled_reason,
            }
        send_event = DummyEvent(target_umo, self.context)
        try:
            setattr(send_event, "_mem_chat_force_self_group_card_target", True)
        except Exception:
            pass
        runtime_self_id = ""
        if isinstance(runtime, dict):
            try:
                bot = runtime.get("bot")
                self_id = str(runtime.get("self_id") or "").strip()
                adapter = runtime.get("adapter")
                if bot is not None:
                    setattr(send_event, "bot", bot)
                if adapter is not None:
                    setattr(send_event, "adapter", adapter)
                if self_id:
                    runtime_self_id = self_id
                    send_event.message_obj.self_id = self_id
            except Exception:
                pass
        if not runtime_self_id:
            try:
                runtime_self_id = str(self.state.get_self_id(target_umo) or "").strip()
                if runtime_self_id:
                    send_event.message_obj.self_id = runtime_self_id
            except Exception:
                runtime_self_id = ""
        group_id = ""
        try:
            group_id = str(send_event.get_group_id() or "")
        except Exception:
            group_id = ""
        if not group_id:
            try:
                group_id = str(target_umo.split(":")[2])
            except Exception:
                group_id = "global"

        if not runtime_self_id or getattr(send_event, "bot", None) is None:
            try:
                bot, sid = await self._resolve_bot_for_group(group_id, target_umo)
                if bot and getattr(send_event, "bot", None) is None:
                    setattr(send_event, "bot", bot)
                if sid:
                    runtime_self_id = sid
                    send_event.message_obj.self_id = sid
            except Exception:
                pass

        current_card_text = ""
        if getattr(send_event, "bot", None) and runtime_self_id and group_id:
            try:
                info = await send_event.bot.get_group_member_info(
                    group_id=int(group_id),
                    user_id=int(runtime_self_id),
                    no_cache=True,
                )
                data = info.get("data", {}) if isinstance(info, dict) else {}
                current_card_raw = str(data.get("card") or "").strip()
                nickname = str(data.get("nickname") or "").strip()
                current_card_text = current_card_raw or nickname
                logger.info(
                    f"[mem_chat] nickname_refresh current card for {group_id}: card={current_card_text}"
                )
            except Exception as e:
                logger.warning(
                    f"[mem_chat] nickname_refresh failed to query current card: {e}"
                )

        recent_chat_lines: list[str] = []
        try:
            recent_msgs = await self.db_mgr.get_recent_raw_group_messages(
                group_id=group_id,
                limit=30,
            )
            for sender_name, sender_id, text, created_at in recent_msgs:
                if not text:
                    continue
                ts_str = ""
                try:
                    ts_str = time.strftime(
                        "%m-%d %H:%M:%S", time.localtime(float(created_at))
                    )
                except Exception:
                    pass
                label = (
                    f"{sender_name}/{sender_id}"
                    if sender_id
                    else (sender_name or "User")
                )
                line = f"[{ts_str}] {label}: {text}" if ts_str else f"{label}: {text}"
                recent_chat_lines.append(line)
        except Exception as e:
            logger.warning(
                f"[mem_chat] nickname_refresh failed to fetch recent chat: {e}"
            )

        query = str(topic or "").strip() or "群名片定时刷新任务"
        (
            memory_context,
            admin_protocols_text,
        ) = await self.scheduler.build_memory_context(
            query=query,
            group_id=group_id,
            sender_name="System",
            task_type="focus",
        )
        prompt = (
            "【系统定时任务：群名片刷新】\n"
            "你现在要根据当前人格设定、最近群聊上下文，判断是否需要更新你自己的群名片。\n"
            f"主题偏好: {str(topic or '无').strip()}\n"
            f"名片长度上限: {int(max(1, max_len))}\n\n"
            f"机器人自身QQ号: {runtime_self_id or '未知'}\n"
            f"当前群名片: {current_card_text or '未知'}\n\n"
        )
        if recent_chat_lines:
            prompt += "【最近群聊记录】\n" + "\n".join(recent_chat_lines) + "\n\n"
        prompt += (
            "执行规则：\n"
            "1) 先判断是否需要改名；需要改名时必须输出 control.set_group_card 执行改名。\n"
            "2) control.set_group_card 必须包含 target_qq(机器人自己的QQ号) 与 new_card(新名片)。\n"
            "2.1) target_qq 必须使用上方给出的真实机器人QQ号；不要写 0、self、bot 或空字符串。\n"
            "3) 如果不需要改名，不要输出 set_group_card。\n"
            "4) 名片要自然、可读，符合你当前人设，不要营销感和符号堆砌。\n"
            "5) 这是内部任务，不要输出“我在执行任务”等机械描述。\n"
        )
        if must_change:
            prompt += (
                "6) 本轮为链路校验任务：必须执行一次 set_group_card，不允许跳过。\n"
            )
        invoke_ok = await self.llm_client.invoke_llm(
            umo=target_umo,
            prompt=prompt,
            memory_context=memory_context,
            event=send_event,
            task_type="focus",
            admin_protocols_text=admin_protocols_text,
            image_urls=None,
        )
        card_result = getattr(send_event, "_mem_chat_group_card_result", None)
        control_decision = str(
            getattr(send_event, "_mem_chat_last_control_decision", "") or ""
        ).strip()
        send_ok = bool(getattr(send_event, "_mem_chat_last_send_ok", False))
        if isinstance(card_result, dict) and card_result.get("attempted"):
            success = bool(card_result.get("success", False))
            return {
                "success": success,
                "outcome": "updated_via_mem_chat"
                if success
                else "mem_chat_card_failed",
                "card": str(card_result.get("new_card") or "").strip(),
                "reason": str(card_result.get("reason") or "").strip(),
                "decision": control_decision,
                "send_ok": send_ok,
            }
        if invoke_ok:
            return {
                "success": True,
                "outcome": "no_change",
                "decision": control_decision or "unknown",
                "send_ok": send_ok,
            }
        return {
            "success": False,
            "outcome": "failed",
            "error": "invoke_failed",
            "decision": control_decision or "unknown",
        }

    async def submit_reminder_task(
        self,
        target_umo: str,
        text: str,
        mention_qq: str = "",
        mention_name: str = "",
    ) -> bool:
        """将提醒任务交给 mem_chat 的 LLM 推送链路生成并发送（优先 @ 目标用户）。"""
        if not target_umo:
            return False
        msg = str(text or "").strip() or "定时提醒：请处理计划事项。"
        send_event = DummyEvent(target_umo, self.context)
        group_id = ""
        try:
            group_id = str(send_event.get_group_id() or "")
        except Exception:
            group_id = ""
        if not group_id:
            try:
                group_id = str(target_umo.split(":")[2])
            except Exception:
                group_id = "global"

        (
            memory_context,
            admin_protocols_text,
        ) = await self.scheduler.build_memory_context(
            query=msg,
            group_id=group_id,
            sender_name="System",
            task_type="focus",
        )
        mqq = str(mention_qq or "").strip()
        mname = str(mention_name or "").strip()
        target_hint = f"{mname}({mqq})" if mqq and mname else (mqq or mname or "无")
        prompt = (
            "【系统定时任务：提醒推送】\n"
            "请根据当前人设，用自然口吻发出这条提醒。\n"
            f"提醒内容：{msg}\n"
            f"提醒对象：{target_hint}\n\n"
            "执行要求：\n"
            "1) 若提醒对象存在QQ号，必须在回复中@该对象（使用 <at:QQ号>）。\n"
            "2) 回复简短自然，不要机械模板，不要输出系统描述。\n"
            "3) 本轮必须回复（control.decision=reply_only）。\n"
        )
        return await self.llm_client.invoke_llm(
            umo=target_umo,
            prompt=prompt,
            memory_context=memory_context,
            event=send_event,
            task_type="focus",
            admin_protocols_text=admin_protocols_text,
            image_urls=None,
        )

    async def persona_share_text_items(
        self,
        target_umo: str,
        items: list[dict],
        *,
        intro: str | None = None,
        query: str | None = None,
    ) -> bool:
        if not target_umo:
            return False
        if not items:
            return False

        send_event = DummyEvent(target_umo, self.context)
        group_id = ""
        try:
            group_id = str(send_event.get_group_id() or "")
        except Exception:
            group_id = ""
        if not group_id:
            try:
                group_id = str(target_umo.split(":")[2])
            except Exception:
                group_id = "global"

        lines: list[str] = []
        query_parts: list[str] = []
        for it in items or []:
            title = str(it.get("title") or "").strip()
            author = str(it.get("author") or "").strip()
            url = str(it.get("url") or "").strip()
            text = str(it.get("text") or "").strip()
            snippet = text[:120].replace("\n", " ").strip() if text else ""

            if title:
                query_parts.append(title)
            if author:
                query_parts.append(author)
            if url:
                query_parts.append(url)

            seg = f"- 标题: {title or '未命名'}"
            if author:
                seg += f" | 作者: {author}"
            if snippet:
                seg += f" | 概要: {snippet}"
            if url:
                seg += f" | 链接: {url}"
            lines.append(seg)

        q = (query or " ".join(query_parts)).strip()
        if not q:
            q = "外部推荐内容"

        (
            memory_context,
            admin_protocols_text,
        ) = await self.scheduler.build_memory_context(
            query=q,
            group_id=group_id,
            sender_name="System",
        )

        prompt = (
            "【场景描述】\n"
            "你正以自己的群聊人格潜伏在群里观察大家的讨论。现在有几条外部推荐内容，需要你用自然的口吻分享给群友。\n\n"
            "【待分享内容】\n" + "\n".join(lines).strip() + "\n\n"
            "【你的任务】\n"
            "1. 结合【最近对话】的氛围，挑选其中 1-3 条你觉得最适合现在分享的内容。\n"
            "2. 用简体中文生成 1-3 句自然的群聊发言，可以适当概括内容亮点或给出评价。\n"
            "3. 语气像真实玩家聊天，不要机械罗列表格字段，不要复读上面的原文。\n"
            "4. 如需附链接，只保留 0-1 个最关键的链接即可。\n"
        )
        if intro and str(intro).strip():
            prompt = str(intro).strip() + "\n\n" + prompt

        return await self.llm_client.invoke_llm(
            umo=target_umo,
            prompt=prompt,
            memory_context=memory_context,
            event=send_event,
            task_type="focus",
            admin_protocols_text=admin_protocols_text,
            image_urls=None,
        )

    async def handle_feedback(self, event: AstrMessageEvent):
        """处理用户对图片的评分和反馈"""
        msg = event.get_plain_text().strip()
        umo = (
            event.get_platform_name()
            + ":"
            + str(event.get_self_id())
            + ":"
            + str(event.get_group_id() or event.get_sender_id())
        )

        # 评分关键词映射
        positive = ["好图", "喜欢", "10分", "好看", "赞", "牛逼", "收藏"]
        neutral = ["一般", "还行", "5分", "中规中矩"]
        negative = ["不喜欢", "差评", "0分", "难看", "不行"]

        rating = None
        if any(k in msg for k in positive):
            rating = 1.0
        elif any(k in msg for k in neutral):
            rating = 0.5
        elif any(k in msg for k in negative):
            rating = 0.0

        if rating is not None:
            last_ids = self.state.last_sent_image_ids.get(umo)
            if last_ids:
                # 给最近发出的所有图片加分（或者只给最后一张，这里给最后一张）
                target_id = last_ids[-1]
                await self.db_mgr.update_image_rating(target_id, rating)
                logger.info(
                    f"[mem_chat] User feedback received for {target_id}: {rating}"
                )
                # 机器人可以简单的回应一下，或者不回应（保持沉浸感）
                # await event.send("记录下你的品味啦~")

    def __del__(self):
        if hasattr(self, "db_mgr") and self.db_mgr:
            self.db_mgr.close()

    def _get_vision_media(self):
        try:
            meta = self.context.get_registered_star("vision_media")
            if not meta or not getattr(meta, "activated", True):
                return None
            inst = getattr(meta, "star_cls", None)
            return inst
        except Exception:
            return None

    async def _fallback_extract_images(self, event: AstrMessageEvent) -> list[str]:
        images: list[str] = []
        try:
            chain = getattr(getattr(event, "message_obj", None), "message", None)
            if not chain:
                return images
            for i in chain:
                if isinstance(i, Image):
                    img_src = i.url if i.url else i.file
                    if img_src:
                        images.append(str(img_src).strip("`").strip())
        except Exception:
            pass
        return images

    async def _fallback_text_with_images(self, event: AstrMessageEvent) -> str:
        try:
            chain = getattr(getattr(event, "message_obj", None), "message", None)
            if not chain:
                return self._enrich_text_with_poke_details(
                    event, event.message_str or ""
                )
            parts: list[str] = []
            for i in chain:
                if isinstance(i, Plain):
                    parts.append(i.text)
                elif isinstance(i, Image):
                    parts.append("[图片]")
                elif isinstance(i, Reply):
                    parts.append("[回复]")
                elif isinstance(i, Poke):
                    parts.extend(self._build_poke_texts(event))
                elif isinstance(i, Node | Nodes):
                    parts.append("[转发消息]")
                elif isinstance(i, Json):
                    try:
                        data = getattr(i, "data", "")
                        s = str(data or "")
                        if any(k in s for k in ("聊天记录", "转发", "forward")):
                            parts.append("[转发消息]")
                        else:
                            parts.append("[卡片消息]")
                    except Exception:
                        parts.append("[卡片消息]")
                else:
                    parts.append("[其他消息]")
            text = "".join(parts).strip() or (event.message_str or "")
            return self._enrich_text_with_poke_details(event, text)
        except Exception:
            return self._enrich_text_with_poke_details(event, event.message_str or "")

    async def _startup_init(self):
        await self._purge_persona_vectors()
        await self._clean_colg_patch_knowledge()
        await self._check_and_update_knowledge()
        await self._check_and_update_persona_rag()
        try:
            if bool(self.config.get("raw_messages_backfill_on_startup", True)):
                stats = await self.db_mgr.backfill_raw_group_messages_from_memory_embeddings(
                    batch_size=int(
                        self.config.get("raw_messages_backfill_batch_size", 2000)
                    ),
                    force=bool(self.config.get("raw_messages_backfill_force", False)),
                )
                if int(stats.get("skipped", 0)) == 1:
                    logger.info(
                        "[mem_chat] raw_group_messages backfill skipped (already done)"
                    )
                else:
                    logger.info(
                        "[mem_chat] raw_group_messages backfill done scanned=%s parsed=%s inserted=%s",
                        stats.get("scanned", 0),
                        stats.get("parsed", 0),
                        stats.get("inserted", 0),
                    )
        except Exception as e:
            logger.error(f"[mem_chat] raw_group_messages backfill failed: {e}")
        try:
            if bool(self.config.get("user_aliases_backfill_on_startup", True)):
                stats = await self.db_mgr.backfill_user_aliases_from_raw_group_messages(
                    batch_size=int(
                        self.config.get("user_aliases_backfill_batch_size", 5000)
                    ),
                    force=bool(self.config.get("user_aliases_backfill_force", False)),
                )
                if int(stats.get("skipped", 0)) == 1:
                    logger.info(
                        "[mem_chat] user_aliases backfill skipped (already done)"
                    )
                else:
                    logger.info(
                        "[mem_chat] user_aliases backfill done scanned=%s upserted=%s",
                        stats.get("scanned", 0),
                        stats.get("upserted", 0),
                    )
        except Exception as e:
            logger.error(f"[mem_chat] user_aliases backfill failed: {e}")

    async def _purge_persona_vectors(self):
        try:
            await self.db_mgr.delete_by_type("persona")
            await self.db_mgr.delete_meta("last_persona_hash")
            self.state.set_last_persona_hash("")
        except Exception as e:
            logger.error(f"[mem_chat] Failed to purge persona vectors: {e}")

    async def _check_and_update_knowledge(self):
        """Check if knowledge files have changed and update vector DB."""
        try:
            knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
            if not os.path.exists(knowledge_dir):
                os.makedirs(knowledge_dir)
                # Create a readme
                with open(
                    os.path.join(knowledge_dir, "README.md"), "w", encoding="utf-8"
                ) as f:
                    f.write(
                        "Place your .txt or .md files here. They will be automatically imported into the knowledge base."
                    )

            # Migration: Move admin_protocols out of knowledge dir
            old_admin_dir = os.path.join(knowledge_dir, "admin_protocols")
            admin_dir = os.path.join(os.path.dirname(__file__), "admin_protocols")

            if os.path.exists(old_admin_dir) and not os.path.exists(admin_dir):
                try:
                    import shutil

                    shutil.move(old_admin_dir, admin_dir)
                    logger.info(
                        f"[mem_chat] Migrated admin_protocols from {old_admin_dir} to {admin_dir}"
                    )
                except Exception as e:
                    logger.error(f"[mem_chat] Failed to migrate admin_protocols: {e}")

            if not os.path.exists(admin_dir):
                os.makedirs(admin_dir)

            files = []
            # Scan knowledge dir
            for root, _, names in os.walk(knowledge_dir):
                # Skip old admin_protocols if it still exists
                if os.path.abspath(root).startswith(os.path.abspath(old_admin_dir)):
                    continue

                for name in names:
                    if name.lower() == "readme.md":
                        continue
                    if not name.lower().endswith((".txt", ".md")):
                        continue
                    if "colg" in name.lower():
                        continue
                    if "arisu_rag" in name.lower() or "persona_rag" in name.lower():
                        continue
                    files.append(os.path.join(root, name))

            # Scan admin_protocols dir
            for root, _, names in os.walk(admin_dir):
                for name in names:
                    if name.lower() == "readme.md":
                        continue
                    if not name.lower().endswith((".txt", ".md")):
                        continue
                    files.append(os.path.join(root, name))
            if not files:
                return

            # Read all content to compute hash
            combined_content = ""
            file_contents = {}
            for fp in sorted(files):
                # Skip README.md
                if os.path.basename(fp).lower() == "readme.md":
                    continue

                try:
                    with open(fp, encoding="utf-8") as f:
                        c = f.read().strip()
                        if self._is_blocked_knowledge_item("", file_path=fp):
                            continue
                        if c:
                            chunks = [x.strip() for x in c.split("\n\n") if x.strip()]
                            kept_chunks = [
                                x
                                for x in chunks
                                if (
                                    not self._is_blocked_knowledge_item(x, file_path=fp)
                                )
                                and ("【爱丽丝语料片段】" not in x)
                            ]
                            if kept_chunks:
                                safe_text = "\n\n".join(kept_chunks).strip()
                                combined_content += safe_text
                                file_contents[fp] = safe_text
                except Exception as e:
                    logger.error(f"[mem_chat] Failed to read {fp}: {e}")

            if not combined_content:
                return

            import hashlib

            h = hashlib.md5(combined_content.encode("utf-8")).hexdigest()

            # Check DB meta first
            last_hash = await self.db_mgr.get_meta("last_knowledge_hash")
            if h == last_hash:
                return

            logger.info(
                "[mem_chat] Knowledge files changed. Performing incremental update..."
            )

            model_name = str(
                self.config.get("embedding_openai_model", "text-embedding-3-small")
                or "text-embedding-3-small"
            )

            def _get_knowledge_group_id(file_path: str) -> str:
                try:
                    if os.path.commonpath(
                        [os.path.abspath(file_path), admin_dir]
                    ) == os.path.abspath(admin_dir):
                        return "protocol_admin"
                except Exception:
                    pass
                return "global"

            # 1. Prepare target state
            target_chunks = {}
            for fp, content in file_contents.items():
                group_id = _get_knowledge_group_id(fp)
                chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
                for chunk in chunks:
                    mid = self.db_mgr.hash_id(chunk, group_id, model_name)
                    target_chunks[mid] = {"content": chunk, "group_id": group_id}
            target_ids = set(target_chunks.keys())
            # 2. Get current state from DB
            existing_ids = await self.db_mgr.get_all_ids_by_type("knowledge")
            # 3. Calculate diff
            ids_to_delete = list(existing_ids - target_ids)
            ids_to_add = list(target_ids - existing_ids)
            # 4. Execute updates
            if ids_to_delete:
                await self.db_mgr.delete_by_ids(ids_to_delete)

            count = 0
            for mid in ids_to_add:
                item = target_chunks[mid]
                chunk = item["content"]
                group_id = item["group_id"]
                await self.db_mgr.get_or_compute_embedding(
                    mid, chunk, group_id, model_name, type="knowledge"
                )
                count += 1
                if count % 2 == 0:  # 每2条稍微停一下，避免触发 SiliconFlow 的频率限制
                    await asyncio.sleep(0.5)

            await self.db_mgr.set_meta("last_knowledge_hash", h)
            self.state.set_last_knowledge_hash(h)
            logger.info(
                f"[mem_chat] Knowledge Update: Added {count}, Deleted {len(ids_to_delete)}, Total {len(target_ids)} entries."
            )

        except Exception as e:
            logger.error(f"[mem_chat] Failed to update knowledge: {e}")

    async def _check_and_update_persona_rag(self):
        """Load dedicated persona RAG chunks for official story dialogue."""
        try:
            rag_dir = os.path.join(os.path.dirname(__file__), "persona_rag")
            if not os.path.exists(rag_dir):
                return

            rag_files: list[str] = []
            for root, _, names in os.walk(rag_dir):
                for name in names:
                    if name.lower() == "readme.md":
                        continue
                    if not name.lower().endswith((".txt", ".md")):
                        continue
                    rag_files.append(os.path.join(root, name))
            if not rag_files:
                return

            chunks: list[str] = []
            for rag_path in sorted(rag_files):
                try:
                    with open(rag_path, encoding="utf-8") as f:
                        raw = (f.read() or "").strip()
                    if not raw:
                        continue
                    for c in [x.strip() for x in raw.split("\n\n") if x.strip()]:
                        cc = re.sub(r"^\s*【爱丽丝语料片段】\s*", "", c).strip()
                        if not cc:
                            continue
                        if self._is_blocked_knowledge_item(cc, file_path=rag_path):
                            continue
                        chunks.append(cc)
                except Exception as e:
                    logger.error(
                        f"[mem_chat] Failed to read persona_rag file {rag_path}: {e}"
                    )
            # dedupe keep order
            deduped: list[str] = []
            seen = set()
            for c in chunks:
                key = " ".join(c.split())
                if not key or key in seen:
                    continue
                seen.add(key)
                deduped.append(c)
            chunks = deduped
            if not chunks:
                return

            import hashlib

            digest = hashlib.md5("\n\n".join(chunks).encode("utf-8")).hexdigest()
            last_hash = await self.db_mgr.get_meta("last_persona_rag_hash")
            if digest == str(last_hash or ""):
                return

            model_name = str(
                self.config.get("embedding_openai_model", "text-embedding-3-small")
                or "text-embedding-3-small"
            )
            await self.db_mgr.delete_by_type("persona_rag")

            added = 0
            for chunk in chunks:
                mid = "persona_rag::" + self.db_mgr.hash_id(chunk, "global", model_name)
                await self.db_mgr.get_or_compute_embedding(
                    mid,
                    chunk,
                    "global",
                    model_name,
                    type="persona_rag",
                )
                added += 1
                if added % 2 == 0:
                    await asyncio.sleep(0.5)

            await self.db_mgr.set_meta("last_persona_rag_hash", digest)
            logger.info(f"[mem_chat] Persona RAG update done: {added} entries.")
        except Exception as e:
            logger.error(f"[mem_chat] Failed to update persona RAG: {e}")

    @filter.command("mem_chat")
    async def cmd_mem_chat(self, event: AstrMessageEvent):
        """mem_chat 插件帮助"""
        msg = "mem_chat 插件指令:\n"
        msg += "/mem_chat help - 显示此帮助\n"
        msg += "/mem_chat clear - 清除当前会话的上下文记忆\n"
        msg += "/mem_chat stats - 显示当前统计信息\n"
        msg += "/mem_chat focus [seconds] - 强制开启专注模式（默认60秒）\n"
        msg += "/mem_chat forget - 遗忘当前对话的历史记录\n"
        yield event.plain_result(msg)

    @filter.command("mem_chat clear")
    async def cmd_mem_chat_clear(self, event: AstrMessageEvent):
        """清除当前会话的上下文记忆"""
        umo = event.unified_msg_origin
        self.state.clear_pending_messages(umo)
        self.state.clear_last_message(umo)
        yield event.plain_result("✅ 已清除当前会话的上下文缓存。")

    @filter.command("mem_chat stats")
    async def cmd_mem_chat_stats(self, event: AstrMessageEvent):
        """显示统计信息"""
        umo = event.unified_msg_origin
        pending = self.state.get_pending_count(umo)
        focus = self.state.is_in_focus(umo)
        focus_status = "✅" if focus else "❌"
        yield event.plain_result(
            f"📊 统计信息:\n待处理消息数: {pending}\n专注模式: {focus_status}",
        )

    @filter.command("mem_chat focus")
    async def cmd_mem_chat_focus(self, event: AstrMessageEvent, duration: int = 60):
        """开启专注模式"""
        umo = event.unified_msg_origin
        self.state.set_focus(umo, duration)
        yield event.plain_result(f"👀 已开启专注模式 {duration} 秒。")

    @filter.command("mem_chat forget")
    async def cmd_mem_chat_forget(self, event: AstrMessageEvent):
        """遗忘历史"""
        umo = event.unified_msg_origin
        cid = await self.context.conversation_manager.get_curr_conversation_id(umo)
        if cid:
            await self.context.conversation_manager.update_conversation(umo, cid, [])
            yield event.plain_result("🗑️ 已遗忘当前对话的历史记录。")
        else:
            yield event.plain_result("❌ 当前没有活跃的对话。")

    @filter.command("mem_chat reload")
    async def cmd_mem_chat_reload(self, event: AstrMessageEvent):
        """重新加载本地知识库和人格文件"""
        await self._clean_colg_patch_knowledge()
        await self._check_and_update_knowledge()
        await self._check_and_update_persona_rag()
        yield event.plain_result("已重新加载本地文件到数据库。")

    @filter.command("mem_chat test_json")
    async def cmd_mem_chat_test_json(self, event: AstrMessageEvent):
        umo = event.unified_msg_origin
        llm_cfg = self.llm_client._prepare_llm_config(umo, event)
        if not llm_cfg.get("use_custom_llm"):
            yield event.plain_result(
                "未启用自定义 LLM（当前走 provider），无法测试强制 JSON 输出参数。"
            )
            return

        cfg = llm_cfg["llm"]
        fmt = str(cfg.get("fmt") or "").lower()
        force = bool(self.config.get("force_llm_json_output"))

        prompt = (
            "请使用 json 格式，只输出一个 JSON 对象，不要任何解释。\n"
            "EXAMPLE JSON OUTPUT:\n"
            '{"response":{"text":"ping"},"control":{"decision":"reply_only"}}'
        )
        try:
            resp = await self.llm_client._call_custom_llm(
                base_url=cfg["base"],
                api_key=cfg["key"],
                model=cfg["model"],
                system_prompt="",
                prompt=prompt,
                contexts=[],
                tools=None,
                image_urls=None,
                temperature=cfg.get("temp"),
                fmt=fmt,
                tag="test_json",
            )
        except Exception as e:
            yield event.plain_result(f"请求失败：{type(e).__name__}: {e}")
            return

        text = (getattr(resp, "completion_text", "") or "").strip() if resp else ""
        if not text:
            yield event.plain_result("请求成功但返回为空。")
            return

        import json

        ok = False
        err = ""
        try:
            obj = json.loads(text)
            ok = isinstance(obj, dict)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        if ok:
            yield event.plain_result(
                f"JSON 测试通过。format={fmt} force_llm_json_output={force}"
            )
        else:
            preview = text[:300].replace("\n", "\\n")
            yield event.plain_result(
                f"JSON 测试失败。format={fmt} force_llm_json_output={force}\n"
                f"解析错误：{err}\n"
                f"输出预览：{preview}"
            )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("mc_rename")
    async def cmd_mc_rename(self, event: AstrMessageEvent):
        """管理员改机器人自己的群名片。用法: /mc_rename <新名片> 或 /mc_rename <群号> [主题提示]"""
        # 解析参数:
        # /mc_rename <new_card>              -> 直接改当前群机器人自己的群名片
        # /mc_rename <group_id> [topic]      -> 兼容旧用法，让 LLM 为指定群决策
        raw = str(event.message_str or "")
        # strip command prefix
        stripped = re.sub(r"^[#/]?mc_rename\s*", "", raw, flags=re.IGNORECASE).strip()
        if not stripped:
            yield event.plain_result(
                "用法: /mc_rename <新名片>\n"
                "例如: /mc_rename 勇者爱丽丝\n"
                "远程 LLM 决策: /mc_rename <群号> [可选主题提示]"
            )
            return
        parts = stripped.split(maxsplit=1)
        group_id = parts[0].strip()
        topic = parts[1].strip() if len(parts) > 1 else ""

        if not group_id.isdigit():
            new_card = stripped.strip()
            current_group_id = str(event.get_group_id() or "").strip()
            self_id = str(event.get_self_id() or "").strip()
            if not current_group_id:
                yield event.plain_result("❌ 只能在群聊里直接改群名片。")
                return
            if not self_id:
                yield event.plain_result("❌ 无法获取机器人自身 QQ。")
                return
            if len(new_card) > 18:
                yield event.plain_result("❌ 群名片太长了，最多 18 个字符。")
                return

            platform = str(event.get_platform_name() or "default").strip()
            target_umo = f"{platform}:GroupMessage:{current_group_id}"
            enabled, reason = self.is_group_enabled_for_target(target_umo)
            if not enabled:
                yield event.plain_result(
                    f"❌ 当前群 {current_group_id} 未在 mem_chat 白名单中: {reason}"
                )
                return

            try:
                bot = getattr(event, "bot", None)
                if bot and hasattr(bot, "set_group_card"):
                    await bot.set_group_card(
                        group_id=int(current_group_id),
                        user_id=int(self_id),
                        card=new_card,
                    )
                elif bot and hasattr(bot, "call_action"):
                    await bot.call_action(
                        "set_group_card",
                        group_id=int(current_group_id),
                        user_id=int(self_id),
                        card=new_card,
                    )
                else:
                    yield event.plain_result(
                        "❌ 当前事件没有可用的 bot 句柄，无法改名。"
                    )
                    return
                yield event.plain_result(f"✅ 已将当前群名片改为: {new_card}")
            except Exception as e:
                yield event.plain_result(f"❌ 改名失败: {e}")
            return

        # 构造 target_umo: {platform}:GroupMessage:{group_id}
        platform = str(event.get_platform_name() or "default").strip()
        target_umo = f"{platform}:GroupMessage:{group_id}"

        # 检查对该群是否启用了 mem_chat
        enabled, reason = self.is_group_enabled_for_target(target_umo)
        if not enabled:
            yield event.plain_result(
                f"❌ 目标群 {group_id} 未在 mem_chat 白名单中: {reason}"
            )
            return

        yield event.plain_result(
            f"🔄 正在通过 mem_chat 链路为群 {group_id} 决策群名片..."
        )

        # 从 mem_scheduler 的 runtime 缓存获取 bot 引用
        runtime = {}
        try:
            mem_scheduler_meta = self.context.get_registered_star("mem_scheduler")
            if mem_scheduler_meta and getattr(mem_scheduler_meta, "star_cls", None):
                ms = mem_scheduler_meta.star_cls
                grt = getattr(ms, "_group_runtime", {}).get(group_id)
                if grt:
                    bot = grt.get("bot")
                    if bot:
                        runtime["bot"] = bot
                    sid = str(grt.get("self_id") or "")
                    if sid:
                        runtime["self_id"] = sid
        except Exception:
            pass
        if "bot" not in runtime:
            try:
                bot = getattr(event, "bot", None)
                if bot:
                    runtime["bot"] = bot
            except Exception:
                pass
        if "self_id" not in runtime:
            try:
                sid = str(event.get_self_id() or "").strip()
                if sid:
                    runtime["self_id"] = sid
            except Exception:
                pass

        result = await self.submit_nickname_refresh_task(
            target_umo=target_umo,
            topic=topic or "管理员手动触发改名",
            max_len=18,
            runtime=runtime if runtime else None,
            must_change=True,
        )
        if result.get("success"):
            card = str(result.get("card") or "").strip()
            if card:
                suffix = ""
                if result.get("send_ok") is False:
                    suffix = "（目标群回复未发送或发送失败）"
                yield event.plain_result(
                    f"✅ 群 {group_id} 已通过 LLM 决策更新群名片为: {card}{suffix}"
                )
            else:
                yield event.plain_result(
                    f"✅ 改名任务已提交，LLM 决策: {result.get('outcome', 'unknown')}"
                )
        else:
            reason = (
                result.get("error")
                or result.get("reason")
                or result.get("outcome", "unknown")
            )
            yield event.plain_result(f"❌ 改名失败: {reason}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("mc_add")
    async def mc_add(self, event: AstrMessageEvent):
        """Add content manually. Format: /mc_add [type] content. Type defaults to knowledge."""
        # Use regex to strip command prefix and command name (supports /mc_add, #mc_add, mc_add)
        content = re.sub(
            r"^[#/]?mc_add\s*",
            "",
            event.message_str,
            flags=re.IGNORECASE,
        ).strip()

        if not content:
            yield event.plain_result("Usage: /mc_add [type] <content>")
            return

        # Simple parsing
        args = content.split(" ", 1)
        type_ = "knowledge"
        text = content

        if len(args) > 1 and args[0] in ["knowledge", "persona", "memory"]:
            type_ = args[0]
            text = args[1]

        model_name = str(
            self.config.get("embedding_openai_model", "text-embedding-3-small")
            or "text-embedding-3-small"
        )
        if type_ != "persona":
            mid = self.db_mgr.hash_id(text, str(event.get_group_id()), model_name)
            await self.db_mgr.get_or_compute_embedding(
                mid,
                text,
                str(event.get_group_id()),
                model_name,
                type=type_,
            )

        # Persist to file if knowledge or persona
        try:
            if type_ == "knowledge":
                k_path = os.path.join(
                    os.path.dirname(__file__), "knowledge", "imported_kb.txt"
                )
                # Ensure dir exists
                os.makedirs(os.path.dirname(k_path), exist_ok=True)
                with open(k_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n{text}")
            elif type_ == "persona":
                p_path = os.path.join(
                    os.path.dirname(__file__), "personas", "personas.txt"
                )
                os.makedirs(os.path.dirname(p_path), exist_ok=True)
                with open(p_path, "a", encoding="utf-8") as f:
                    f.write(f"\n{text}")
        except Exception as e:
            logger.error(f"[mem_chat] Failed to persist mc_add to file: {e}")

        yield event.plain_result(f"Added to {type_}: {text[:20]}...")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("mc_list")
    async def mc_list(self, event: AstrMessageEvent):
        """List content from vector DB. Format: /mc_list [type] [limit]"""
        # Use regex to strip command prefix and command name
        content = re.sub(
            r"^[#/]?mc_list\s*",
            "",
            event.message_str,
            flags=re.IGNORECASE,
        ).strip()

        args = content.split()
        type_ = "persona"
        limit = 20

        if len(args) >= 1:
            type_ = args[0]
        if len(args) >= 2:
            try:
                limit = int(args[1])
            except Exception:
                pass

        if type_ not in ["persona", "knowledge", "memory", "impression"]:
            yield event.plain_result(
                "Usage: /mc_list [persona|knowledge|memory|impression] [limit]",
            )
            return

        items = await self.db_mgr.get_all_embeddings_by_type(type_, limit)
        if not items:
            yield event.plain_result(f"No items found for type: {type_}")
            return

        msg = f"【{type_} (Top {len(items)})】\n"
        for i, item in enumerate(items):
            # Replace newlines with spaces for better list display
            preview = item[:50].replace("\n", " ")
            msg += f"{i + 1}. {preview}...\n"

        yield event.plain_result(msg)

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        umo = event.unified_msg_origin
        if not umo:
            return

        # Private chats are allowed only for explicitly allowlisted owners.
        if not event.get_group_id():
            sender = str(event.get_sender_id() or "").strip()
            allowed = {
                str(u).strip()
                for u in (self.config.get("private_chat_uids", []) or [])
                if str(u).strip()
            }
            if not sender or sender not in allowed:
                logger.info(f"[mem_chat] 拒绝非白名单私聊 {sender or '?'}")
                return
            logger.info(f"[mem_chat] 私聊会话 {sender}")

        try:
            await self.handle_feedback(event)
        except Exception:
            pass

        # Enforce group whitelist from config (private chats already gated above).
        group_id = event.get_group_id()
        try:
            group_whitelist = [
                str(g) for g in (self.config.get("group_whitelist", []) or [])
            ]
            if group_id and group_whitelist:
                gid_str = str(group_id)
                if gid_str not in group_whitelist:
                    logger.info(f"[mem_chat] 拒绝非白名单群组 {gid_str} 的请求")
                    return
        except Exception:
            pass

        try:
            self.state.set_self_id(umo, event.get_self_id())
            self.state.add_session(umo)
            try:
                gid = event.get_group_id()
                if gid:
                    group = await event.get_group(gid)
                    if group and getattr(group, "members", None):
                        self_id = str(event.get_self_id() or "")
                        current_name = None
                        for m in group.members or []:
                            try:
                                if str(getattr(m, "user_id", "")) != self_id:
                                    continue
                                name = str(
                                    getattr(m, "nickname", "")
                                    or getattr(m, "card", "")
                                    or ""
                                ).strip()
                                if name:
                                    current_name = name
                                break
                            except Exception:
                                continue
                        if current_name:
                            cached_name = self.state.get_self_name(umo)
                            if cached_name != current_name:
                                self.state.set_self_name(umo, current_name)
            except Exception:
                pass

            self_id = event.get_self_id()
            sender_id = event.get_sender_id()
            if self_id and sender_id and str(self_id) == str(sender_id):
                return
            cached_self_id = self.state.get_self_id(umo)
            if cached_self_id and sender_id and str(cached_self_id) == str(sender_id):
                return
        except Exception:
            pass

        self.state.set_self_id(umo, event.get_self_id())
        self.state.add_session(umo)

        sender_name = event.get_sender_name() or "User"
        sender_id = event.get_sender_id()

        if _should_ignore_sender(
            event.get_self_id(), self.state.get_self_id(umo), sender_id
        ):
            return

        self_id_str = str(event.get_self_id() or "")
        is_at = False
        for message in event.get_messages() or []:
            if isinstance(message, At):
                qq = getattr(message, "qq", None)
                if qq is not None and str(qq) == self_id_str:
                    is_at = True
                    break
            if isinstance(message, AtAll):
                is_at = True
                break
            if isinstance(message, Reply):
                reply_sender_id = getattr(message, "sender_id", None)
                if reply_sender_id is not None and str(reply_sender_id) == self_id_str:
                    is_at = True
                    break
            if isinstance(message, Poke):
                target_id = getattr(message, "qq", None)
                if target_id is None or str(target_id) == self_id_str:
                    is_at = True
                    break
        # A private chat has no @ or reply quote, so treat every message as
        # directed at the bot or the owner would never get an answer.
        if not event.get_group_id():
            is_at = True

        try:
            setattr(event, "is_at_or_wake_command", is_at)
        except Exception:
            pass

        try:
            if getattr(event, "meme_triggered", False):
                return
        except Exception:
            pass

        msg_id = None
        meta = None
        try:
            msg_obj = getattr(event, "message_obj", None)
            if msg_obj:
                msg_id = getattr(msg_obj, "message_id", None)
        except Exception:
            msg_obj = None

        vision_media = self._get_vision_media()
        if vision_media:
            try:
                text_with_images = await vision_media.render_text_with_captions(event)
            except Exception:
                text_with_images = await self._fallback_text_with_images(event)
            try:
                image_urls = await vision_media.extract_images(event)
            except Exception:
                image_urls = await self._fallback_extract_images(event)
        else:
            text_with_images = await self._fallback_text_with_images(event)
            image_urls = await self._fallback_extract_images(event)

        try:
            text_with_images = self._enrich_text_with_poke_details(
                event, text_with_images
            )
        except Exception:
            pass

        # The adapter turns QQ notices ("typing…", read receipts) into normal
        # friend/group messages with an empty body, so this is what keeps the
        # bot from answering a signal it can never see.
        if not (text_with_images or "").strip() and not image_urls:
            logger.info(f"[mem_chat] empty message content, ignoring for {umo}")
            return

        try:
            if msg_id is not None and msg_obj is not None:
                reply_target_id = None
                reply_sender_id = None
                reply_sender_name = None
                reply_text = None
                at_targets = []
                at_all = False
                poke_meta = {}
                for message in event.get_messages() or []:
                    if isinstance(message, Reply):
                        target_mid = getattr(message, "id", None)
                        if target_mid:
                            reply_target_id = target_mid
                            reply_sender_id = getattr(message, "sender_id", None)
                            reply_sender_name = getattr(
                                message, "sender_nickname", None
                            )
                            reply_text = getattr(message, "message_str", None)
                    elif isinstance(message, At):
                        qq = getattr(message, "qq", None)
                        if qq is not None:
                            at_targets.append(
                                {
                                    "id": str(qq),
                                    "name": str(getattr(message, "name", "") or ""),
                                }
                            )
                    elif isinstance(message, AtAll):
                        at_all = True
                try:
                    poke_meta = self._extract_poke_meta(
                        event,
                        self_id=self_id_str,
                        sender_id=str(sender_id) if sender_id is not None else None,
                        sender_name=sender_name,
                    )
                except Exception:
                    poke_meta = {}
                if reply_target_id is not None:
                    self.state.set_reply_target(
                        umo,
                        str(msg_id),
                        str(reply_target_id),
                    )
                    self.state.set_reply_summary(
                        umo,
                        str(reply_target_id),
                        str(reply_sender_id) if reply_sender_id is not None else None,
                        str(reply_sender_name)
                        if reply_sender_name is not None
                        else None,
                        str(reply_text) if reply_text is not None else None,
                    )
                    try:
                        self.state.cleanup_reply_data(
                            umo,
                            reply_targets_ttl_sec=float(
                                self.config.get("reply_targets_ttl_sec", 7 * 24 * 3600)
                            ),
                            reply_targets_max_size=int(
                                self.config.get("reply_targets_max_size", 4096)
                            ),
                            reply_summaries_ttl_sec=float(
                                self.config.get(
                                    "reply_summaries_ttl_sec", 7 * 24 * 3600
                                )
                            ),
                            reply_summaries_max_size=int(
                                self.config.get("reply_summaries_max_size", 4096)
                            ),
                        )
                    except Exception:
                        pass
                meta = self._infer_message_target(
                    self_id=self_id_str,
                    sender_id=str(sender_id) if sender_id is not None else None,
                    sender_name=sender_name,
                    reply_target_id=str(reply_target_id)
                    if reply_target_id is not None
                    else None,
                    reply_sender_id=str(reply_sender_id)
                    if reply_sender_id is not None
                    else None,
                    reply_sender_name=str(reply_sender_name)
                    if reply_sender_name is not None
                    else None,
                    at_targets=at_targets,
                    at_all=at_all,
                    poke_meta=poke_meta,
                )
                meta["msg_id"] = str(msg_id or "").strip()
                meta["text"] = str(text_with_images or "")
                try:
                    meta["ts"] = float(getattr(msg_obj, "time", None) or time.time())
                except Exception:
                    meta["ts"] = time.time()
                self.state.set_message_meta(umo, str(msg_id or ""), meta)
        except Exception:
            pass

        if self.state.is_duplicate_incoming(
            umo,
            msg_id,
            text_with_images,
            str(sender_id) if sender_id is not None else None,
            is_at=is_at,
        ):
            return

        # 3. Update State
        self.state.update_last_message(
            umo,
            str(msg_id) if msg_id is not None else None,
            text_with_images,
            sender_name,
            str(sender_id) if sender_id is not None else None,
        )

        # 4. Check Interrupt
        if self.state.is_processing(umo):
            if is_at:
                self.state.set_interrupt(umo)
                logger.debug(f"[mem_chat] Set interrupt for {umo} (is_at={is_at})")

        # 5. Check if Command (Ignore)
        if getattr(event, "is_command", False):
            return

        # 6. Save to Memory
        gid = event.get_group_id() or event.get_sender_id()
        # Save with Sender Name/ID format to include QQ in memory
        sender_info = f"{sender_name}/{sender_id}" if sender_id else sender_name
        try:
            message_seq = None
            msg_time = None
            created_ts = None
            if msg_obj is not None:
                message_seq = getattr(msg_obj, "message_seq", None) or getattr(
                    msg_obj, "seq", None
                )
                msg_time = getattr(msg_obj, "time", None)
            try:
                if str(msg_time or "").strip():
                    created_ts = float(msg_time)
            except Exception:
                created_ts = None
            asyncio.create_task(
                self.db_mgr.save_raw_group_message(
                    group_id=str(gid or ""),
                    message_id=str(msg_id or ""),
                    message_seq=str(message_seq or ""),
                    sender_id=str(sender_id or ""),
                    sender_name=str(sender_name or ""),
                    text=str(text_with_images or ""),
                    raw_text=str(getattr(event, "message_str", "") or ""),
                    image_urls_json=json.dumps(image_urls or [], ensure_ascii=False),
                    reply_to_msg_id=meta.get("reply_to_msg_id", "")
                    if isinstance(meta, dict)
                    else "",
                    reply_to_sender_id=meta.get("reply_to_sender_id", "")
                    if isinstance(meta, dict)
                    else "",
                    reply_to_sender_name=meta.get("reply_to_sender_name", "")
                    if isinstance(meta, dict)
                    else "",
                    at_targets_json=json.dumps(
                        meta.get("at_targets", []), ensure_ascii=False
                    )
                    if isinstance(meta, dict)
                    else "[]",
                    at_all=bool(meta.get("at_all", False))
                    if isinstance(meta, dict)
                    else False,
                    explicit_target_type=meta.get("explicit_target_type", "")
                    if isinstance(meta, dict)
                    else "",
                    explicit_target_id=meta.get("explicit_target_id", "")
                    if isinstance(meta, dict)
                    else "",
                    explicit_target_name=meta.get("explicit_target_name", "")
                    if isinstance(meta, dict)
                    else "",
                    created_at=created_ts,
                )
            )
        except Exception:
            pass
        # Keep the input and pending write so the reply's salience score can
        # update this user's memory after its embedding has finished.
        event._mem_chat_memory_input = (text_with_images, gid, sender_info)
        event._mem_chat_memory_task = asyncio.create_task(
            self.db_mgr.ensure_message_embedded(
                text_with_images,
                gid,
                sender_info,
                importance=getattr(event, "_mem_chat_memory_importance", None),
            )
        )

        # Legacy periodic profile summarization (disabled by default after ImpressionV3 stream rollout).
        if bool(self.config.get("legacy_periodic_profile_enabled", False)):
            c = self.state.increment_summary_msg_count(umo)
            if c >= int(
                self.config.get("legacy_periodic_profile_every_n_msgs", 50) or 50
            ):
                self.state.reset_summary_msg_count(umo)
                sid = event.get_sender_id()
                if sid:
                    asyncio.create_task(
                        self.scheduler.summarize_session(umo, gid, sid, sender_name)
                    )

        read_air_enable = bool(self.config.get("read_air_enable", True))
        if read_air_enable:
            self.state.increment_read_air_msg_count(umo)

        # 7. Cycle Logic
        # State Machine Trigger
        if is_at:
            # Event: AT
            # Action: Interrupt, (If already in focus: wait focus cycle; else: reply soon), Enter/Refresh Focus

            was_in_focus = False
            try:
                was_in_focus = bool(self.state.is_in_focus(umo))
            except Exception:
                was_in_focus = False

            def _to_float(v, d):
                try:
                    return float(v)
                except Exception:
                    return float(d)

            def _clamp(v, lo, hi):
                if hi < lo:
                    hi = lo
                if v < lo:
                    return lo
                if v > hi:
                    return hi
                return v

            def _step_towards(curr, target, step_max, lo, hi):
                curr = _clamp(curr, lo, hi)
                target = _clamp(target, lo, hi)
                step_max = max(0.0, float(step_max))
                if step_max <= 0:
                    return target
                delta = target - curr
                if abs(delta) <= step_max:
                    return target
                return curr + (step_max if delta > 0 else -step_max)

            base_delay = _to_float(self.config.get("focus_mode_cycle", 15), 15)
            min_delay = _to_float(
                self.config.get(
                    "focus_cycle_min_sec",
                    self.config.get("focus_mode_cycle_min", 10),
                ),
                min(10.0, base_delay),
            )
            max_delay = _to_float(
                self.config.get("focus_cycle_max_sec", base_delay + 20.0),
                base_delay + 20.0,
            )
            step_max = _to_float(
                self.config.get("focus_cycle_step_max_sec", 20),
                20.0,
            )
            if min_delay < 1:
                min_delay = 1.0
            if max_delay < min_delay:
                max_delay = min_delay
            if step_max <= 0:
                step_max = max_delay - min_delay
            base_delay = _clamp(base_delay, min_delay, max_delay)
            reward_ratio = self.config.get("focus_mode_cycle_at_reward_ratio", None)
            at_reduce = float(self.config.get("focus_mode_cycle_at_reduce", 20))
            try:
                reward_ratio = float(reward_ratio) if reward_ratio is not None else None
            except Exception:
                reward_ratio = None
            if reward_ratio is None:
                reward_ratio = (at_reduce / base_delay) if base_delay > 0 else 0.0

            if umo not in self.state.focus_delays:
                self.state.set_focus_delay(umo, base_delay, min_val=min_delay)
            curr_delay = self.state.get_focus_delay(umo, default=base_delay)
            target_delay = float(curr_delay) - abs(base_delay * reward_ratio)
            next_delay = _step_towards(
                float(curr_delay), target_delay, step_max, min_delay, max_delay
            )
            self.state.set_focus_delay(umo, next_delay, min_val=min_delay)

            # Enter/Refresh Focus State
            focus_duration = int(
                _to_float(self.config.get("focus_duration_sec", 100), 100)
            )
            focus_min = int(
                _to_float(self.config.get("focus_duration_min_sec", 60), 60)
            )
            focus_max = int(
                _to_float(self.config.get("focus_duration_max_sec", 900), 900)
            )
            focus_step_max = int(
                _to_float(self.config.get("focus_duration_step_max_sec", 120), 120)
            )
            if focus_min < 1:
                focus_min = 1
            if focus_max < focus_min:
                focus_max = focus_min
            if focus_step_max <= 0:
                focus_step_max = max(1, focus_max - focus_min)
            focus_duration = int(
                _clamp(float(focus_duration), float(focus_min), float(focus_max))
            )
            remaining = (
                float(self.state.get_focus_remaining_sec(umo)) if was_in_focus else 0.0
            )
            # @ 奖励续期：在现有剩余专注时长基础上做增量，避免每次都重置为固定值。
            # 剩余时长越高，单次增量越小，防止高频 @ 将专注时长无限堆到上限。
            at_extend_base = max(1, int(min_delay or 1))
            at_extend_cap = max(1, int(focus_step_max))
            decay = 1.0
            if focus_max > 0:
                decay = 1.0 - (remaining / float(focus_max))
            decay = _clamp(decay, 0.35, 1.0)
            focus_at_extend = int(
                max(1, min(float(at_extend_base), float(at_extend_cap)) * decay)
            )
            remaining_int = int(remaining) + (
                1 if (remaining - int(remaining)) > 0 else 0
            )
            target_duration = max(int(focus_duration), remaining_int) + focus_at_extend
            target_duration = int(
                _clamp(float(target_duration), float(focus_min), float(focus_max))
            )
            self.state.set_focus(umo, target_duration)

            # Prevent Core Reply
            try:
                event.should_call_llm(False)
            except Exception:
                pass

            logger.debug(
                f"[mem_chat] AT message received for {umo}. Triggering AT response (was_in_focus={was_in_focus})."
            )
            # Always route explicit @/reply/poke into the AT lane.
            # If we keep it in focus lane while focused, a newer non-@ message can become tail,
            # causing the @ target to be missed.
            await self.scheduler.trigger_view_cycle(umo, "at", event, image_urls)

        else:
            # Event: Normal Message
            # Action: Maintain current state cycle

            # is_in_focus() checks time and automatically downgrades to READ_AIR if expired
            if self.state.is_in_focus(umo):
                # Current State: FOCUS
                await self.scheduler.trigger_view_cycle(umo, "focus", event, image_urls)
            else:
                # Current State: READ_AIR
                if read_air_enable:
                    await self.scheduler.trigger_view_cycle(
                        umo, "read_air", event, image_urls
                    )
