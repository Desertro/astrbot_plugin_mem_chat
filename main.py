import asyncio
import glob
import os
import re

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import At, AtAll, Reply

try:
    from .src.core.scheduler import Scheduler
    from .src.core.state import MemChatState
    from .src.db.database import DatabaseManager
    from .src.llm_client import LLMClient
    from .src.media.manager import MediaManager
except ImportError:
    from src.core.scheduler import Scheduler
    from src.core.state import MemChatState
    from src.db.database import DatabaseManager
    from src.llm_client import LLMClient
    from src.media.manager import MediaManager


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

        self._mem_db_path = os.path.join(os.path.dirname(__file__), "mem_chat_embeddings.db")

        # Initialize Managers
        self.media_mgr = MediaManager(
            self.config, db_path=self._mem_db_path, get_conf=self.config.get
        )
        self.db_mgr = DatabaseManager(self._mem_db_path, self.config)
        self.llm_client = LLMClient(self)
        self.scheduler = Scheduler(self, self.context, self.config, self.state, self.db_mgr, self.llm_client)

        asyncio.create_task(self._startup_init())

        logger.info(f"[mem_chat] Plugin initialized. DB path: {self._mem_db_path}")

    def __del__(self):
        if hasattr(self, "media_mgr") and self.media_mgr:
            try:
                self.media_mgr.close()
            except Exception:
                pass
        if hasattr(self, "db_mgr") and self.db_mgr:
            self.db_mgr.close()

    async def _startup_init(self):
        await self._purge_persona_vectors()
        await self._check_and_update_knowledge()

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
                with open(os.path.join(knowledge_dir, "README.md"), "w", encoding="utf-8") as f:
                    f.write("Place your .txt or .md files here. They will be automatically imported into the knowledge base.")

            files = glob.glob(os.path.join(knowledge_dir, "*.txt")) + glob.glob(os.path.join(knowledge_dir, "*.md"))
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
                        if c:
                            combined_content += c
                            file_contents[fp] = c
                except Exception as e:
                    logger.error(f"[mem_chat] Failed to read {fp}: {e}")

            if not combined_content:
                return

            import hashlib
            h = hashlib.md5(combined_content.encode("utf-8")).hexdigest()

            # Check DB meta first
            last_hash = self.db_mgr.get_meta("last_knowledge_hash")
            if h == last_hash:
                return

            logger.info("[mem_chat] Knowledge files changed. Performing incremental update...")

            model_name = str(self.config.get("embedding_openai_model", "text-embedding-3-small") or "text-embedding-3-small")

            # 1. Prepare target state
            target_chunks = {} # mid -> content
            for fp, content in file_contents.items():
                chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
                for chunk in chunks:
                    mid = self.db_mgr.hash_id(chunk, "global", model_name)
                    target_chunks[mid] = chunk
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
                chunk = target_chunks[mid]
                await self.db_mgr.get_or_compute_embedding(mid, chunk, "global", model_name, type="knowledge")
                count += 1

            await self.db_mgr.set_meta("last_knowledge_hash", h)
            self.state.set_last_knowledge_hash(h)
            logger.info(f"[mem_chat] Knowledge Update: Added {count}, Deleted {len(ids_to_delete)}, Total {len(target_ids)} entries.")

        except Exception as e:
            logger.error(f"[mem_chat] Failed to update knowledge: {e}")

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
        await self._check_and_update_knowledge()
        yield event.plain_result("已重新加载本地文件到数据库。")

    @filter.command("mem_chat test_json")
    async def cmd_mem_chat_test_json(self, event: AstrMessageEvent):
        umo = event.unified_msg_origin
        llm_cfg = self.llm_client._prepare_llm_config(umo, event)
        if not llm_cfg.get("use_custom_llm"):
            yield event.plain_result("未启用自定义 LLM（当前走 provider），无法测试强制 JSON 输出参数。")
            return

        cfg = llm_cfg["llm"]
        fmt = str(cfg.get("fmt") or "").lower()
        force = bool(self.config.get("force_llm_json_output"))

        prompt = (
            "只输出一个 JSON 对象，不要任何解释。\n"
            "输出："
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
            yield event.plain_result(f"JSON 测试通过。format={fmt} force_llm_json_output={force}")
        else:
            preview = text[:300].replace("\n", "\\n")
            yield event.plain_result(
                f"JSON 测试失败。format={fmt} force_llm_json_output={force}\n"
                f"解析错误：{err}\n"
                f"输出预览：{preview}"
            )

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

        model_name = str(self.config.get("embedding_openai_model", "text-embedding-3-small") or "text-embedding-3-small")
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
                k_path = os.path.join(os.path.dirname(__file__), "knowledge", "imported_kb.txt")
                # Ensure dir exists
                os.makedirs(os.path.dirname(k_path), exist_ok=True)
                with open(k_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n{text}")
            elif type_ == "persona":
                p_path = os.path.join(os.path.dirname(__file__), "personas", "personas.txt")
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
            msg += f"{i+1}. {preview}...\n"

        yield event.plain_result(msg)

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        umo = event.unified_msg_origin
        if not umo:
            return

        # Ignore private messages
        if not event.get_group_id():
            return

        # Enforce group whitelist from config
        group_id = event.get_group_id()
        try:
            group_whitelist = [str(g) for g in (self.config.get("group_whitelist", []) or [])]
            if group_whitelist:
                gid_str = str(group_id)
                if gid_str not in group_whitelist:
                    logger.info(f"[mem_chat] 拒绝非白名单群组 {gid_str} 的请求")
                    return
        except Exception:
            pass

        try:
            self.state.set_self_id(umo, event.get_self_id())
            self.state.add_session(umo)
            if not self.state.get_self_name(umo):
                try:
                    gid = event.get_group_id()
                    if gid:
                        group = await event.get_group(gid)
                        if group and getattr(group, "members", None):
                            self_id = str(event.get_self_id() or "")
                            for m in group.members or []:
                                try:
                                    if str(getattr(m, "user_id", "")) != self_id:
                                        continue
                                    name = str(getattr(m, "nickname", "") or "").strip()
                                    if name:
                                        self.state.set_self_name(umo, name)
                                    break
                                except Exception:
                                    continue
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

        if _should_ignore_sender(event.get_self_id(), self.state.get_self_id(umo), sender_id):
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
        try:
            setattr(event, "is_at_or_wake_command", is_at)
        except Exception:
            pass

        # 获取消息 ID
        msg_id = None
        try:
            msg_obj = getattr(event, "message_obj", None)
            if msg_obj:
                msg_id = getattr(msg_obj, "message_id", None)
        except Exception:
            pass

        # 2. Extract Text and Images
        # Delegate to MediaManager
        text_with_images = await self.media_mgr.get_text_with_images(event)
        image_urls = await self.media_mgr.extract_images_from_event(event)

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
        asyncio.create_task(self.db_mgr.ensure_message_embedded(text_with_images, gid, sender_info))

        # Check if we should summarize (every 50 messages)
        c = self.state.increment_summary_msg_count(umo)
        if c >= 50:
            self.state.reset_summary_msg_count(umo)
            sid = event.get_sender_id()
            if sid:
                asyncio.create_task(self.scheduler.summarize_session(umo, gid, sid, sender_name))

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

            base_delay = float(self.config.get("focus_mode_cycle", 15))
            min_delay = float(self.config.get("focus_mode_cycle_min", 10))
            after_reply_add = float(self.config.get("focus_mode_cycle_after_reply_add", 20))
            reward_ratio = self.config.get("focus_mode_cycle_at_reward_ratio", None)
            at_reduce = float(self.config.get("focus_mode_cycle_at_reduce", 20))
            try:
                reward_ratio = float(reward_ratio) if reward_ratio is not None else None
            except Exception:
                reward_ratio = None
            if reward_ratio is None:
                reward_ratio = (at_reduce / base_delay) if base_delay > 0 else 0.0
            max_delay = base_delay + max(0.0, after_reply_add)

            if umo not in self.state.focus_delays:
                self.state.set_focus_delay(umo, base_delay, min_val=min_delay)

            self.state.adjust_focus_delay(
                umo,
                -abs(base_delay * reward_ratio),
                min_val=min_delay,
                max_val=max_delay,
            )

            # Enter/Refresh Focus State
            focus_duration = int(self.config.get("focus_duration_sec", 100))
            self.state.set_focus(umo, focus_duration)

            # Prevent Core Reply
            try:
                event.should_call_llm(False)
            except Exception:
                pass

            logger.debug(f"[mem_chat] AT message received for {umo}. Triggering AT response.")
            await self.scheduler.trigger_view_cycle(umo, "focus" if was_in_focus else "at", event, image_urls)

        else:
            # Event: Normal Message
            # Action: Maintain current state cycle

            # is_in_focus() checks time and automatically downgrades to READ_AIR if expired
            if self.state.is_in_focus(umo):
                # Current State: FOCUS
                await self.scheduler.trigger_view_cycle(umo, "focus", event, image_urls)
            else:
                # Current State: READ_AIR
                await self.scheduler.trigger_view_cycle(umo, "read_air", event, image_urls)
