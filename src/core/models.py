import time
from typing import Any

from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata


class DummyEvent(AstrMessageEvent):
    def __init__(self, u, ctx):
        self.context = ctx
        # Optional runtime handles (can be injected by scheduler) for control actions.
        self.bot = None
        self.adapter = None
        platform_id = "dummy"
        message_type = MessageType.OTHER_MESSAGE
        session_id = u
        try:
            ms = MessageSession.from_str(u)
            platform_id = ms.platform_name
            message_type = ms.message_type
            session_id = ms.session_id
        except Exception:
            parts = u.split(":")
            if len(parts) >= 3:
                platform_id = parts[0]
                s1 = parts[1]
                try:
                    message_type = MessageType(s1)
                except Exception:
                    s1l = s1.lower()
                    if s1l in ("group", "groupmessage", "group_msg"):
                        message_type = MessageType.GROUP_MESSAGE
                    elif s1l in (
                        "friend",
                        "friendmessage",
                        "private",
                        "private_message",
                    ):
                        message_type = MessageType.FRIEND_MESSAGE
                session_id = parts[2]

        abm = AstrBotMessage()
        abm.type = message_type
        abm.self_id = "bot"
        abm.session_id = session_id
        abm.message_id = ""
        abm.message = []
        abm.message_str = ""
        abm.sender = MessageMember(user_id="system", nickname="System")
        if message_type == MessageType.GROUP_MESSAGE:
            abm.group_id = session_id

        meta = PlatformMetadata(name=platform_id, description="", id=platform_id)
        super().__init__("", abm, meta, session_id)
        self.unified_msg_origin = f"{platform_id}:{message_type.value}:{session_id}"

    async def send(self, message):
        return await self.context.send_message(self.unified_msg_origin, message)


class ReplyTask:
    def __init__(
        self,
        umo: str,
        priority: int,
        task_type: str,
        event=None,
        user_msgs_to_save: list[dict[str, Any]] | None = None,
        image_urls: list[str] | None = None,
        extra_args: dict[str, Any] | None = None,
    ):
        self.umo = umo
        self.priority = priority  # 0 = High (At), 10 = Low (Focus)
        self.task_type = task_type  # 'at', 'focus'
        self.event = event
        self.user_msgs_to_save = user_msgs_to_save or []
        self.image_urls = image_urls or []
        self.extra_args = extra_args or {}
        self.created_at = time.time()  # Timestamp for timeout check

    def __lt__(self, other):
        return self.priority < other.priority


__all__ = ["DummyEvent", "ReplyTask"]
