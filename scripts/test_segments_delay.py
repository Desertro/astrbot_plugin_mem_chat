import asyncio
import sys
import time
from pathlib import Path
from types import ModuleType


def _plugin_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _expected_delay(text: str) -> float:
    s = "".join(str(text or "").split())
    n = len(s)
    if n > 400:
        n = 400
    d = 0.3 + (n * 1.7 / 400.0)
    if d < 0.3:
        d = 0.3
    if d > 2.0:
        d = 2.0
    return d


class _FakeState:
    def __init__(self):
        self.last_sent_image_ids = {}


class _FakeDb:
    async def add_image_to_library(self, *args, **kwargs):
        return None


class _FakePlugin:
    def __init__(self):
        self.config = {}
        self.context = None
        self.db_mgr = _FakeDb()
        self.state = _FakeState()


class _FakeEvent:
    def __init__(self):
        self.sent = []

    async def send(self, chain):
        self.sent.append((time.perf_counter(), chain))


def _install_stubs():
    if "aiohttp" not in sys.modules:
        sys.modules["aiohttp"] = ModuleType("aiohttp")

    astrbot = ModuleType("astrbot")
    astrbot_api = ModuleType("astrbot.api")
    astrbot_api_event = ModuleType("astrbot.api.event")
    astrbot_api_msg = ModuleType("astrbot.api.message_components")
    astrbot_core = ModuleType("astrbot.core")
    astrbot_core_agent = ModuleType("astrbot.core.agent")
    astrbot_core_agent_tool = ModuleType("astrbot.core.agent.tool")
    astrbot_core_star = ModuleType("astrbot.core.star")
    astrbot_core_session_llm = ModuleType("astrbot.core.star.session_llm_manager")

    class AstrMessageEvent: ...

    class MessageChain:
        def __init__(self):
            self.chain = []

        def message(self, text: str):
            self.chain.append(("message", str(text)))
            return self

        def at(self, *, name: str = "", qq: str = ""):
            self.chain.append(("at", str(qq)))
            return self

        def get_plain_text(self) -> str:
            out = []
            for kind, val in self.chain:
                if kind == "message":
                    out.append(val)
                elif kind == "at":
                    out.append(f"@{val}")
            return "".join(out)

    class Image:
        def __init__(self, *, file: str | None = None, url: str | None = None):
            self.file = file
            self.url = url

        @classmethod
        def fromFileSystem(cls, path: str):
            return cls(file=path, url=path)

    class Poke:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class Record:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class Reply:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    async def register_file(_path: str):
        return "token"

    file_token_service = ModuleType("astrbot.core.file_token_service")
    file_token_service.register_file = register_file

    class ToolSet: ...

    class SessionServiceManager:
        @staticmethod
        def should_process_tts_request(_event) -> bool:
            return False

    astrbot_api_event.AstrMessageEvent = AstrMessageEvent
    astrbot_api_event.MessageChain = MessageChain
    astrbot_api_msg.Image = Image
    astrbot_api_msg.Poke = Poke
    astrbot_api_msg.Record = Record
    astrbot_api_msg.Reply = Reply

    astrbot_core.file_token_service = file_token_service
    astrbot_core_agent_tool.ToolSet = ToolSet
    astrbot_core_session_llm.SessionServiceManager = SessionServiceManager

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = astrbot_api
    sys.modules["astrbot.api.event"] = astrbot_api_event
    sys.modules["astrbot.api.message_components"] = astrbot_api_msg
    sys.modules["astrbot.core"] = astrbot_core
    sys.modules["astrbot.core.agent"] = astrbot_core_agent
    sys.modules["astrbot.core.agent.tool"] = astrbot_core_agent_tool
    sys.modules["astrbot.core.star"] = astrbot_core_star
    sys.modules["astrbot.core.star.session_llm_manager"] = astrbot_core_session_llm

    src_models = ModuleType("src.models")

    class DummyEvent:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    src_models.DummyEvent = DummyEvent
    sys.modules["src.models"] = src_models


async def main():
    plugin_root = _plugin_root()
    _install_stubs()
    sys_path_added = [str(plugin_root)]
    sys.path.insert(0, str(plugin_root))
    try:
        from src.llm_client import LLMClient
    finally:
        for _ in range(len(sys_path_added)):
            sys.path.pop(0)

    plugin = _FakePlugin()
    client = LLMClient(plugin)
    event = _FakeEvent()

    final_reply = {
        "response": {
            "segments": [
                {"text": "hi"},
                {"text": "x" * 200},
                {"text": "y" * 500},
            ]
        },
        "control": {"decision": "reply_only"},
    }

    sleep_calls = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        sleep_calls.append(float(delay))
        return None

    asyncio.sleep = fake_sleep
    try:
        ok = await client._process_and_send_final_reply(
            final_reply=final_reply,
            event=event,
            task_type="at",
            quote_candidates=[],
            at_candidates=[],
            llm_contexts=[],
            umo="qq:group:1",
            user_msgs_to_save=[],
        )
    finally:
        asyncio.sleep = real_sleep

    if not ok:
        raise SystemExit("expected ok=True")
    if len(event.sent) != 3:
        raise SystemExit(f"expected 3 sends, got {len(event.sent)}")
    if len(sleep_calls) != 2:
        raise SystemExit(f"expected 2 sleeps, got {len(sleep_calls)}")

    exp1 = _expected_delay("hi")
    exp2 = _expected_delay("x" * 200)
    got1, got2 = sleep_calls
    if abs(got1 - exp1) > 1e-6:
        raise SystemExit(f"sleep1 mismatch: got={got1} exp={exp1}")
    if abs(got2 - exp2) > 1e-6:
        raise SystemExit(f"sleep2 mismatch: got={got2} exp={exp2}")

    print(
        "OK",
        {
            "sends": len(event.sent),
            "sleep_calls": sleep_calls,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
