"""Controlled Codex CLI bridge for mem_chat."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from astrbot.api import logger

_THREAD_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_SESSION_NAME_RE = re.compile(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+")
DEFAULT_CODEX_CLI = (
    "/mnt/c/Users/Desertro/AppData/Local/Programs/Codex CLI/bin/codex.exe"
)


def parse_codex_events(stdout: str) -> dict:
    thread_id = ""
    messages: list[str] = []
    errors: list[str] = []
    unknown_types: set[str] = set()
    for raw_line in str(stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type") or "")
        if event_type == "thread.started":
            thread_id = str(event.get("thread_id") or "").strip()
            continue
        if event_type in {"turn.failed", "error"}:
            message = str(
                event.get("message") or event.get("error") or event.get("detail") or ""
            ).strip()
            if message:
                errors.append(message)
            continue
        if event_type != "item.completed":
            if event_type and event_type not in {"turn.started", "turn.completed"}:
                unknown_types.add(event_type)
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type == "agent_message":
            text = str(item.get("text") or "").strip()
            if text:
                messages.append(text)
        elif item_type == "error":
            message = str(item.get("message") or "").strip()
            if message:
                errors.append(message)
    return {
        "thread_id": thread_id,
        "message": messages[-1] if messages else "",
        "messages": messages,
        "errors": errors,
        "unknown_types": sorted(unknown_types),
    }


class CodexService:
    def __init__(self, config: dict, *, store_path: str | Path | None = None):
        self.config = config or {}
        self.store_path = Path(store_path or self._default_store_path())
        self._store_lock = asyncio.Lock()
        self._scope_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @staticmethod
    def _default_store_path() -> Path:
        return (
            Path(__file__).resolve().parents[4]
            / "plugin_data"
            / "mem_chat"
            / "codex_sessions.json"
        )

    def _get_conf(self, key: str, default=None):
        try:
            return self.config.get(key, default)
        except Exception:
            return default

    def _cli_path(self) -> str:
        configured = str(self._get_conf("codex_cli_path", "") or "").strip()
        if configured:
            return configured
        env_path = str(os.environ.get("CODEX_CLI_PATH", "") or "").strip()
        if env_path:
            return env_path
        if os.path.isfile(DEFAULT_CODEX_CLI):
            return DEFAULT_CODEX_CLI
        return shutil.which("codex") or "codex"

    def _workdir(self) -> str:
        return str(
            self._get_conf(
                "codex_workdir", r"\\wsl.localhost\Ubuntu\home\desertro"
            )
            or r"\\wsl.localhost\Ubuntu\home\desertro"
        ).strip()

    def _model(self) -> str:
        return str(
            self._get_conf("codex_model", "deepseek-v4.1-flash") or ""
        ).strip()

    def _sandbox(self) -> str:
        value = str(
            self._get_conf("codex_sandbox", "workspace-write") or "workspace-write"
        ).strip()
        if value not in {"read-only", "workspace-write", "danger-full-access"}:
            value = "workspace-write"
        return value

    def _timeout_sec(self) -> float:
        try:
            value = float(self._get_conf("codex_timeout_sec", 180) or 180)
        except (TypeError, ValueError):
            value = 180.0
        return max(15.0, min(600.0, value))

    def _allowed_uids(self) -> set[str]:
        allowed = self._get_conf("codex_allowed_uids", None)
        if not allowed:
            allowed = self._get_conf("private_chat_uids", []) or []
        if isinstance(allowed, str):
            allowed = re.split(r"[,;\s]+", allowed)
        return {str(item).strip() for item in allowed or [] if str(item).strip()}

    def is_authorized(self, event) -> bool:
        sender_id = str(getattr(event, "get_sender_id", lambda: "")() or "").strip()
        if not sender_id or sender_id not in self._allowed_uids():
            return False
        is_group = bool(getattr(event, "get_group_id", lambda: "")())
        if is_group and not bool(self._get_conf("codex_allow_groups", False)):
            return False
        return True

    @staticmethod
    def _scope_id(event) -> str:
        scope = str(getattr(event, "unified_msg_origin", "") or "").strip()
        if scope:
            return scope
        group_id = str(getattr(event, "get_group_id", lambda: "")() or "").strip()
        sender_id = str(getattr(event, "get_sender_id", lambda: "")() or "").strip()
        return group_id or sender_id

    @staticmethod
    def _normalize_session_name(value: str, *, fallback: str) -> str:
        name = _SESSION_NAME_RE.sub("-", str(value or "").strip()).strip("-")
        return (name or fallback)[:48]

    async def _load_store(self) -> dict:
        if not self.store_path.exists():
            return {"version": 1, "scopes": {}}
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("scopes"), dict):
                return data
        except Exception as exc:
            logger.warning(f"[mem_chat] Codex session store read failed: {exc}")
        return {"version": 1, "scopes": {}}

    async def _save_store(self, data: dict) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, self.store_path)

    async def _bind_session(
        self, scope: str, *, thread_id: str, session_name: str, cwd: str
    ) -> None:
        now = time.time()
        async with self._store_lock:
            data = await self._load_store()
            scopes = data.setdefault("scopes", {})
            scope_data = scopes.setdefault(scope, {"sessions": {}})
            sessions = scope_data.setdefault("sessions", {})
            previous = (
                sessions.get(thread_id)
                if isinstance(sessions.get(thread_id), dict)
                else {}
            )
            sessions[thread_id] = {
                "thread_id": thread_id,
                "name": session_name,
                "cwd": cwd,
                "created_at": float(previous.get("created_at") or now),
                "last_used_at": now,
            }
            scope_data["last_thread_id"] = thread_id
            await self._save_store(data)

    async def _resolve_session(
        self, scope: str, *, thread_id: str = "", session_name: str = ""
    ) -> dict:
        requested_thread = str(thread_id or "").strip()
        if requested_thread and not _THREAD_RE.match(requested_thread):
            return {"error": "thread_id 必须是 Codex 的 UUID，不能是任意字符串。"}
        async with self._store_lock:
            data = await self._load_store()
        scope_data = (data.get("scopes") or {}).get(scope) or {}
        sessions = scope_data.get("sessions") or {}
        if requested_thread:
            entry = sessions.get(requested_thread)
            if isinstance(entry, dict):
                return dict(entry)
            return {"thread_id": requested_thread, "name": "", "cwd": self._workdir()}
        requested_name = str(session_name or "").strip()
        if requested_name:
            for entry in sessions.values():
                if (
                    isinstance(entry, dict)
                    and str(entry.get("name") or "") == requested_name
                ):
                    return dict(entry)
            return {"error": f"当前会话没有名为“{requested_name}”的 Codex 线程。"}
        last_thread = str(scope_data.get("last_thread_id") or "").strip()
        entry = sessions.get(last_thread) if last_thread else None
        if isinstance(entry, dict):
            return dict(entry)
        return {"error": "当前聊天还没有绑定 Codex 会话；请先用 action=new 发起新会话。"}

    async def list_sessions(self, scope: str) -> str:
        async with self._store_lock:
            data = await self._load_store()
        scope_data = (data.get("scopes") or {}).get(scope) or {}
        sessions = scope_data.get("sessions") or {}
        last_thread = str(scope_data.get("last_thread_id") or "")
        if not sessions:
            return "当前聊天还没有 Codex 会话。"
        lines = ["当前聊天绑定的 Codex 会话："]
        for thread_id, entry in sorted(
            sessions.items(),
            key=lambda item: float((item[1] or {}).get("last_used_at") or 0),
            reverse=True,
        ):
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "(未命名)")
            marker = " [最近使用]" if thread_id == last_thread else ""
            used = datetime.fromtimestamp(
                float(entry.get("last_used_at") or 0)
            ).strftime("%Y-%m-%d %H:%M")
            lines.append(f"- {name}{marker} | {thread_id} | {used}")
        return "\n".join(lines)

    def _build_new_command(self, prompt: str) -> list[str]:
        command = [
            self._cli_path(),
            "exec",
            "--skip-git-repo-check",
            "--json",
            "-C",
            self._workdir(),
            "-s",
            self._sandbox(),
            "-c",
            'approval_policy="never"',
            "--color",
            "never",
        ]
        if self._model():
            command.extend(["-m", self._model()])
        command.append(prompt)
        return command

    def _build_resume_command(self, thread_id: str, prompt: str) -> list[str]:
        command = [
            self._cli_path(),
            "exec",
            "resume",
            "--skip-git-repo-check",
            "--json",
            "-c",
            'approval_policy="never"',
            "-c",
            f'sandbox_mode="{self._sandbox()}"',
        ]
        if self._model():
            command.extend(["-m", self._model()])
        command.extend([thread_id, prompt])
        return command

    async def _run_cli(
        self, command: list[str], timeout: float
    ) -> tuple[int, str, str]:
        env = dict(os.environ)
        env.setdefault("NO_COLOR", "1")
        env.setdefault("RUST_LOG", "error")
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            await proc.wait()
            return -1, "", f"Codex 调用超时（{timeout:.0f} 秒）。"
        return (
            int(proc.returncode or 0),
            stdout_b.decode("utf-8", errors="replace"),
            stderr_b.decode("utf-8", errors="replace"),
        )

    async def execute_tool(
        self,
        event,
        *,
        action: str = "resume",
        prompt: str = "",
        thread_id: str = "",
        session_name: str = "",
    ) -> str:
        if not bool(self._get_conf("codex_enabled", True)):
            return "Codex 工具当前未启用。"
        if not self.is_authorized(event):
            return "Codex 工具仅允许配置中的 owner 在私聊使用。"
        scope = self._scope_id(event)
        if not scope:
            return "无法确定当前会话，Codex 调用已取消。"
        action_norm = str(action or "resume").strip().lower()
        if action_norm in {"start", "create", "new_task"}:
            action_norm = "new"
        elif action_norm in {"continue", "reply", "followup", "follow_up"}:
            action_norm = "resume"
        if action_norm == "list":
            return await self.list_sessions(scope)
        if action_norm not in {"new", "resume"}:
            return "action 只能是 new、resume 或 list。"
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            return "prompt 不能为空。"
        if len(prompt_text) > 12000:
            return "prompt 过长，请压缩到 12000 字符以内。"

        async with self._scope_locks[scope]:
            if action_norm == "new":
                name = self._normalize_session_name(
                    session_name,
                    fallback="alice-" + datetime.now().strftime("%Y%m%d-%H%M%S"),
                )
                command = self._build_new_command(prompt_text)
                expected_thread = ""
                action_label = "新会话"
            else:
                target = await self._resolve_session(
                    scope,
                    thread_id=thread_id,
                    session_name=session_name,
                )
                if target.get("error"):
                    return str(target["error"])
                expected_thread = str(target.get("thread_id") or "").strip()
                if not _THREAD_RE.match(expected_thread):
                    return "无法解析有效的 Codex thread_id。"
                name = self._normalize_session_name(
                    session_name or str(target.get("name") or ""),
                    fallback=expected_thread[:8],
                )
                command = self._build_resume_command(expected_thread, prompt_text)
                action_label = "继续会话"

            logger.info(
                f"[mem_chat] Codex exec start scope={scope} action={action_norm} "
                f"prompt_chars={len(prompt_text)}"
            )
            rc, stdout, stderr = await self._run_cli(command, self._timeout_sec())
            parsed = parse_codex_events(stdout)
            thread_id_out = str(parsed.get("thread_id") or expected_thread).strip()
            message = str(parsed.get("message") or "").strip()
            if not message:
                details = parsed.get("errors") or []
                detail_text = "\n".join(str(x) for x in details).strip()
                if not detail_text:
                    detail_text = str(stderr or stdout or "").strip()
                logger.warning(
                    f"[mem_chat] Codex exec failed scope={scope} rc={rc} "
                    f"detail={detail_text[:500]}"
                )
                return (
                    f"Codex {action_label}失败（退出码 {rc}）。\n"
                    f"{detail_text[:1500] or '没有返回可用内容。'}"
                )
            if not thread_id_out:
                return "Codex 有输出，但没有返回 thread_id，无法安全继承该会话。"
            await self._bind_session(
                scope,
                thread_id=thread_id_out,
                session_name=name,
                cwd=self._workdir(),
            )
            clipped = (
                message
                if len(message) <= 6000
                else message[:6000] + "\n[输出已截断]"
            )
            logger.info(
                f"[mem_chat] Codex exec done scope={scope} action={action_norm} "
                f"thread={thread_id_out} output_chars={len(message)}"
            )
            return (
                f"Codex {action_label}完成。\n"
                f"session_name: {name}\n"
                f"thread_id: {thread_id_out}\n\n"
                f"Codex 最终回复：\n{clipped}"
            )
