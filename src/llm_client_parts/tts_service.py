import os
import re
import uuid
import wave
from collections.abc import Callable


class TtsService:
    def __init__(self, get_data_path: Callable[[], str] | None = None):
        self._replacement_audio_cache: dict[str, str | None] = {}
        if get_data_path is None:
            try:
                from astrbot.core.utils.astrbot_path import (
                    get_astrbot_data_path as _get_astrbot_data_path,
                )

                get_data_path = _get_astrbot_data_path
            except Exception:
                get_data_path = os.getcwd
        self._get_data_path = get_data_path

    def text_is_speakable(self, text: str) -> bool:
        s = (text or "").strip()
        if not s:
            return False
        meaningful = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", s)
        return bool(meaningful)

    def cleanup_text(self, text: str) -> str:
        s = str(text or "")
        s = re.sub(r"\bTTS\s*:\s*(?:yes|no)\b", "", s, flags=re.IGNORECASE)
        s = re.sub(r"(https?://[^\s]+)", "", s)

        def _strip_brackets(pattern: str, input_text: str) -> str:
            def _repl(m: re.Match):
                inner = (m.group(1) or "").strip()
                if not inner:
                    return ""
                if len(inner) <= 30:
                    return ""
                return m.group(0)

            return re.sub(pattern, _repl, input_text)

        s = _strip_brackets(r"\(([^()]*)\)", s)
        s = _strip_brackets(r"（([^（）]*)）", s)
        s = _strip_brackets(r"\[([^\[\]]*)\]", s)
        s = _strip_brackets(r"【([^【】]*)】", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def strip_inline_at_tokens(self, text: str) -> str:
        """移除内联艾特占位符，避免 TTS 分支被占位符干扰。"""
        s = str(text or "")
        s = re.sub(r"<at:\w+>", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def split_text(self, text: str, max_chars: int = 120) -> list[str]:
        s = (text or "").strip()
        if not s:
            return []

        sentences = re.findall(r".+?(?:[。！？!?…]+|\n+|$)", s, flags=re.DOTALL)
        chunks: list[str] = []
        buf = ""
        for sent in sentences:
            seg = (sent or "").strip()
            if not seg:
                continue
            if not buf:
                buf = seg
                continue
            if len(buf) + 1 + len(seg) <= max_chars:
                buf = f"{buf} {seg}".strip()
                continue
            if buf:
                chunks.append(buf)
            buf = seg

        if buf:
            chunks.append(buf)

        normalized: list[str] = []
        for c in chunks:
            c = (c or "").strip()
            if not c:
                continue
            if len(c) <= max_chars:
                if self.text_is_speakable(c):
                    normalized.append(c)
                continue
            for i in range(0, len(c), max_chars):
                piece = c[i : i + max_chars].strip()
                if piece and self.text_is_speakable(piece):
                    normalized.append(piece)
        return normalized

    def get_replacement_audio_path(self, token: str) -> str | None:
        token = (token or "").strip()
        if not token:
            return None
        if token in self._replacement_audio_cache:
            return self._replacement_audio_cache[token]

        patterns: list[str] = []
        if token in ("邦邦卡邦", "邦吧咔邦"):
            patterns = ["pankapapan"]
        elif token in ("光啊", "よ光"):
            patterns = ["hikariyo"]

        if not patterns:
            self._replacement_audio_cache[token] = None
            return None

        data_path = self._get_data_path()
        roots = [
            os.path.join(data_path, "GPT-SoVITS-v2pro-20250604", "models"),
            os.path.join(data_path, "gpt_sovits", "models"),
        ]
        exts = (".wav", ".mp3", ".ogg", ".m4a", ".flac")
        found: str | None = None
        for root in roots:
            if not os.path.isdir(root):
                continue
            try:
                for dirpath, _, filenames in os.walk(root):
                    for fn in filenames:
                        fn_lower = fn.lower()
                        if not fn_lower.endswith(exts):
                            continue

                        for p in patterns:
                            if p.lower() in fn_lower:
                                found = os.path.join(dirpath, fn)
                                break
                        if found:
                            break
                    if found:
                        break
            except Exception:
                continue
            if found:
                break

        self._replacement_audio_cache[token] = found
        return found

    def build_parts(self, text: str) -> list[tuple[str, str]]:
        s = str(text or "")
        keys = ["邦邦卡邦", "邦吧咔邦", "光啊", "よ光"]
        pattern = "(" + "|".join(re.escape(k) for k in keys) + ")"
        parts: list[tuple[str, str]] = []
        last = 0
        for m in re.finditer(pattern, s):
            if m.start() > last:
                parts.append(("text", s[last : m.start()]))
            tok = m.group(1)
            audio_path = self.get_replacement_audio_path(tok)
            if audio_path:
                parts.append(("audio", audio_path))
            else:
                if tok in ("光啊", "よ光"):
                    parts.append(("text", "よ光"))
                else:
                    parts.append(("text", "邦邦卡邦"))
            last = m.end()
        if last < len(s):
            parts.append(("text", s[last:]))
        return parts

    def try_concat_wavs(self, wav_paths: list[str]) -> str | None:
        wav_paths = [p for p in (wav_paths or []) if isinstance(p, str) and p.strip()]
        if not wav_paths:
            return None
        if len(wav_paths) == 1:
            return wav_paths[0]
        if not all(os.path.exists(p) for p in wav_paths):
            return None

        if all(p.lower().endswith(".wav") for p in wav_paths):
            try:
                params = None
                frames: list[bytes] = []
                for p in wav_paths:
                    with wave.open(p, "rb") as wf:
                        p_params = wf.getparams()
                        if params is None:
                            params = p_params
                        elif (
                            p_params.nchannels != params.nchannels
                            or p_params.sampwidth != params.sampwidth
                            or p_params.framerate != params.framerate
                            or p_params.comptype != params.comptype
                            or p_params.compname != params.compname
                        ):
                            params = None
                            break
                        frames.append(wf.readframes(wf.getnframes()))

                if params:
                    temp_dir = os.path.join(self._get_data_path(), "temp")
                    os.makedirs(temp_dir, exist_ok=True)
                    out_path = os.path.join(
                        temp_dir,
                        f"memchat_tts_combined_{uuid.uuid4().hex}.wav",
                    )
                    with wave.open(out_path, "wb") as out_wf:
                        out_wf.setparams(params)
                        for fr in frames:
                            out_wf.writeframes(fr)
                    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                        return out_path
            except Exception:
                pass

        try:
            from pydub import AudioSegment

            audio: AudioSegment | None = None
            for p in wav_paths:
                seg = AudioSegment.from_file(p)
                audio = seg if audio is None else audio + seg

            if audio is None:
                return None

            temp_dir = os.path.join(self._get_data_path(), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            out_path = os.path.join(
                temp_dir,
                f"memchat_tts_combined_{uuid.uuid4().hex}.wav",
            )
            audio.export(out_path, format="wav")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return out_path
        except Exception:
            return None

        return None

    def ensure_min_wav_duration(self, wav_path: str, min_seconds: float = 1.2) -> str:
        """保证 wav 最小时长，优先在前面拼接“邦邦卡邦”替换音频。"""
        try:
            src = str(wav_path or "").strip()
            if not src or not os.path.exists(src) or not src.lower().endswith(".wav"):
                return wav_path
            if min_seconds <= 0:
                return wav_path

            def _wav_duration(path: str) -> float:
                try:
                    with wave.open(path, "rb") as wf:
                        fr = wf.getframerate()
                        if fr <= 0:
                            return 0.0
                        return wf.getnframes() / float(fr)
                except Exception:
                    return 0.0

            duration = _wav_duration(src)
            if duration >= float(min_seconds):
                return wav_path

            # 优先在开头拼接“邦邦卡邦”替换音频，让短句语音更自然可感知
            bong_audio = self.get_replacement_audio_path("邦邦卡邦")
            if bong_audio and os.path.exists(bong_audio):
                combined = self.try_concat_wavs([bong_audio, src])
                if combined and os.path.exists(combined):
                    if _wav_duration(combined) >= float(min_seconds):
                        return combined
                    src = combined

            # 兜底：若仍不足最小时长，则补静音防止平台显示 0s
            with wave.open(src, "rb") as wf:
                params = wf.getparams()
                nframes = wf.getnframes()
                framerate = params.framerate
                if framerate <= 0:
                    return src
                duration = nframes / float(framerate)
                if duration >= float(min_seconds):
                    return src
                target_frames = int(float(min_seconds) * framerate)
                need_frames = max(0, target_frames - nframes)
                if need_frames <= 0:
                    return src
                raw = wf.readframes(nframes)

            silence = b"\x00" * need_frames * params.sampwidth * params.nchannels
            out_dir = os.path.join(self._get_data_path(), "temp")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir,
                f"memchat_tts_minlen_{uuid.uuid4().hex}.wav",
            )
            with wave.open(out_path, "wb") as out_wf:
                out_wf.setparams(params)
                out_wf.writeframes(raw)
                out_wf.writeframes(silence)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return out_path
            return wav_path
        except Exception:
            return wav_path
