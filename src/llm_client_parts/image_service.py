import logging
import os
import random
import re
import sqlite3
import struct
import tempfile
import time
from collections import deque
from collections.abc import Callable
from typing import Protocol
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger("astrbot")

# Cached meme vectors keyed by emotion: (index mtime, model name, vectors).
_MEME_VEC_CACHE: dict[
    str, tuple[tuple[int, int], str, list[tuple[str, list[float]]]]
] = {}
_MEME_TAG_CACHE: dict[str, tuple[tuple[int, int], dict]] = {}

# Words the bot writes in its intent text, mapped to the closed-vocabulary
# emotion labels stored in meme_tags. Literal containment alone missed cases
# like "困得不行" not matching the label "困倦".
_EMOTION_HINTS = {
    "开心": ("开心", "高兴", "快乐", "乐", "爽", "笑", "兴奋", "激动", "热血"),
    "得意": ("得意", "骄傲", "炫耀", "嘚瑟", "牛", "装厉害"),
    "喜欢": ("喜欢", "爱", "心动", "可爱", "萌"),
    "撒娇": ("撒娇", "卖萌", "娇", "求"),
    "佩服": ("佩服", "厉害", "服了", "牛逼", "respect"),
    "愤怒": ("愤怒", "生气", "火大", "恼", "怒", "气"),
    "嫌弃": ("嫌弃", "嫌", "鄙视", "看不上", "恶心"),
    "无语": ("无语", "没话说", "懒得", "服了", "绝了"),
    "委屈": ("委屈", "憋屈", "可怜", "被欺负", "难过", "伤心", "悲", "哭", "失落"),
    "焦虑": ("焦虑", "慌", "急", "紧张", "不安"),
    "嘲讽": ("嘲讽", "讽", "嘲", "挖苦", "阴阳怪气", "坏笑", "阴险", "不怀好意"),
    "调侃": ("调侃", "吐槽", "开玩笑", "逗", "玩梗"),
    "装傻": ("装傻", "装作不懂", "装不知道"),
    "摆烂": ("摆烂", "不管了", "随便", "爱咋咋", "没招"),
    "困倦": ("困", "睡", "累", "乏", "躺平", "熬夜"),
    "懵": (
        "懵",
        "没懂",
        "啥",
        "迷",
        "困惑",
        "不明白",
        "当机",
        "宕机",
        "脑子",
        "死机",
        "烧脑",
        "反应不过来",
    ),
    "平静": ("平静", "淡定", "无所谓", "佛"),
    "害羞": ("害羞", "羞", "脸红", "不好意思", "被夸", "尴尬", "无措", "社死"),
    "哭穷": ("哭穷", "没钱", "穷", "要钱", "转账", "养我"),
}


# Model used to describe a newly captured meme before indexing it.
_MEME_VISION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
_MEME_VISION_PROMPT = (
    "这张表情包适合在什么聊天场合发？用一句中文写出：(1)别人说什么话时可以用它回应 "
    "(2)它传达的情绪 (3)图上的文字（没有就写无）。30-50字，直接输出，不要解释。"
)


def _cosine(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of two equal-length vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Similarity in [-1, 1], or 0.0 when either vector is unusable.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a**0.5 * norm_b**0.5)


class _HasSenderId(Protocol):
    def get_sender_id(self) -> str | None: ...


class ImageService:
    def __init__(
        self,
        get_conf: Callable[[str, object], object],
        *,
        get_day: Callable[[], str] | None = None,
        get_cwd: Callable[[], str] | None = None,
    ):
        self._get_conf = get_conf
        self._get_day = get_day or (lambda: time.strftime("%Y-%m-%d"))
        self._get_cwd = get_cwd or os.getcwd
        self._user_daily_images: dict[str, dict[str, set[str]]] = {}
        self._recent_meme_paths: dict[str, deque[str]] = {}

    def normalize_image_key(self, url: str) -> str:
        u = (url or "").strip("`\"' ")
        m = re.search(r"pixiv\.net/(?:en/)?artworks/(\d+)", u)
        if m:
            return f"pixiv:{m.group(1)}"
        if "i.pximg.net" in u:
            p = urlparse(u)
            return f"pximg:{p.path}"
        if "pbs.twimg.com" in u:
            p = urlparse(u)
            return f"pbs:{p.path}"
        p = urlparse(u)
        return f"url:{p.netloc}{p.path}"

    async def pick_meme_image_file(
        self,
        emotion: str = "",
        query: str = "",
        *,
        conversation_id: str = "",
        selected_meme: dict[str, str] | None = None,
    ) -> str | None:
        """Pick the meme that best matches an intent, globally or per directory.

        Args:
            emotion: Emotion directory to restrict to, or empty to search all.
            query: Free-form intent used to rank candidates.
            conversation_id: Conversation whose recent selections are avoided.
            selected_meme: Optional output for original path, description and OCR.

        Returns:
            Path to a send-ready image, or None when nothing matches.
        """
        if selected_meme is not None:
            selected_meme.clear()
        emo = (emotion or "").strip()
        text = (query or "").strip()
        memes_dir = os.path.join(self._get_cwd(), "data", "memes_data", "memes")

        if emo:
            emotion_path = os.path.join(memes_dir, emo)
            if not os.path.exists(emotion_path):
                return None
            memes = [
                f
                for f in os.listdir(emotion_path)
                if str(f).lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))
            ]
            if not memes:
                return None
            chosen: str | None = None
            if text:
                vectors = self._load_meme_vectors(emo)
                if vectors:
                    qvec = await self._embed_query(text)
                    if qvec:
                        available = set(memes)
                        scored = [
                            (_cosine(qvec, vec), path)
                            for path, vec in vectors
                            if os.path.basename(path) in available
                            and os.path.exists(path)
                        ]
                        if scored:
                            scored.sort(key=lambda item: item[0], reverse=True)
                            chosen = random.choice(scored[:3])[1]
            if not chosen:
                chosen = os.path.join(emotion_path, random.choice(memes))
            if self.is_blank_image(chosen):
                logger.warning(f"[mem_chat] Blank meme in library: {chosen}")
                return ""
            ready = self._shrink_meme_for_send(chosen)
            if ready and selected_meme is not None:
                selected_meme.update(self._load_meme_tags().get(chosen, {}))
                selected_meme["path"] = chosen
            return ready

        # Global mode: rank every indexed meme by the intent description.
        if not text:
            return None
        vectors = self._load_meme_vectors("")
        if not vectors:
            return None
        qvec = await self._embed_query(text)
        if not qvec:
            return None
        tags = self._load_meme_tags()
        want_attack = any(
            w in text for w in ("怼", "吵", "骂", "嘲", "讽", "挑衅", "反击", "不服")
        )
        want_friendly = any(
            w in text for w in ("夸", "谢", "喜欢", "可爱", "撒娇", "安慰", "祝")
        )
        # Being cute is not evidence that the other speaker praised the bot.
        if any(w in text for w in ("被夸", "被表扬", "受到称赞")):
            want_reply = "被夸"
        elif any(w in text for w in ("被怼", "被骂", "被嘲", "被讽", "被挑衅")):
            want_reply = "被怼"
        elif any(w in text for w in ("被问", "被提问", "回答问题")):
            want_reply = "被问"
        elif any(w in text for w in ("被催", "催促")):
            want_reply = "被催"
        elif any(w in text for w in ("被拒", "遭到拒绝")):
            want_reply = "被拒"
        else:
            want_reply = ""
        want_loud = any(
            w in text for w in ("激动", "炸", "疯", "狂", "爆", "整活", "搞怪")
        )
        want_quiet = any(w in text for w in ("安静", "淡定", "平静", "认真", "正经"))
        scored_all = []
        for path, vec in vectors:
            if not os.path.isfile(path) or len(qvec) != len(vec):
                continue
            similarity = _cosine(qvec, vec)
            score = similarity
            tag = tags.get(path, {})
            ocr = tag.get("ocr_text", "").strip()
            exact_ocr = len(ocr) >= 3 and ocr in text
            if exact_ocr:
                score += 0.20
            # Tags are noisy auxiliary evidence. Count emotion agreement once
            # and never reward an empty reply type or loose character overlap.
            if any(
                word
                and (
                    word in text or any(h in text for h in _EMOTION_HINTS.get(word, ()))
                )
                for word in tag.get("emotion", "").split("、")
            ):
                score += 0.04
            stance = tag.get("stance", "")
            if want_attack:
                score += 0.04 if stance == "攻击" else -0.04
            elif want_friendly and stance == "攻击":
                score -= 0.06
            if want_reply and tag.get("reply_to") == want_reply:
                score += 0.04
            if want_loud and tag.get("energy") == "夸张":
                score += 0.02
            elif want_quiet and tag.get("energy") == "原图":
                score += 0.02
            scored_all.append((score, path, similarity, exact_ocr))
        if not scored_all:
            return None
        best_similarity = max(item[2] for item in scored_all)
        # Diversity is limited to semantically competitive candidates, never
        # the whole library. Explicitly quoted image text may override cosine.
        ranked = sorted(
            (
                item
                for item in scored_all
                if item[2] >= best_similarity - 0.08 or item[3]
            ),
            reverse=True,
        )
        if any(item[3] for item in ranked):
            ranked = [item for item in ranked if item[3]]
        best_score = ranked[0][0]
        candidates = [item for item in ranked[:24] if item[0] >= best_score - 0.08]
        scope = conversation_id or "*"
        if scope not in self._recent_meme_paths:
            if len(self._recent_meme_paths) >= 128:
                self._recent_meme_paths.pop(next(iter(self._recent_meme_paths)))
            self._recent_meme_paths[scope] = deque(maxlen=24)
        recent = self._recent_meme_paths[scope]
        while candidates:
            unused = [item for item in candidates if item[1] not in recent]
            if unused:
                chosen = random.choice(unused)
            else:
                # A small relevant pool may be exhausted. Reuse its oldest
                # selection instead of broadening into unrelated emotions.
                chosen = min(candidates, key=lambda item: list(recent).index(item[1]))
            score, picked, similarity, _ = chosen
            if self.is_blank_image(picked):
                logger.warning("[mem_chat] Blank meme in library: %s", picked)
                candidates.remove(chosen)
                continue
            ready = self._shrink_meme_for_send(picked)
            if not ready:
                candidates.remove(chosen)
                continue
            if picked in recent:
                recent.remove(picked)
            recent.append(picked)
            if selected_meme is not None:
                selected_meme.update(tags.get(picked, {}))
                selected_meme["path"] = picked
            logger.info(
                "[mem_chat] Meme selected: conversation=%s intent=%r file=%s cosine=%.3f score=%.3f candidates=%d fresh=%d",
                scope,
                text[:100],
                picked,
                similarity,
                score,
                len(candidates),
                len(unused),
            )
            return ready
        return None

    def _load_meme_tags(self) -> dict[str, dict[str, str]]:
        """Load the fine-grained tags produced by the labelling pass.

        The tags live in their own table so the legacy meme_index stays intact.
        Missing table or file simply means tag-aware ranking is skipped.

        Returns:
            Mapping of image path to its tag fields. Empty when unavailable.
        """
        db_path = os.path.join(self._get_cwd(), "data", "memes_data", "meme_index.db")
        if not os.path.exists(db_path):
            return {}
        try:
            mtime = (
                os.stat(db_path).st_mtime_ns,
                os.stat(db_path + "-wal").st_mtime_ns
                if os.path.exists(db_path + "-wal")
                else 0,
            )
        except OSError:
            return {}
        cached = _MEME_TAG_CACHE.get(db_path)
        if cached and cached[0] == mtime:
            return cached[1]
        out: dict[str, dict[str, str]] = {}
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                rows = con.execute(
                    "SELECT t.path, t.emotion, t.intensity, t.stance, "
                    "t.target, t.ocr_text, COALESCE(i.description, ''), "
                    "COALESCE(t.energy, ''), COALESCE(t.reply_to, '') "
                    "FROM meme_tags t "
                    "LEFT JOIN meme_index i ON i.path = t.path"
                ).fetchall()
            finally:
                con.close()
        except Exception:
            # The table does not exist yet on installs that never ran labelling.
            return {}
        for (
            path,
            emotion,
            intensity,
            stance,
            target,
            ocr,
            desc,
            energy,
            reply_to,
        ) in rows:
            out[path] = {
                "emotion": emotion or "",
                "intensity": intensity or "",
                "stance": stance or "",
                "target": target or "",
                "ocr_text": ocr or "",
                "description": desc or "",
                "energy": energy or "",
                "reply_to": reply_to or "",
            }
        _MEME_TAG_CACHE[db_path] = (mtime, out)
        return out

    def _load_meme_vectors(self, emotion: str = "") -> list[tuple[str, list[float]]]:
        """Load indexed meme vectors, globally or for one emotion directory.

        Args:
            emotion: Emotion directory name; empty means every directory.

        Returns:
            List of (image path, embedding) pairs. Empty when the index file is
            missing or was built with a different embedding model.
        """
        db_path = os.path.join(self._get_cwd(), "data", "memes_data", "meme_index.db")
        model = str(self._get_conf("embedding_openai_model", "") or "")
        if not model or not os.path.exists(db_path):
            return []
        try:
            mtime = (
                os.stat(db_path).st_mtime_ns,
                os.stat(db_path + "-wal").st_mtime_ns
                if os.path.exists(db_path + "-wal")
                else 0,
            )
        except OSError:
            return []

        cache_key = f"{db_path}:{emotion or '*'}"
        cached = _MEME_VEC_CACHE.get(cache_key)
        if cached and cached[0] == mtime and cached[1] == model:
            return cached[2]

        rows: list[tuple[str, bytes, int]] = []
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                if emotion:
                    rows = con.execute(
                        "SELECT path, vector, dim FROM meme_index "
                        "WHERE emotion = ? AND model = ?",
                        (emotion, model),
                    ).fetchall()
                else:
                    rows = con.execute(
                        "SELECT path, vector, dim FROM meme_index WHERE model = ?",
                        (model,),
                    ).fetchall()
            finally:
                con.close()
        except Exception as exc:
            logger.warning(f"[mem_chat] Failed to read meme index: {exc}")
            return []

        out: list[tuple[str, list[float]]] = []
        for path, blob, dim in rows:
            try:
                n = int(dim or 0)
                if not n or not blob:
                    continue
                out.append((path, list(struct.unpack(f"<{n}f", blob))))
            except Exception:
                continue
        _MEME_VEC_CACHE[cache_key] = (mtime, model, out)
        return out

    async def _embed_query(self, text: str) -> list[float] | None:
        """Embed a query string with the configured embedding provider.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector, or None on failure.
        """
        base = str(self._get_conf("embedding_openai_base_url", "") or "").rstrip("/")
        key = str(self._get_conf("embedding_openai_api_key", "") or "")
        model = str(self._get_conf("embedding_openai_model", "") or "")
        if not (base and key and model):
            return None
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{base}/embeddings",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": model, "input": [text[:1500]]},
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
            return data["data"][0]["embedding"]
        except Exception as exc:
            logger.warning(f"[mem_chat] Meme query embed failed: {exc}")
            return None

    async def capture_meme_image(self, image_url: str, emotion: str) -> str | None:
        """Download a user-supplied meme, describe it, index it and store it.

        Args:
            image_url: Remote URL or local cached path from the QQ message.
            emotion: Target emotion directory chosen by the LLM.

        Returns:
            Stored file path on success, None when the image is unusable or the
            emotion directory does not exist.
        """
        import base64
        import hashlib
        import io

        from PIL import Image as PILImage

        emo = (emotion or "").strip()
        if not emo or not (image_url or "").strip():
            return None
        memes_dir = os.path.join(self._get_cwd(), "data", "memes_data", "memes")
        emotion_path = os.path.join(memes_dir, emo)
        if not os.path.isdir(emotion_path):
            return None

        base = str(self._get_conf("embedding_openai_base_url", "") or "").rstrip("/")
        key = str(self._get_conf("embedding_openai_api_key", "") or "")
        emb_model = str(self._get_conf("embedding_openai_model", "") or "")
        if not (base and key and emb_model):
            return None

        # AstrBot caches incoming pictures on disk, so accept a local path too.
        local_path = ""
        if image_url.startswith("file://"):
            local_path = image_url[7:]
        elif image_url.startswith("/") and os.path.isfile(image_url):
            local_path = image_url

        try:
            if local_path:
                with open(local_path, "rb") as fh:
                    raw = fh.read()
            else:
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(image_url) as resp:
                        if resp.status != 200:
                            return None
                        raw = await resp.read()
            if not raw or len(raw) > 12 * 1024 * 1024:
                return None
        except Exception as exc:
            logger.warning(f"[mem_chat] Meme download failed: {exc}")
            return None

        digest = hashlib.sha256(raw).hexdigest()[:32]
        try:
            for d in os.listdir(memes_dir):
                dpath = os.path.join(memes_dir, d)
                if not os.path.isdir(dpath):
                    continue
                for name in os.listdir(dpath):
                    if name.startswith(digest):
                        logger.info(f"[mem_chat] Meme already stored: {d}/{name}")
                        return os.path.join(dpath, name)
        except Exception:
            pass

        try:
            with PILImage.open(io.BytesIO(raw)) as im:
                animated = (
                    getattr(im, "is_animated", False) or getattr(im, "n_frames", 1) > 1
                )
                work = im.convert("RGB")
                work.thumbnail((336, 336), PILImage.LANCZOS)
                buf = io.BytesIO()
                work.save(buf, "JPEG", quality=88)
                jpeg = buf.getvalue()
            suffix = ".gif" if animated else ".jpg"
        except Exception as exc:
            logger.warning(f"[mem_chat] Meme decode failed: {exc}")
            return None

        b64 = base64.b64encode(jpeg).decode()
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            timeout = aiohttp.ClientTimeout(total=45)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{base}/chat/completions",
                    headers=headers,
                    json={
                        "model": _MEME_VISION_MODEL,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{b64}"
                                        },
                                    },
                                    {"type": "text", "text": _MEME_VISION_PROMPT},
                                ],
                            }
                        ],
                        "max_tokens": 150,
                    },
                ) as resp:
                    if resp.status != 200:
                        return None
                    chat = await resp.json()
                desc = str(chat["choices"][0]["message"]["content"]).strip()
                async with session.post(
                    f"{base}/embeddings",
                    headers=headers,
                    json={"model": emb_model, "input": [desc]},
                ) as resp:
                    if resp.status != 200:
                        return None
                    emb = await resp.json()
            vec = emb["data"][0]["embedding"]
        except Exception as exc:
            logger.warning(f"[mem_chat] Meme describe/index failed: {exc}")
            return None

        target = os.path.join(emotion_path, digest + suffix)
        try:
            if suffix == ".gif":
                with open(target, "wb") as fh:
                    fh.write(raw)
            else:
                with open(target, "wb") as fh:
                    fh.write(jpeg)
            db_path = os.path.join(
                self._get_cwd(), "data", "memes_data", "meme_index.db"
            )
            con = sqlite3.connect(db_path)
            con.execute(
                """CREATE TABLE IF NOT EXISTS meme_index (
                       path TEXT PRIMARY KEY,
                       emotion TEXT,
                       description TEXT,
                       vector BLOB,
                       dim INTEGER,
                       model TEXT,
                       updated_at REAL
                   )"""
            )
            con.execute(
                "INSERT OR REPLACE INTO meme_index VALUES(?,?,?,?,?,?,?)",
                (
                    target,
                    emo,
                    desc,
                    struct.pack(f"<{len(vec)}f", *vec),
                    len(vec),
                    emb_model,
                    time.time(),
                ),
            )
            con.commit()
            con.close()
        except Exception as exc:
            logger.warning(f"[mem_chat] Meme store failed: {exc}")
            return None

        logger.info(f"[mem_chat] Captured meme -> {emo}/{digest}{suffix} | {desc[:60]}")
        return target

    def is_blank_image(self, path: str) -> bool:
        """Report whether an image would show up as a blank black rectangle.

        Stickers occasionally come out all black (a truncated download, a
        transparency-to-JPEG flatten, or a bad frame grab). Sending one looks
        like a glitch, so the caller skips the image instead. Only the visible
        pixels count: a fully transparent PNG renders as blank too.

        Args:
            path: Image file to inspect.

        Returns:
            True when the image is empty or effectively all black.
        """
        try:
            from PIL import Image as PILImage
        except Exception:
            return False
        try:
            with PILImage.open(path) as im:
                im.seek(0)
                rgba = im.convert("RGBA")
                if max(rgba.size) > 256:
                    rgba = rgba.resize((96, 96))
                px = list(rgba.getdata())
        except Exception as exc:
            logger.warning(f"[mem_chat] Failed to inspect meme {path}: {exc}")
            return False
        if not px:
            return True
        visible = [p for p in px if p[3] > 10]
        if not visible:
            return True
        mean = sum(0.299 * r + 0.587 * g + 0.114 * b for r, g, b, _ in visible) / len(
            visible
        )
        return mean < 6.0

    def _shrink_meme_for_send(self, path: str, max_side: int = 240) -> str:
        """Downscale a still meme image to chat sticker size.

        Args:
            path: Source meme file path.
            max_side: Longest allowed side in pixels.

        Returns:
            Path to a temporary downscaled copy, or the original path when the
            image is animated, already small enough, or cannot be processed.
        """
        try:
            from PIL import Image as PILImage
        except Exception:
            return path
        try:
            with PILImage.open(path) as im:
                if getattr(im, "is_animated", False) or getattr(im, "n_frames", 1) > 1:
                    return path
                im.load()
                if max(im.size) <= max_side:
                    return path
                has_alpha = im.mode in ("RGBA", "LA") or (
                    im.mode == "P" and "transparency" in im.info
                )
                work = im.convert("RGBA" if has_alpha else "RGB")
                work.thumbnail((max_side, max_side), PILImage.LANCZOS)
                suffix = ".png" if has_alpha else ".jpg"
                fd, tmp_path = tempfile.mkstemp(prefix="meme_send_", suffix=suffix)
                os.close(fd)
                if has_alpha:
                    work.save(tmp_path, "PNG", optimize=True)
                else:
                    work.save(tmp_path, "JPEG", quality=88, optimize=True)
                return tmp_path
        except Exception as exc:
            logger.warning(f"[mem_chat] Failed to shrink meme {path}: {exc}")
            return path

    def get_user_daily_set(self, event: _HasSenderId | None):
        try:
            if not event:
                return None
            uid = event.get_sender_id()
            if not uid:
                return None
            day = self._get_day()
            d = self._user_daily_images.setdefault(day, {})
            for k in list(self._user_daily_images.keys()):
                if k != day:
                    del self._user_daily_images[k]
            s = d.setdefault(uid, set())
            return s
        except Exception:
            return None

    async def prepare_llm_image_contents(self, image_urls: list[str]) -> list[dict]:
        contents = []
        try:
            import base64
            import io

            from PIL import Image
        except Exception:
            return [
                {"type": "image_url", "image_url": {"url": u}}
                for u in (image_urls or [])
            ]

        max_kb = int(self._get_conf("mem_chat_llm_image_max_kb", 500))
        max_dim = int(self._get_conf("mem_chat_llm_image_max_dim", 1280))
        init_quality = int(self._get_conf("mem_chat_llm_image_quality_init", 85))
        min_quality = int(self._get_conf("mem_chat_llm_image_quality_min", 50))
        gif_max_frames = int(self._get_conf("mem_chat_llm_gif_max_frames", 4))
        gif_total_max_kb = int(self._get_conf("mem_chat_llm_gif_total_max_kb", 1000))

        async def _load_bytes(u: str) -> bytes | None:
            try:
                if not u:
                    return None
                s = str(u).strip()
                if s.startswith("`") and s.endswith("`") and len(s) >= 2:
                    s = s[1:-1].strip()
                if s.startswith("<") and s.endswith(">") and len(s) >= 2:
                    s = s[1:-1].strip()
                s = s.strip().strip("`").strip()
                if s.startswith("data:image/"):
                    try:
                        head, b64 = s.split(",", 1)
                        return base64.b64decode(b64)
                    except Exception:
                        return None
                if os.path.exists(s):
                    with open(s, "rb") as f:
                        return f.read()
                if s.startswith("http://") or s.startswith("https://"):
                    try:
                        async with aiohttp.ClientSession(trust_env=True) as session:
                            async with session.get(
                                s,
                                timeout=15,
                                allow_redirects=True,
                                headers={
                                    "User-Agent": "Mozilla/5.0",
                                    "Accept": "*/*",
                                },
                            ) as resp:
                                if resp.status == 200:
                                    ct = str(
                                        resp.headers.get("Content-Type") or ""
                                    ).lower()
                                    if "application/json" in ct:
                                        return None
                                    b = await resp.read()
                                    head = b.lstrip()[:1]
                                    if head in (b"{", b"[") and len(b) < 10240:
                                        return None
                                    return b
                    except Exception:
                        return None
                return None
            except Exception:
                return None

        def _guess_mime(data: bytes) -> str:
            try:
                if data.startswith(b"\x89PNG"):
                    return "image/png"
                if data.startswith(b"GIF8"):
                    return "image/gif"
                if data.startswith(b"RIFF") and b"WEBP" in data[:12]:
                    return "image/webp"
            except Exception:
                pass
            return "image/jpeg"

        def _compress_pil_to_jpeg(im: Image.Image, max_bytes: int) -> bytes:
            try:
                w, h = im.size
                if max(w, h) > max_dim:
                    ratio = max_dim / float(max(w, h))
                    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
                    try:
                        im = im.resize(new_size, Image.LANCZOS)
                    except Exception:
                        im = im.resize(new_size)
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
                q = init_quality
                step = 7
                last_bytes = b""
                for _ in range(10):
                    buf = io.BytesIO()
                    try:
                        im.save(buf, format="JPEG", quality=q, optimize=True)
                    except Exception:
                        im.save(buf, format="JPEG", quality=q)
                    b = buf.getvalue()
                    if len(b) <= max_bytes:
                        return b
                    last_bytes = b
                    q = max(min_quality, q - step)
                    if q == min_quality:
                        break
                return last_bytes
            except Exception:
                return b""

        def _extract_gif_frame_indices(frame_count: int, take: int) -> list[int]:
            if frame_count <= 0:
                return [0]
            if take <= 1 or frame_count == 1:
                return [0]
            take = min(take, frame_count)
            idxs: list[int] = []
            for i in range(take):
                pos = int(round(i * (frame_count - 1) / float(take - 1)))
                if not idxs or idxs[-1] != pos:
                    idxs.append(pos)
            return idxs

        for u in image_urls or []:
            b = await _load_bytes(u)
            if not b:
                contents.append({"type": "image_url", "image_url": {"url": u}})
                continue

            mime = _guess_mime(b)
            try:
                im = Image.open(io.BytesIO(b))
            except Exception:
                try:
                    b64 = base64.b64encode(b).decode("utf-8")
                    contents.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        }
                    )
                except Exception:
                    contents.append({"type": "image_url", "image_url": {"url": u}})
                continue

            is_animated = bool(getattr(im, "is_animated", False)) and (
                str(getattr(im, "format", "") or "").upper() == "GIF"
            )
            if is_animated:
                try:
                    frame_count = int(getattr(im, "n_frames", 1) or 1)
                except Exception:
                    frame_count = 1
                take = max(1, min(gif_max_frames, frame_count))
                idxs = _extract_gif_frame_indices(frame_count, take)
                total_budget = max(1, gif_total_max_kb) * 1024
                per_budget = max(1, total_budget // max(1, len(idxs)))
                per_budget = min(per_budget, max_kb * 1024)

                gif_contents: list[dict] = []
                for idx in idxs:
                    try:
                        im.seek(idx)
                        frame = im.convert("RGB")
                    except Exception:
                        continue
                    jb = _compress_pil_to_jpeg(frame, per_budget)
                    if not jb:
                        continue
                    try:
                        b64 = base64.b64encode(jb).decode("utf-8")
                        gif_contents.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            }
                        )
                    except Exception:
                        continue
                if gif_contents:
                    contents.extend(gif_contents)
                    continue

            target_bytes = max_kb * 1024
            if len(b) > target_bytes or mime != "image/jpeg":
                jb = _compress_pil_to_jpeg(im, target_bytes)
                if jb:
                    b = jb
                    mime = "image/jpeg"
            try:
                b64 = base64.b64encode(b).decode("utf-8")
                contents.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    }
                )
            except Exception:
                contents.append({"type": "image_url", "image_url": {"url": u}})

        return contents
