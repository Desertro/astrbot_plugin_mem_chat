import hashlib
import logging
import os
import sqlite3
import time
from urllib.parse import parse_qsl, urlparse

logger = logging.getLogger("astrbot")


class ImageVisionCache:
    _GLOBAL_GROUP_ID = "__global__"

    def __init__(
        self,
        db_path: str,
        get_conf,
        *,
        now=None,
    ):
        self._db_path = db_path
        self._get_conf = get_conf
        self._now = now or time.time
        self._conn: sqlite3.Connection | None = None
        self._init_conn()

    def _bucket_group_id(self, group_id: str) -> str:
        return self._GLOBAL_GROUP_ID

    def _init_conn(self):
        if self._conn:
            return
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        self._init_tables()

    def close(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _init_tables(self):
        if not self._conn:
            return
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS image_vision_cache (
                group_id TEXT NOT NULL,
                image_key TEXT NOT NULL,
                orig_url TEXT,
                file_token TEXT,
                mime_hint TEXT,
                content_sha1 TEXT,
                caption TEXT,
                caption_model TEXT,
                status TEXT DEFAULT 'ready',
                err TEXT,
                first_seen_ts REAL,
                last_seen_ts REAL,
                seen_count INTEGER DEFAULT 0,
                last_sender_id TEXT,
                last_msg_id TEXT,
                PRIMARY KEY(group_id, image_key)
            )
            """
        )
        try:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_ivc_last_seen ON image_vision_cache(group_id, last_seen_ts)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_ivc_seen ON image_vision_cache(group_id, seen_count)"
            )
        except Exception:
            pass
        cur.close()
        try:
            self._conn.commit()
        except Exception:
            pass

    def _is_enabled(self) -> bool:
        try:
            return bool(self._get_conf("vision_image_cache_enable", True))
        except Exception:
            return True

    def _ttl_sec(self) -> float:
        try:
            days = int(self._get_conf("vision_image_cache_ttl_days", 30))
        except Exception:
            days = 30
        return float(max(1, days)) * 86400.0

    def _max_records(self) -> int:
        try:
            v = int(self._get_conf("vision_image_cache_max_records", 20000))
            return max(100, v)
        except Exception:
            return 20000

    def _canonicalize_url(self, url: str) -> str:
        u = (url or "").strip().strip("`").strip()
        if not u:
            return ""
        try:
            p = urlparse(u)
            if not p.scheme or not p.netloc:
                return u
            host = p.netloc.lower()
            path = p.path or ""
            q = []
            try:
                q = parse_qsl(p.query, keep_blank_values=True)
            except Exception:
                q = []
            if "multimedia.nt.qq.com.cn" in host:
                keep = []
                for k, v in q:
                    if k in ("appid", "rkey"):
                        keep.append((k, v))
                q = keep
            q.sort()
            query = "&".join([f"{k}={v}" for k, v in q]) if q else ""
            return f"{p.scheme.lower()}://{host}{path}" + (f"?{query}" if query else "")
        except Exception:
            return u

    def normalize_image_key(
        self,
        *,
        group_id: str,
        url: str | None,
        file_token: str | None,
    ) -> tuple[str, str | None]:
        ft = str(file_token or "").strip().strip("`").strip()
        if ft:
            low = ft.lower()
            ext = os.path.splitext(low)[1].lstrip(".")
            mime_hint = None
            if ext in ("gif",):
                mime_hint = "image/gif"
            elif ext in ("jpg", "jpeg"):
                mime_hint = "image/jpeg"
            elif ext in ("png",):
                mime_hint = "image/png"
            elif ext in ("webp",):
                mime_hint = "image/webp"
            elif ext in ("bmp",):
                mime_hint = "image/bmp"
            return f"qqfile:{low}", mime_hint

        cu = self._canonicalize_url(url or "")
        if cu:
            h = hashlib.sha1(cu.encode("utf-8")).hexdigest()
            return f"urlsha1:{h}", None
        return "unknown", None

    def get_caption(
        self,
        *,
        group_id: str,
        image_key: str,
    ) -> str | None:
        if not self._is_enabled():
            return None
        if not self._conn:
            self._init_conn()
        if not self._conn:
            return None
        gid = self._bucket_group_id(group_id)
        ik = str(image_key or "").strip()
        if not gid or not ik:
            return None
        ttl = self._ttl_sec()
        now_ts = self._now()
        try:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT caption, status, last_seen_ts FROM image_vision_cache WHERE group_id = ? AND image_key = ?",
                (gid, ik),
            )
            row = cur.fetchone()
            cur.close()
        except Exception:
            return None
        if not row:
            return None
        caption, status, last_seen_ts = row
        if str(status or "") != "ready":
            return None
        try:
            last_seen = float(last_seen_ts or 0.0)
        except Exception:
            last_seen = 0.0
        if last_seen <= 0 or now_ts - last_seen > ttl:
            return None
        c = str(caption or "").strip()
        return c or None

    def claim_caption_generation(
        self,
        *,
        group_id: str,
        image_key: str,
    ) -> bool:
        if not self._is_enabled():
            return False
        if not self._conn:
            self._init_conn()
        if not self._conn:
            return False
        gid = self._bucket_group_id(group_id)
        ik = str(image_key or "").strip()
        if not gid or not ik:
            return False
        now_ts = self._now()
        stale_ts = now_ts - 30.0
        try:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE image_vision_cache
                SET status = 'pending', last_seen_ts = ?
                WHERE group_id = ? AND image_key = ?
                  AND (caption IS NULL OR TRIM(caption) = '')
                  AND (
                        status IS NULL
                     OR status = ''
                     OR status = 'ready'
                     OR (status = 'pending' AND (last_seen_ts IS NULL OR last_seen_ts < ?))
                  )
                """,
                (now_ts, gid, ik, stale_ts),
            )
            changed = cur.rowcount > 0
            self._conn.commit()
            cur.close()
            return changed
        except Exception:
            return False

    def wait_caption_ready(
        self,
        *,
        group_id: str,
        image_key: str,
        timeout_sec: float = 1.2,
        interval_sec: float = 0.2,
    ) -> str | None:
        if not self._is_enabled():
            return None
        if not self._conn:
            self._init_conn()
        if not self._conn:
            return None
        gid = self._bucket_group_id(group_id)
        ik = str(image_key or "").strip()
        if not gid or not ik:
            return None
        deadline = self._now() + max(0.0, float(timeout_sec))
        while self._now() <= deadline:
            cap = self.get_caption(group_id=gid, image_key=ik)
            if cap:
                return cap
            time.sleep(max(0.05, float(interval_sec)))
        return None

    def record_seen(
        self,
        *,
        group_id: str,
        url: str | None,
        file_token: str | None,
        sender_id: str | None,
        msg_id: str | None,
        is_at: bool,
        mime_hint: str | None = None,
    ) -> tuple[str, str | None]:
        if not self._is_enabled() or is_at:
            return self.normalize_image_key(
                group_id=group_id, url=url, file_token=file_token
            )
        if not self._conn:
            self._init_conn()
        if not self._conn:
            return self.normalize_image_key(
                group_id=group_id, url=url, file_token=file_token
            )

        gid = self._bucket_group_id(group_id)
        ik, inferred_mime = self.normalize_image_key(
            group_id=gid, url=url, file_token=file_token
        )
        mt = mime_hint or inferred_mime
        now_ts = self._now()
        try:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO image_vision_cache(
                    group_id, image_key, orig_url, file_token, mime_hint,
                    first_seen_ts, last_seen_ts, seen_count, last_sender_id, last_msg_id
                ) VALUES(?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(group_id, image_key) DO UPDATE SET
                    last_seen_ts=excluded.last_seen_ts,
                    seen_count=seen_count+1,
                    last_sender_id=excluded.last_sender_id,
                    last_msg_id=excluded.last_msg_id,
                    orig_url=COALESCE(orig_url, excluded.orig_url),
                    file_token=COALESCE(file_token, excluded.file_token),
                    mime_hint=COALESCE(mime_hint, excluded.mime_hint)
                """,
                (
                    gid,
                    ik,
                    (url or "").strip() or None,
                    (file_token or "").strip() or None,
                    (mt or "").strip() or None,
                    now_ts,
                    now_ts,
                    (sender_id or "").strip() or None,
                    (msg_id or "").strip() or None,
                ),
            )
            self._conn.commit()
            cur.close()
        except Exception as e:
            logger.error(f"[mem_chat] image cache record_seen failed: {e}")

        self._prune_if_needed(gid)
        return ik, mt

    def set_caption(
        self,
        *,
        group_id: str,
        image_key: str,
        caption: str,
        model: str | None = None,
        status: str = "ready",
        err: str | None = None,
    ):
        if not self._is_enabled():
            return
        if not self._conn:
            self._init_conn()
        if not self._conn:
            return
        gid = self._bucket_group_id(group_id)
        ik = str(image_key or "").strip()
        cap = str(caption or "").strip()
        if not gid or not ik:
            return
        now_ts = self._now()
        try:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE image_vision_cache
                SET caption = ?, caption_model = ?, status = ?, err = ?, last_seen_ts = ?
                WHERE group_id = ? AND image_key = ?
                """,
                (
                    cap or None,
                    (model or "").strip() or None,
                    str(status or "ready"),
                    (err or "").strip() or None,
                    now_ts,
                    gid,
                    ik,
                ),
            )
            if cur.rowcount <= 0:
                cur.execute(
                    """
                    INSERT INTO image_vision_cache(
                        group_id, image_key, caption, caption_model, status, err,
                        first_seen_ts, last_seen_ts, seen_count
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        gid,
                        ik,
                        cap or None,
                        (model or "").strip() or None,
                        str(status or "ready"),
                        (err or "").strip() or None,
                        now_ts,
                        now_ts,
                    ),
                )
            self._conn.commit()
            cur.close()
        except Exception as e:
            logger.error(f"[mem_chat] image cache set_caption failed: {e}")

    def _prune_if_needed(self, group_id: str):
        if not self._conn:
            return
        gid = str(group_id or "").strip()
        if not gid:
            return
        max_records = self._max_records()
        ttl = self._ttl_sec()
        now_ts = self._now()
        try:
            cur = self._conn.cursor()
            cur.execute(
                "DELETE FROM image_vision_cache WHERE group_id = ? AND last_seen_ts < ?",
                (gid, now_ts - ttl),
            )
            cur.execute(
                "SELECT COUNT(1) FROM image_vision_cache WHERE group_id = ?",
                (gid,),
            )
            row = cur.fetchone()
            total = int(row[0] or 0) if row else 0
            if total > max_records:
                drop = total - max_records
                cur.execute(
                    """
                    DELETE FROM image_vision_cache
                    WHERE rowid IN (
                        SELECT rowid FROM image_vision_cache
                        WHERE group_id = ?
                        ORDER BY last_seen_ts ASC
                        LIMIT ?
                    )
                    """,
                    (gid, drop),
                )
            self._conn.commit()
            cur.close()
        except Exception:
            pass


__all__ = ["ImageVisionCache"]
