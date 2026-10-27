import asyncio
import collections
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

import aiohttp

try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    faiss = None
    np = None

logger = logging.getLogger("astrbot")


class DatabaseManager:
    def __init__(self, db_path: str, config: dict):
        self.db_path = db_path
        self.config = config
        self._db_conn = None
        self._db_write_lock = asyncio.Lock()  # Lock for SQLite writes
        self._recall_read_slots = asyncio.Semaphore(2)
        self._vector_dimension: int | None = None
        self._emb_cache = collections.OrderedDict()
        self._faiss_lock = asyncio.Lock()
        self._faiss_indexes: dict[str, object] = {}
        self._faiss_ready_buckets: set[str] = set()
        self._faiss_last_save_ts: dict[str, float] = {}
        self._emb_fail_until = 0.0
        self._emb_last_log_ts = 0.0
        self._emb_last_err = ""
        try:
            self._emb_cache_ttl = int(self.config.get("embedding_cache_ttl", 600))
        except Exception:
            self._emb_cache_ttl = 600
        try:
            self._emb_cache_size = int(self.config.get("embedding_cache_size", 1024))
        except Exception:
            self._emb_cache_size = 1024
        self.init_db_conn()

    def init_db_conn(self):
        if self._db_conn:
            return

        try:
            self._db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._db_conn.execute("PRAGMA journal_mode=WAL;")
            self._init_tables()
            self._maybe_migrate_faiss_ids()
        except Exception as e:
            logger.error(f"[mem_chat] Failed to connect to database: {e}")

    def _init_tables(self):
        cur = self._db_conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id TEXT PRIMARY KEY,
                content TEXT,
                embedding BLOB,
                vector_dimension INTEGER,
                faiss_id INTEGER,
                group_id TEXT,
                model_name TEXT,
                type TEXT DEFAULT 'memory',
                created_at REAL,
                last_updated REAL,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_access_at REAL
            )
            """
        )
        try:
            cur.execute("PRAGMA table_info(memory_embeddings)")
            cols = [r[1] for r in cur.fetchall()]
            if "model_name" not in cols:
                cur.execute("ALTER TABLE memory_embeddings ADD COLUMN model_name TEXT")
            if "type" not in cols:
                cur.execute(
                    "ALTER TABLE memory_embeddings ADD COLUMN type TEXT DEFAULT 'memory'"
                )
            if "faiss_id" not in cols:
                cur.execute("ALTER TABLE memory_embeddings ADD COLUMN faiss_id INTEGER")
            # Generative-Agents retrieval needs an importance weight plus access
            # bookkeeping. ADD COLUMN with a DEFAULT is a header-only change in
            # SQLite, so this stays cheap even on a multi-gigabyte table; rows
            # written before the migration simply read back as the default.
            if "importance" not in cols:
                cur.execute(
                    "ALTER TABLE memory_embeddings ADD COLUMN importance REAL DEFAULT 0.5"
                )
            if "access_count" not in cols:
                cur.execute(
                    "ALTER TABLE memory_embeddings ADD COLUMN access_count INTEGER DEFAULT 0"
                )
            if "last_access_at" not in cols:
                cur.execute(
                    "ALTER TABLE memory_embeddings ADD COLUMN last_access_at REAL"
                )
        except Exception:
            pass
        try:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_gmt ON memory_embeddings(group_id, model_name, type)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_mt ON memory_embeddings(model_name, type)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_g_created ON memory_embeddings(group_id, created_at)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_faiss ON memory_embeddings(faiss_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_type_created "
                "ON memory_embeddings(type, created_at DESC)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_gmt_created "
                "ON memory_embeddings(group_id, model_name, type, created_at DESC)"
            )
        except Exception:
            pass
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS impressions (
                name TEXT,
                group_id TEXT,
                score REAL,
                summary TEXT,
                user_id TEXT,
                msg_count INTEGER DEFAULT 0,
                PRIMARY KEY(name, group_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                group_id TEXT,
                user_id TEXT,
                display_name TEXT,
                profile_json TEXT,
                canonical_text TEXT,
                msg_count INTEGER DEFAULT 0,
                version INTEGER DEFAULT 2,
                created_at REAL,
                updated_at REAL,
                PRIMARY KEY(group_id, user_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS profile_facts (
                fact_id TEXT PRIMARY KEY,
                group_id TEXT,
                user_id TEXT,
                key TEXT,
                value TEXT,
                polarity INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.5,
                evidence_count INTEGER DEFAULT 1,
                first_seen REAL,
                last_seen REAL,
                updated_at REAL
            )
            """
        )
        try:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_profile_facts_gu ON profile_facts(group_id, user_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_profile_facts_last ON profile_facts(group_id, user_id, last_seen)"
            )
        except Exception:
            pass
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS impression_profiles_v3 (
                group_id TEXT,
                user_id TEXT,
                display_name TEXT,
                summary TEXT,
                tags_json TEXT,
                confidence REAL DEFAULT 0.5,
                source TEXT,
                created_at REAL,
                updated_at REAL,
                PRIMARY KEY(group_id, user_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS impression_facts_v3 (
                fact_id TEXT PRIMARY KEY,
                group_id TEXT,
                user_id TEXT,
                fact_text TEXT,
                reason TEXT,
                confidence REAL DEFAULT 0.5,
                evidence_count INTEGER DEFAULT 1,
                source TEXT,
                first_seen REAL,
                last_seen REAL,
                updated_at REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS impression_embeddings_v3 (
                embedding_id TEXT PRIMARY KEY,
                group_id TEXT,
                user_id TEXT,
                scope TEXT,
                ref_id TEXT,
                content TEXT,
                embedding BLOB,
                vector_dimension INTEGER,
                model_name TEXT,
                created_at REAL,
                last_updated REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_aliases (
                group_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                alias TEXT NOT NULL,
                source TEXT,
                first_seen REAL,
                last_seen REAL,
                evidence_count INTEGER DEFAULT 1,
                is_current INTEGER DEFAULT 0,
                updated_at REAL,
                PRIMARY KEY(group_id, user_id, alias)
            )
            """
        )
        try:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_impr_v3_profile_gu ON impression_profiles_v3(group_id, user_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_impr_v3_fact_gu ON impression_facts_v3(group_id, user_id, last_seen)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_impr_v3_emb_gmt ON impression_embeddings_v3(group_id, model_name, scope)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_impr_v3_emb_gu ON impression_embeddings_v3(group_id, user_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_aliases_gu_last ON user_aliases(group_id, user_id, last_seen DESC)"
            )
        except Exception:
            pass
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS affinity_state (
                group_id TEXT,
                user_id TEXT,
                affinity REAL DEFAULT 0,
                trust REAL DEFAULT 50,
                mood_bias TEXT DEFAULT 'neutral',
                last_event_ts REAL DEFAULT 0,
                updated_at REAL,
                PRIMARY KEY(group_id, user_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS affinity_events (
                event_id TEXT PRIMARY KEY,
                group_id TEXT,
                user_id TEXT,
                delta_affinity REAL,
                reason TEXT,
                source_msg_id TEXT,
                created_at REAL
            )
            """
        )
        try:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_aff_state_gu ON affinity_state(group_id, user_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_aff_state_updated ON affinity_state(group_id, updated_at)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_aff_event_gu ON affinity_events(group_id, user_id, created_at)"
            )
        except Exception:
            pass
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS system_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS image_library (
                image_id TEXT PRIMARY KEY,
                url TEXT,
                author TEXT,
                source TEXT,
                tags TEXT,
                score REAL DEFAULT 0,
                rating_count INTEGER DEFAULT 0,
                metadata TEXT,
                created_at REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_group_messages (
                group_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                message_seq TEXT,
                sender_id TEXT NOT NULL,
                sender_name TEXT,
                text TEXT NOT NULL,
                raw_text TEXT,
                image_urls_json TEXT,
                reply_to_msg_id TEXT,
                reply_to_sender_id TEXT,
                reply_to_sender_name TEXT,
                at_targets_json TEXT,
                at_all INTEGER DEFAULT 0,
                explicit_target_type TEXT,
                explicit_target_id TEXT,
                explicit_target_name TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY(group_id, message_id)
            )
            """
        )
        for alter_sql in (
            "ALTER TABLE raw_group_messages ADD COLUMN image_urls_json TEXT",
            "ALTER TABLE raw_group_messages ADD COLUMN reply_to_msg_id TEXT",
            "ALTER TABLE raw_group_messages ADD COLUMN reply_to_sender_id TEXT",
            "ALTER TABLE raw_group_messages ADD COLUMN reply_to_sender_name TEXT",
            "ALTER TABLE raw_group_messages ADD COLUMN at_targets_json TEXT",
            "ALTER TABLE raw_group_messages ADD COLUMN at_all INTEGER DEFAULT 0",
            "ALTER TABLE raw_group_messages ADD COLUMN explicit_target_type TEXT",
            "ALTER TABLE raw_group_messages ADD COLUMN explicit_target_id TEXT",
            "ALTER TABLE raw_group_messages ADD COLUMN explicit_target_name TEXT",
        ):
            try:
                cur.execute(alter_sql)
            except Exception:
                pass
        try:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_raw_msg_gs_ct ON raw_group_messages(group_id, sender_id, created_at DESC)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_raw_msg_gc ON raw_group_messages(group_id, created_at DESC)"
            )
        except Exception:
            pass
        cur.close()

    def set_meta_sync_safe(self, key: str, value: str):
        """Sync version of set_meta for use in __init__ or other sync contexts."""
        if not self._db_conn:
            self.init_db_conn()
        try:
            self._set_meta_sync(key, value)
        except Exception as e:
            logger.error(f"[mem_chat] Failed to set meta sync {key}: {e}")

    def _maybe_migrate_faiss_ids(self) -> None:
        if not self._db_conn:
            return
        try:
            # get_meta is now async? No, we need to check get_meta
            pass
        except Exception:
            pass

        # We need a sync get_meta too
        val = self._get_meta_sync("faiss_id_migrated_v1")
        if str(val or "") == "1":
            return

        cur = None
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT faiss_id, memory_id FROM memory_embeddings WHERE faiss_id IS NOT NULL"
            )
            used_by_id: dict[int, str] = {}
            dup_mids: list[str] = []
            for fid, mid in cur.fetchall():
                try:
                    fid_i = int(fid)
                except Exception:
                    continue
                mid_s = str(mid or "")
                if fid_i in used_by_id and used_by_id[fid_i] != mid_s:
                    dup_mids.append(mid_s)
                else:
                    used_by_id[fid_i] = mid_s

            cur.execute(
                "SELECT memory_id FROM memory_embeddings WHERE faiss_id IS NULL"
            )
            null_mids = [str(r[0] or "") for r in cur.fetchall() if r and r[0]]

            to_fix: list[str] = []
            seen_mid: set[str] = set()
            for mid in null_mids + dup_mids:
                if not mid or mid in seen_mid:
                    continue
                seen_mid.add(mid)
                to_fix.append(mid)

            if not to_fix:
                cur.close()
                return

            used_ids: set[int] = set(used_by_id.keys())
            updates: list[tuple[int, str]] = []
            for mid in to_fix:
                fid_i = self._allocate_faiss_id(mid, used_ids)
                updates.append((fid_i, mid))

            cur.executemany(
                "UPDATE memory_embeddings SET faiss_id = ? WHERE memory_id = ?",
                updates,
            )
            self._db_conn.commit()
            cur.close()
            try:
                self._set_meta_sync("faiss_id_migrated_v1", "1")
            except Exception:
                pass
        except Exception:
            try:
                if cur:
                    cur.close()
            except Exception:
                pass

    def _get_meta_sync(self, key: str) -> str | None:
        if not self._db_conn:
            return None
        try:
            cur = self._db_conn.cursor()
            cur.execute("SELECT value FROM system_meta WHERE key = ?", (key,))
            row = cur.fetchone()
            cur.close()
            return row[0] if row else None
        except Exception:
            return None

    async def get_meta(self, key: str) -> str | None:
        if not self._db_conn:
            self.init_db_conn()
        return await asyncio.to_thread(self._get_meta_sync, key)

    async def set_meta(self, key: str, value: str):
        if not self._db_conn:
            self.init_db_conn()
        try:
            async with self._db_write_lock:
                await asyncio.to_thread(self._set_meta_sync, key, value)
        except Exception as e:
            logger.error(f"[mem_chat] Failed to set meta {key}: {e}")

    def _set_meta_sync(self, key: str, value: str):
        cur = self._db_conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO system_meta(key, value) VALUES(?, ?)",
            (key, value),
        )
        self._db_conn.commit()
        cur.close()

    async def delete_meta(self, key: str):
        if not self._db_conn:
            self.init_db_conn()
        try:
            async with self._db_write_lock:
                await asyncio.to_thread(self._delete_meta_sync, key)
        except Exception as e:
            logger.error(f"[mem_chat] Failed to delete meta {key}: {e}")

    def _delete_meta_sync(self, key: str):
        cur = self._db_conn.cursor()
        cur.execute("DELETE FROM system_meta WHERE key = ?", (key,))
        self._db_conn.commit()
        cur.close()

    async def add_image_to_library(
        self,
        image_id: str,
        url: str,
        author: str = "",
        source: str = "",
        tags: str = "",
        metadata: dict | None = None,
    ):
        if not self._db_conn:
            self.init_db_conn()
        try:
            async with self._db_write_lock:
                await asyncio.to_thread(
                    self._add_image_sync, image_id, url, author, source, tags, metadata
                )
        except Exception as e:
            logger.error(f"[mem_chat] Failed to add image to library: {e}")

    def _add_image_sync(
        self,
        image_id: str,
        url: str,
        author: str,
        source: str,
        tags: str,
        metadata: dict | None,
    ):
        cur = self._db_conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO image_library(image_id, url, author, source, tags, metadata, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                url,
                author,
                source,
                tags,
                json.dumps(metadata or {}),
                time.time(),
            ),
        )
        self._db_conn.commit()
        cur.close()

    async def update_image_rating(self, image_id: str, rating: float):
        if not self._db_conn:
            self.init_db_conn()
        try:
            async with self._db_write_lock:
                await asyncio.to_thread(self._update_rating_sync, image_id, rating)
        except Exception as e:
            logger.error(f"[mem_chat] Failed to update image rating: {e}")

    def _update_rating_sync(self, image_id: str, rating: float):
        cur = self._db_conn.cursor()
        # 计算新分数：(旧分数 * 旧次数 + 新分数) / (旧次数 + 1)
        cur.execute(
            """
            UPDATE image_library
            SET score = (score * rating_count + ?) / (rating_count + 1),
                rating_count = rating_count + 1
            WHERE image_id = ?
            """,
            (rating, image_id),
        )
        self._db_conn.commit()
        cur.close()

    async def get_search_efficiency_report(self, days: int = 7) -> dict:
        if not self._db_conn:
            self.init_db_conn()
        try:
            return await asyncio.to_thread(self._get_report_sync, days)
        except Exception as e:
            logger.error(f"[mem_chat] Failed to get search efficiency report: {e}")
            return {}

    def _get_report_sync(self, days: int) -> dict:
        cur = self._db_conn.cursor()
        since = time.time() - (days * 86400)

        # Total sent
        cur.execute("SELECT COUNT(*) FROM image_library WHERE created_at > ?", (since,))
        total_sent = cur.fetchone()[0]

        # Total with feedback
        cur.execute(
            "SELECT COUNT(*) FROM image_library WHERE created_at > ? AND rating_count > 0",
            (since,),
        )
        total_feedback = cur.fetchone()[0]

        # Positive feedback (score >= 0.5)
        cur.execute(
            "SELECT COUNT(*) FROM image_library WHERE created_at > ? AND rating_count > 0 AND score >= 0.5",
            (since,),
        )
        positive_feedback = cur.fetchone()[0]

        # Source breakdown
        cur.execute(
            "SELECT source, COUNT(*) FROM image_library WHERE created_at > ? GROUP BY source",
            (since,),
        )
        sources = dict(cur.fetchall())

        cur.close()

        satisfaction_rate = (
            (positive_feedback / total_feedback) if total_feedback > 0 else 0.0
        )
        match_rate = (positive_feedback / total_sent) if total_sent > 0 else 0.0

        return {
            "days": days,
            "total_sent": total_sent,
            "total_feedback": total_feedback,
            "positive_feedback": positive_feedback,
            "satisfaction_rate": satisfaction_rate,
            "match_rate": match_rate,
            "sources": sources,
        }

    async def get_high_rated_images(self, limit: int = 10) -> list[dict]:
        if not self._db_conn:
            self.init_db_conn()
        try:
            return await asyncio.to_thread(self._get_high_rated_sync, limit)
        except Exception as e:
            logger.error(f"[mem_chat] Failed to get high rated images: {e}")
            return []

    def _get_high_rated_sync(self, limit: int) -> list[dict]:
        cur = self._db_conn.cursor()
        cur.execute(
            "SELECT * FROM image_library WHERE rating_count > 0 ORDER BY score DESC LIMIT ?",
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        cur.close()
        return [dict(zip(cols, r)) for r in rows]

    def close(self):
        if self._db_conn:
            try:
                self._db_conn.close()
            except Exception:
                pass
            self._db_conn = None

    def _faiss_enabled(self) -> bool:
        try:
            return bool(self.config.get("enable_faiss_recall", True))
        except Exception:
            return True

    def _faiss_index_dir(self) -> str:
        try:
            d = str(self.config.get("faiss_index_dir", "") or "").strip()
            if d:
                return d
        except Exception:
            pass
        return os.path.join(os.path.dirname(self.db_path), "faiss")

    def _faiss_autosave_interval_sec(self) -> float:
        try:
            v = float(self.config.get("faiss_autosave_interval_sec", 30) or 0)
            return 0.0 if v < 0 else v
        except Exception:
            return 30.0

    def _bucket_key(self, group_id: str, model_name: str, type: str) -> str:
        return f"{group_id}::{model_name}::{type}"

    def _bucket_index_path(self, bucket_key: str) -> str:
        h = hashlib.sha1(bucket_key.encode("utf-8")).hexdigest()
        return os.path.join(self._faiss_index_dir(), f"{h}.faiss")

    def _faiss_id_candidate(self, memory_id: str, attempt: int = 0) -> int:
        s = str(memory_id or "")
        if attempt:
            s = s + ":" + str(attempt)
        b = hashlib.sha1(s.encode("utf-8")).digest()[:8]
        return int.from_bytes(b, "big", signed=False) & ((1 << 63) - 1)

    def _to_faiss_id(self, memory_id: str) -> int:
        return self._faiss_id_candidate(memory_id, 0)

    def _allocate_faiss_id(self, memory_id: str, used_ids: set[int]) -> int:
        for attempt in range(0, 128):
            fid = self._faiss_id_candidate(memory_id, attempt)
            if fid not in used_ids:
                used_ids.add(fid)
                return fid
        fid = self._faiss_id_candidate(memory_id, 0)
        used_ids.add(fid)
        return fid

    def _alloc_faiss_id_db(self, memory_id: str) -> int:
        if not self._db_conn:
            return self._to_faiss_id(memory_id)
        cur = None
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT faiss_id FROM memory_embeddings WHERE memory_id = ? AND faiss_id IS NOT NULL",
                (memory_id,),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                try:
                    return int(row[0])
                except Exception:
                    pass
            for attempt in range(0, 128):
                fid = self._faiss_id_candidate(memory_id, attempt)
                cur.execute(
                    "SELECT memory_id FROM memory_embeddings WHERE faiss_id = ? LIMIT 1",
                    (fid,),
                )
                r2 = cur.fetchone()
                if not r2:
                    return fid
                if str(r2[0] or "") == str(memory_id or ""):
                    return fid
            return self._to_faiss_id(memory_id)
        except Exception:
            return self._to_faiss_id(memory_id)
        finally:
            try:
                if cur:
                    cur.close()
            except Exception:
                pass

    def _safe_path(self, path: str) -> str:
        try:
            if os.name == "nt" and path and not path.startswith("\\\\?\\"):
                return "\\\\?\\" + os.path.abspath(path)
        except Exception:
            pass
        return path

    async def _save_faiss_bucket(self, bucket_key: str) -> None:
        if not faiss or not np:
            return
        idx = self._faiss_indexes.get(bucket_key)
        if not idx:
            return
        path = self._bucket_index_path(bucket_key)
        try:
            await asyncio.to_thread(os.makedirs, os.path.dirname(path), exist_ok=True)
            await asyncio.to_thread(faiss.write_index, idx, self._safe_path(path))
            self._faiss_last_save_ts[bucket_key] = time.time()
        except Exception:
            return

    def _create_faiss_index(self, dim: int):
        base = faiss.IndexFlatIP(dim)
        return faiss.IndexIDMap2(base)

    async def _ensure_faiss_bucket_ready(
        self,
        group_id: str,
        model_name: str,
        type: str,
        dim: int,
        allow_rebuild: bool = True,
    ) -> str | None:
        if not self._faiss_enabled() or not faiss or not np:
            return None
        bucket_key = self._bucket_key(group_id, model_name, type)
        async with self._faiss_lock:
            if bucket_key in self._faiss_ready_buckets:
                return bucket_key
            idx = None
            path = self._bucket_index_path(bucket_key)
            if os.path.exists(path):
                try:
                    idx = await asyncio.to_thread(
                        faiss.read_index, self._safe_path(path)
                    )
                except Exception:
                    idx = None
            if idx is None:
                if not allow_rebuild:
                    idx = self._create_faiss_index(dim)
                else:
                    idx = await self._rebuild_faiss_bucket_from_db(
                        group_id, model_name, type, dim
                    )
            try:
                if int(getattr(idx, "d", dim)) != int(dim):
                    idx = await self._rebuild_faiss_bucket_from_db(
                        group_id, model_name, type, dim
                    )
            except Exception:
                pass
            if allow_rebuild and self._db_conn:
                cur = None
                try:
                    cur = self._db_conn.cursor()
                    cur.execute(
                        "SELECT COUNT(*) FROM memory_embeddings WHERE group_id = ? AND model_name = ? AND type = ? AND vector_dimension > 0 AND embedding IS NOT NULL AND length(embedding) > 0",
                        (group_id, model_name, type),
                    )
                    expected = int((cur.fetchone() or [0])[0] or 0)
                    cur.execute(
                        "SELECT COUNT(*) FROM memory_embeddings WHERE group_id = ? AND model_name = ? AND type = ? AND faiss_id IS NULL",
                        (group_id, model_name, type),
                    )
                    missing_ids = int((cur.fetchone() or [0])[0] or 0)
                    cur.close()
                    ntotal = int(getattr(idx, "ntotal", 0) or 0)
                    if missing_ids > 0 or (expected > 0 and ntotal != expected):
                        idx = await self._rebuild_faiss_bucket_from_db(
                            group_id, model_name, type, dim
                        )
                except Exception:
                    try:
                        if cur:
                            cur.close()
                    except Exception:
                        pass
            self._faiss_indexes[bucket_key] = idx
            self._faiss_ready_buckets.add(bucket_key)
            return bucket_key

    async def _rebuild_faiss_bucket_from_db(
        self, group_id: str, model_name: str, type: str, dim: int
    ):
        idx = self._create_faiss_index(dim)
        if not self._db_conn:
            self.init_db_conn()
        if not self._db_conn:
            return idx
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT memory_id, faiss_id, embedding, vector_dimension FROM memory_embeddings WHERE group_id = ? AND model_name = ? AND type = ? AND vector_dimension > 0 AND embedding IS NOT NULL AND length(embedding) > 0",
                (group_id, model_name, type),
            )
            rows = cur.fetchall()
            cur.close()
        except Exception:
            try:
                cur.close()
            except Exception:
                pass
            return idx
        ids: list[int] = []
        vecs: list[list[float]] = []
        updates: list[tuple[int, str]] = []
        used_ids: set[int] = set()
        for mid, fid, blob, vd in rows:
            emb = self._deserialize_embedding(blob, int(vd or 0))
            if not emb:
                continue
            faiss_id = None
            try:
                faiss_id = int(fid) if fid is not None else None
            except Exception:
                faiss_id = None
            if faiss_id is None or faiss_id in used_ids:
                faiss_id = self._allocate_faiss_id(str(mid), used_ids)
                updates.append((faiss_id, str(mid)))
            else:
                used_ids.add(faiss_id)
            if len(emb) != dim:
                continue
            ids.append(faiss_id)
            vecs.append([float(x) for x in emb])
        if updates:
            try:
                cur2 = self._db_conn.cursor()
                cur2.executemany(
                    "UPDATE memory_embeddings SET faiss_id = ? WHERE memory_id = ?",
                    updates,
                )
                self._db_conn.commit()
                cur2.close()
            except Exception:
                try:
                    cur2.close()
                except Exception:
                    pass
        if ids and vecs:
            try:
                arr = np.asarray(vecs, dtype=np.float32)
                faiss.normalize_L2(arr)
                idx.add_with_ids(arr, np.asarray(ids, dtype=np.int64))
            except Exception:
                pass
        try:
            bucket_key = self._bucket_key(group_id, model_name, type)
            self._faiss_indexes[bucket_key] = idx
            await self._save_faiss_bucket(bucket_key)
        except Exception:
            pass
        return idx

    async def _faiss_maybe_autosave(self, bucket_key: str) -> None:
        interval = self._faiss_autosave_interval_sec()
        if interval <= 0:
            return
        last = float(self._faiss_last_save_ts.get(bucket_key, 0.0) or 0.0)
        if time.time() - last < interval:
            return
        await self._save_faiss_bucket(bucket_key)

    async def _faiss_upsert_if_ready(
        self,
        group_id: str,
        model_name: str,
        type: str,
        faiss_id: int,
        emb: list[float],
    ) -> None:
        if not self._faiss_enabled() or not faiss or not np:
            return
        bucket_key = self._bucket_key(group_id, model_name, type)
        if bucket_key not in self._faiss_ready_buckets:
            return
        if not emb:
            return
        try:
            vec = np.asarray([emb], dtype=np.float32)
            faiss.normalize_L2(vec)
        except Exception:
            return
        async with self._faiss_lock:
            idx = self._faiss_indexes.get(bucket_key)
            if not idx:
                return
            try:
                idx.remove_ids(np.asarray([faiss_id], dtype=np.int64))
            except Exception:
                pass
            try:
                idx.add_with_ids(vec, np.asarray([faiss_id], dtype=np.int64))
            except Exception:
                return
        await self._faiss_maybe_autosave(bucket_key)

    async def _faiss_delete_if_ready(
        self,
        group_id: str,
        model_name: str,
        type: str,
        faiss_ids: list[int],
    ) -> None:
        if not self._faiss_enabled() or not faiss or not np:
            return
        bucket_key = self._bucket_key(group_id, model_name, type)
        if bucket_key not in self._faiss_ready_buckets:
            return
        if not faiss_ids:
            return
        async with self._faiss_lock:
            idx = self._faiss_indexes.get(bucket_key)
            if not idx:
                return
            try:
                idx.remove_ids(np.asarray(faiss_ids, dtype=np.int64))
            except Exception:
                return
        await self._faiss_maybe_autosave(bucket_key)

    def hash_id(self, text: str, group_id: str, model_name: str) -> str:
        h = hashlib.sha1()
        h.update(
            (group_id + "::" + (model_name or "") + "::" + (text or "")).encode("utf-8")
        )
        return h.hexdigest()

    def _normalize_user_alias(self, alias: str) -> str:
        s = str(alias or "").replace("\r", " ").replace("\n", " ").strip()
        s = re.sub(r"\s+", " ", s)
        if not s:
            return ""
        if s.lower() in {"user", "unknown", "system"} or s in {"未知", "匿名"}:
            return ""
        return s[:64]

    def _upsert_user_alias_with_cursor(
        self,
        cur,
        group_id: str,
        user_id: str,
        alias: str,
        *,
        source: str = "",
        seen_ts: float | None = None,
        is_current: bool = True,
    ) -> bool:
        gid = str(group_id or "").strip()
        uid = str(user_id or "").strip()
        name = self._normalize_user_alias(alias)
        if not gid or not uid or not name:
            return False
        now = time.time()
        seen = float(seen_ts or now)
        if is_current:
            cur.execute(
                "UPDATE user_aliases SET is_current = 0 WHERE group_id = ? AND user_id = ?",
                (gid, uid),
            )
        cur.execute(
            """
            INSERT INTO user_aliases(
                group_id, user_id, alias, source, first_seen, last_seen,
                evidence_count, is_current, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(group_id, user_id, alias) DO UPDATE SET
                source=excluded.source,
                last_seen=excluded.last_seen,
                evidence_count=COALESCE(user_aliases.evidence_count, 0) + 1,
                is_current=CASE
                    WHEN excluded.is_current = 1 THEN 1
                    ELSE user_aliases.is_current
                END,
                updated_at=excluded.updated_at
            """,
            (
                gid,
                uid,
                name,
                str(source or "")[:40],
                seen,
                seen,
                1,
                1 if is_current else 0,
                now,
            ),
        )
        return True

    def upsert_user_alias(
        self,
        group_id: str,
        user_id: str,
        alias: str,
        *,
        source: str = "",
        seen_ts: float | None = None,
        is_current: bool = True,
    ) -> bool:
        if not self._db_conn:
            self.init_db_conn()
        try:
            cur = self._db_conn.cursor()
            ok = self._upsert_user_alias_with_cursor(
                cur,
                group_id,
                user_id,
                alias,
                source=source,
                seen_ts=seen_ts,
                is_current=is_current,
            )
            if ok:
                self._db_conn.commit()
            cur.close()
            return ok
        except Exception as e:
            logger.error(f"[mem_chat] Failed to upsert user alias: {e}")
            return False

    def list_user_aliases(
        self, group_id: str, user_id: str, limit: int = 8
    ) -> list[tuple[str, bool, int, float, float]]:
        if not self._db_conn:
            self.init_db_conn()
        gid = str(group_id or "").strip()
        uid = str(user_id or "").strip()
        if not gid or not uid:
            return []
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                """
                SELECT alias, is_current, evidence_count, first_seen, last_seen
                FROM user_aliases
                WHERE group_id = ? AND user_id = ?
                ORDER BY is_current DESC, last_seen DESC, evidence_count DESC
                LIMIT ?
                """,
                (gid, uid, max(1, int(limit or 8))),
            )
            rows = cur.fetchall()
            cur.close()
            out: list[tuple[str, bool, int, float, float]] = []
            for alias, is_current, evidence_count, first_seen, last_seen in rows or []:
                name = self._normalize_user_alias(alias)
                if not name:
                    continue
                out.append(
                    (
                        name,
                        bool(is_current),
                        int(evidence_count or 0),
                        float(first_seen or 0.0),
                        float(last_seen or 0.0),
                    )
                )
            return out
        except Exception as e:
            logger.error(f"[mem_chat] Failed to list user aliases: {e}")
            return []

    def _serialize_embedding(self, emb: list[float]) -> bytes:
        return json.dumps(emb).encode("utf-8")

    def _deserialize_embedding(self, data: bytes, dim: int) -> list[float] | None:
        try:
            arr = json.loads(data.decode("utf-8"))
            if isinstance(arr, list) and (dim is None or len(arr) == dim):
                return [float(x) for x in arr]
        except Exception:
            return None
        return None

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def get_embedding(self, text: str) -> list[float]:
        base = str(self.config.get("embedding_openai_base_url", "") or "").rstrip("/")
        key = str(self.config.get("embedding_openai_api_key", "") or "")
        model = str(
            self.config.get("embedding_openai_model", "text-embedding-3-small")
            or "text-embedding-3-small"
        )
        now = time.time()
        k = model + "::" + (text or "").strip()
        try:
            v = self._emb_cache.get(k)
            if v and (now - v[0] <= self._emb_cache_ttl):
                emb_cached = v[1]
                if isinstance(emb_cached, list) and emb_cached:
                    self._emb_cache.move_to_end(k)
                    return [float(x) for x in emb_cached]
        except Exception:
            pass
        if now < float(self._emb_fail_until or 0.0):
            return []
        if base and key:
            if "/embeddings" in base:
                url = base
            elif base.endswith("/v1"):
                url = base + "/embeddings"
            else:
                url = base + "/v1/embeddings"

            payload = {"model": model, "input": text}
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }

            # 增加重试逻辑处理频率限制
            for attempt in range(3):
                try:
                    timeout = aiohttp.ClientTimeout(total=60)
                    async with aiohttp.ClientSession(
                        timeout=timeout, trust_env=True
                    ) as session:
                        async with session.post(
                            url, headers=headers, json=payload
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                arr = data.get("data") or []
                                if arr:
                                    emb = arr[0].get("embedding") or []
                                    if isinstance(emb, list) and emb:
                                        logger.info(
                                            f"[mem_chat] Successfully fetched embedding from {url} (dim: {len(emb)})"
                                        )
                                        if self._vector_dimension is None:
                                            self._vector_dimension = len(emb)
                                        try:
                                            self._emb_cache[k] = (now, emb)
                                            self._emb_cache.move_to_end(k)
                                            if (
                                                len(self._emb_cache)
                                                > self._emb_cache_size
                                            ):
                                                self._emb_cache.popitem(last=False)
                                        except Exception:
                                            pass
                                        return [float(x) for x in emb]
                            elif resp.status in [403, 429]:
                                body = await resp.text()
                                if "limit" in body.lower() or "rate" in body.lower():
                                    wait_time = 2 * (attempt + 1)
                                    logger.warning(
                                        f"[mem_chat] Rate limit hit (403/429), retrying in {wait_time}s... ({attempt + 1}/3)"
                                    )
                                    await asyncio.sleep(wait_time)
                                    continue

                            # 其他错误
                            try:
                                body = await resp.text()
                            except Exception:
                                body = ""
                            msg = f"HTTP {resp.status} {url} {body[:200]}".strip()
                            self._emb_fail_until = now + 300.0
                            if (
                                msg != self._emb_last_err
                                or now - float(self._emb_last_log_ts or 0.0) > 60.0
                            ):
                                self._emb_last_err = msg
                                self._emb_last_log_ts = now
                                logger.warning(
                                    f"[mem_chat] Embedding request failed: {msg}"
                                )
                            return []
                except Exception as e:
                    if attempt < 2:
                        await asyncio.sleep(1)
                        continue
                    msg = f"{type(e).__name__}: {e}"
                    self._emb_fail_until = now + 300.0
                    if (
                        msg != self._emb_last_err
                        or now - float(self._emb_last_log_ts or 0.0) > 60.0
                    ):
                        self._emb_last_err = msg
                        self._emb_last_log_ts = now
                        logger.warning(f"[mem_chat] Error getting embedding: {msg}")
                    return []
            return []
        return []

    async def get_or_compute_embedding(
        self,
        memory_id: str,
        content: str,
        group_id: str,
        model_name: str,
        type: str = "memory",
        importance: float | None = None,
    ) -> list[float]:
        """Return the embedding for a memory, computing and storing it if new.

        Args:
            memory_id: Stable id of the memory row.
            content: Text to embed.
            group_id: Owning conversation/group bucket.
            model_name: Embedding model identifier.
            type: Memory kind (memory/persona/impression/knowledge/reflection).
            importance: Optional 0..1 salience weight used by retrieval. When
                omitted, an existing stored weight is kept so a rewrite does
                not silently reset it to the column default.

        Returns:
            The embedding vector, or an empty list when embedding failed.
        """
        if not self._db_conn:
            self.init_db_conn()
        cur = self._db_conn.cursor()
        prev_importance = None
        try:
            cur.execute(
                "SELECT content, embedding, vector_dimension, importance FROM memory_embeddings WHERE memory_id = ? AND group_id = ? AND model_name = ?",
                (memory_id, group_id, model_name),
            )
            row = cur.fetchone()
            if row:
                try:
                    prev_importance = float(row[3]) if row[3] is not None else None
                except (TypeError, ValueError):
                    prev_importance = None
                existing_content = str(row[0] or "")
                if existing_content == str(content or "") and row[1]:
                    emb = self._deserialize_embedding(row[1], int(row[2] or 0))
                    cur.close()
                    if emb:
                        if importance is not None:
                            score = max(0.0, min(1.0, float(importance)))
                            self._db_conn.execute(
                                "UPDATE memory_embeddings SET importance = ? WHERE memory_id = ? AND group_id = ? AND model_name = ?",
                                (score, memory_id, group_id, model_name),
                            )
                            self._db_conn.commit()
                        return emb
        except Exception:
            cur.close()
            pass

        cur.close()

        # INSERT OR REPLACE rewrites the whole row, so an unspecified
        # importance would fall back to the column default and discard the
        # score this memory already earned.
        if importance is None:
            importance = (
                prev_importance
                if prev_importance is not None
                else self.config.get("ga_importance_default", 0.5)
            )
        try:
            importance = max(0.0, min(1.0, float(importance)))
        except (TypeError, ValueError):
            importance = 0.5

        emb = await self.get_embedding(content)
        if emb:
            try:
                if not self._db_conn:
                    self.init_db_conn()
                cur = self._db_conn.cursor()
                now = time.time()
                faiss_id = self._alloc_faiss_id_db(memory_id)
                cur.execute(
                    "INSERT OR REPLACE INTO memory_embeddings(memory_id, content, embedding, vector_dimension, faiss_id, group_id, model_name, type, created_at, last_updated, importance) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        memory_id,
                        content,
                        self._serialize_embedding(emb),
                        len(emb),
                        faiss_id,
                        group_id,
                        model_name,
                        type,
                        now,
                        now,
                        importance,
                    ),
                )
                self._db_conn.commit()
            except Exception:
                pass
            finally:
                try:
                    cur.close()
                except Exception:
                    pass
            try:
                await self._faiss_upsert_if_ready(
                    group_id, model_name, type, faiss_id, emb
                )
            except Exception:
                pass
        return emb or []

    def upsert_user_profile(
        self,
        group_id: str,
        user_id: str,
        display_name: str,
        profile_json: str,
        canonical_text: str,
        msg_count: int | None = None,
        version: int = 2,
    ) -> None:
        if not self._db_conn:
            self.init_db_conn()
        now = time.time()
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT msg_count, created_at FROM user_profiles WHERE group_id = ? AND user_id = ?",
                (group_id, user_id),
            )
            row = cur.fetchone()
            prev_count = int(row[0] or 0) if row else 0
            created_at = float(row[1] or now) if row else now
            final_count = prev_count
            if msg_count is not None:
                final_count = max(0, int(msg_count))
            self._upsert_user_alias_with_cursor(
                cur,
                group_id,
                user_id,
                display_name or "User",
                source="profile",
                seen_ts=now,
                is_current=True,
            )
            cur.execute(
                "INSERT OR REPLACE INTO user_profiles(group_id, user_id, display_name, profile_json, canonical_text, msg_count, version, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    group_id,
                    user_id,
                    display_name or "User",
                    profile_json or "",
                    canonical_text or "",
                    final_count,
                    int(version),
                    created_at,
                    now,
                ),
            )
            self._db_conn.commit()
            cur.close()
        except Exception as e:
            logger.error(f"[mem_chat] Failed to upsert user profile: {e}")

    def increment_user_profile_msg_count(
        self, group_id: str, user_id: str, delta: int
    ) -> None:
        if not self._db_conn:
            self.init_db_conn()
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "UPDATE user_profiles SET msg_count = COALESCE(msg_count, 0) + ?, updated_at = ? WHERE group_id = ? AND user_id = ?",
                (int(delta), time.time(), group_id, user_id),
            )
            self._db_conn.commit()
            cur.close()
        except Exception:
            pass

    def get_user_profile(
        self, group_id: str, user_id: str
    ) -> tuple[str, str, str] | None:
        if not self._db_conn:
            self.init_db_conn()
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT display_name, profile_json, canonical_text FROM user_profiles WHERE group_id = ? AND user_id = ?",
                (group_id, user_id),
            )
            row = cur.fetchone()
            cur.close()
            if not row:
                return None
            return (str(row[0] or "User"), str(row[1] or ""), str(row[2] or ""))
        except Exception:
            return None

    def list_user_profiles(self, group_id: str) -> list[tuple[str, str, str, str]]:
        if not self._db_conn:
            self.init_db_conn()
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT user_id, display_name, profile_json, canonical_text FROM user_profiles WHERE group_id = ?",
                (group_id,),
            )
            rows = cur.fetchall()
            cur.close()
            out: list[tuple[str, str, str, str]] = []
            for r in rows:
                try:
                    out.append(
                        (
                            str(r[0] or ""),
                            str(r[1] or "User"),
                            str(r[2] or ""),
                            str(r[3] or ""),
                        )
                    )
                except Exception:
                    continue
            return out
        except Exception:
            return []

    def _clamp_affinity(self, v: float) -> float:
        if v < -100.0:
            return -100.0
        if v > 100.0:
            return 100.0
        return v

    def _clamp_trust(self, v: float) -> float:
        if v < 0.0:
            return 0.0
        if v > 100.0:
            return 100.0
        return v

    def ensure_affinity_state(
        self,
        group_id: str,
        user_id: str,
        default_affinity: float = 0.0,
        default_trust: float = 50.0,
    ) -> None:
        if not self._db_conn:
            self.init_db_conn()
        gid = str(group_id or "").strip()
        uid = str(user_id or "").strip()
        if not gid or not uid:
            return
        now = time.time()
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO affinity_state(group_id, user_id, affinity, trust, mood_bias, last_event_ts, updated_at) VALUES(?,?,?,?,?,?,?)",
                (
                    gid,
                    uid,
                    self._clamp_affinity(float(default_affinity)),
                    self._clamp_trust(float(default_trust)),
                    "neutral",
                    0.0,
                    now,
                ),
            )
            self._db_conn.commit()
            cur.close()
        except Exception:
            pass

    def get_affinity_state(
        self, group_id: str, user_id: str
    ) -> tuple[float, float, str, float, float] | None:
        if not self._db_conn:
            self.init_db_conn()
        gid = str(group_id or "").strip()
        uid = str(user_id or "").strip()
        if not gid or not uid:
            return None
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT affinity, trust, mood_bias, last_event_ts, updated_at FROM affinity_state WHERE group_id = ? AND user_id = ?",
                (gid, uid),
            )
            row = cur.fetchone()
            cur.close()
            if not row:
                return None
            aff = self._clamp_affinity(float(row[0] or 0.0))
            trust = self._clamp_trust(float(row[1] or 50.0))
            mood_bias = str(row[2] or "neutral")
            last_event_ts = float(row[3] or 0.0)
            updated_at = float(row[4] or 0.0)
            return (aff, trust, mood_bias, last_event_ts, updated_at)
        except Exception:
            return None

    def set_affinity_state(
        self,
        group_id: str,
        user_id: str,
        affinity: float | None = None,
        trust: float | None = None,
        mood_bias: str | None = None,
        last_event_ts: float | None = None,
    ) -> None:
        if not self._db_conn:
            self.init_db_conn()
        gid = str(group_id or "").strip()
        uid = str(user_id or "").strip()
        if not gid or not uid:
            return
        now = time.time()
        try:
            curr = self.get_affinity_state(gid, uid)
            if curr is None:
                self.ensure_affinity_state(gid, uid)
                curr = self.get_affinity_state(gid, uid)
            if curr is None:
                return
            curr_aff, curr_trust, curr_mood, curr_last_evt, _curr_updated = curr
            next_aff = self._clamp_affinity(
                float(curr_aff if affinity is None else affinity)
            )
            next_trust = self._clamp_trust(
                float(curr_trust if trust is None else trust)
            )
            next_mood = (
                str(curr_mood if mood_bias is None else mood_bias).strip() or "neutral"
            )
            next_last_evt = float(
                curr_last_evt if last_event_ts is None else last_event_ts
            )
            cur = self._db_conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO affinity_state(group_id, user_id, affinity, trust, mood_bias, last_event_ts, updated_at) VALUES(?,?,?,?,?,?,?)",
                (gid, uid, next_aff, next_trust, next_mood, next_last_evt, now),
            )
            self._db_conn.commit()
            cur.close()
        except Exception:
            pass

    def get_affinity_event_count(self, group_id: str, user_id: str) -> int:
        if not self._db_conn:
            self.init_db_conn()
        gid = str(group_id or "").strip()
        uid = str(user_id or "").strip()
        if not gid or not uid:
            return 0
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM affinity_events WHERE group_id = ? AND user_id = ?",
                (gid, uid),
            )
            row = cur.fetchone()
            cur.close()
            return int((row[0] if row else 0) or 0)
        except Exception:
            return 0

    def apply_affinity_delta(
        self,
        group_id: str,
        user_id: str,
        delta_affinity: float,
        reason: str = "",
        source_msg_id: str = "",
        event_id: str = "",
        cooldown_sec: float = 120.0,
    ) -> tuple[float, float] | None:
        if not self._db_conn:
            self.init_db_conn()
        gid = str(group_id or "").strip()
        uid = str(user_id or "").strip()
        if not gid or not uid:
            return None
        now = time.time()
        try:
            self.ensure_affinity_state(gid, uid)
            curr = self.get_affinity_state(gid, uid)
            if curr is None:
                return None
            curr_aff, curr_trust, curr_mood, curr_last_evt, _curr_updated = curr
            cd = max(0.0, float(cooldown_sec or 0.0))
            if cd > 0 and curr_last_evt > 0 and now - curr_last_evt < cd:
                return (curr_aff, curr_trust)
            delta = float(delta_affinity or 0.0)
            # Stage-based volatility:
            # - first 10 effective updates: fast swings (up to ~40)
            # - next stage: medium swings (up to ~10)
            # - mature stage: slow swings (up to ~5)
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM affinity_events WHERE group_id = ? AND user_id = ?",
                (gid, uid),
            )
            row = cur.fetchone()
            event_count = int((row[0] if row else 0) or 0)
            if event_count < 10:
                max_abs = 40.0
            elif event_count < 50:
                max_abs = 10.0
            else:
                max_abs = 5.0
            if delta > max_abs:
                delta = max_abs
            elif delta < -max_abs:
                delta = -max_abs
            next_aff = self._clamp_affinity(curr_aff + delta)
            cur.execute(
                "INSERT OR REPLACE INTO affinity_state(group_id, user_id, affinity, trust, mood_bias, last_event_ts, updated_at) VALUES(?,?,?,?,?,?,?)",
                (gid, uid, next_aff, curr_trust, curr_mood, now, now),
            )
            eid = str(event_id or "").strip()
            if not eid:
                eid = self.hash_id(
                    f"{gid}|{uid}|{delta}|{reason}|{source_msg_id}|{now}",
                    "affinity",
                    "event",
                )
            cur.execute(
                "INSERT OR REPLACE INTO affinity_events(event_id, group_id, user_id, delta_affinity, reason, source_msg_id, created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    eid,
                    gid,
                    uid,
                    delta,
                    str(reason or "")[:160],
                    str(source_msg_id or "")[:64],
                    now,
                ),
            )
            self._db_conn.commit()
            cur.close()
            return (next_aff, curr_trust)
        except Exception:
            return None

    def decay_affinity_towards_neutral(
        self, group_id: str, step: float = 1.0, idle_sec: float = 3600.0
    ) -> int:
        if not self._db_conn:
            self.init_db_conn()
        gid = str(group_id or "").strip()
        if not gid:
            return 0
        now = time.time()
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT user_id, affinity, last_event_ts FROM affinity_state WHERE group_id = ?",
                (gid,),
            )
            rows = cur.fetchall()
            changed = 0
            for uid, aff, last_evt in rows or []:
                try:
                    aff_f = float(aff or 0.0)
                    last_evt_f = float(last_evt or 0.0)
                except Exception:
                    continue
                if abs(aff_f) < 0.01:
                    continue
                if (
                    idle_sec > 0
                    and last_evt_f > 0
                    and now - last_evt_f < float(idle_sec)
                ):
                    continue
                if aff_f > 0:
                    new_aff = max(0.0, aff_f - abs(float(step or 1.0)))
                else:
                    new_aff = min(0.0, aff_f + abs(float(step or 1.0)))
                if abs(new_aff - aff_f) < 0.001:
                    continue
                cur.execute(
                    "UPDATE affinity_state SET affinity = ?, updated_at = ? WHERE group_id = ? AND user_id = ?",
                    (self._clamp_affinity(new_aff), now, gid, str(uid or "")),
                )
                changed += 1
            self._db_conn.commit()
            cur.close()
            return changed
        except Exception:
            return 0

    def upsert_profile_fact(
        self,
        group_id: str,
        user_id: str,
        key: str,
        value: str,
        polarity: int = 1,
        confidence: float = 0.5,
        evidence_count: int = 1,
        seen_ts: float | None = None,
    ) -> None:
        if not self._db_conn:
            self.init_db_conn()
        now = float(seen_ts or time.time())
        fact_id = self.hash_id(
            f"{group_id}::{user_id}::{key}::{value}::{polarity}", "global", "fact_v2"
        )
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT evidence_count, first_seen FROM profile_facts WHERE fact_id = ?",
                (fact_id,),
            )
            row = cur.fetchone()
            prev_ev = int(row[0] or 0) if row else 0
            first_seen = float(row[1] or now) if row else now
            cur.execute(
                "INSERT OR REPLACE INTO profile_facts(fact_id, group_id, user_id, key, value, polarity, confidence, evidence_count, first_seen, last_seen, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    fact_id,
                    group_id,
                    user_id,
                    key or "",
                    value or "",
                    int(polarity),
                    float(confidence),
                    max(1, prev_ev + int(evidence_count)),
                    first_seen,
                    now,
                    time.time(),
                ),
            )
            self._db_conn.commit()
            cur.close()
        except Exception:
            pass

    def list_profile_facts(
        self, group_id: str, user_id: str, limit: int = 50
    ) -> list[tuple[str, str, int, float, int, float]]:
        if not self._db_conn:
            self.init_db_conn()
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT key, value, polarity, confidence, evidence_count, last_seen FROM profile_facts WHERE group_id = ? AND user_id = ? ORDER BY last_seen DESC LIMIT ?",
                (group_id, user_id, int(limit)),
            )
            rows = cur.fetchall()
            cur.close()
            out: list[tuple[str, str, int, float, int, float]] = []
            for r in rows:
                try:
                    out.append(
                        (
                            str(r[0] or ""),
                            str(r[1] or ""),
                            int(r[2] or 1),
                            float(r[3] or 0.5),
                            int(r[4] or 1),
                            float(r[5] or 0.0),
                        )
                    )
                except Exception:
                    continue
            return out
        except Exception:
            return []

    def upsert_impression_profile_v3(
        self,
        group_id: str,
        user_id: str,
        display_name: str,
        summary: str,
        tags: list[str] | None = None,
        confidence: float = 0.5,
        source: str = "",
    ) -> None:
        if not self._db_conn:
            self.init_db_conn()
        gid = str(group_id or "").strip()
        uid = str(user_id or "").strip()
        if not gid or not uid:
            return
        now = time.time()
        summ = str(summary or "").strip()[:240]
        tags_list: list[str] = []
        for t in tags or []:
            s = str(t or "").strip()
            if not s:
                continue
            if s not in tags_list:
                tags_list.append(s[:24])
            if len(tags_list) >= 12:
                break
        tags_json = json.dumps(tags_list, ensure_ascii=False)
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT created_at FROM impression_profiles_v3 WHERE group_id = ? AND user_id = ?",
                (gid, uid),
            )
            row = cur.fetchone()
            created_at = float(row[0] or now) if row else now
            self._upsert_user_alias_with_cursor(
                cur,
                gid,
                uid,
                str(display_name or "User"),
                source=str(source or "impression_v3")[:40],
                seen_ts=now,
                is_current=True,
            )
            cur.execute(
                "INSERT OR REPLACE INTO impression_profiles_v3(group_id, user_id, display_name, summary, tags_json, confidence, source, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    gid,
                    uid,
                    str(display_name or "User")[:64],
                    summ,
                    tags_json,
                    max(0.0, min(1.0, float(confidence))),
                    str(source or "")[:40],
                    created_at,
                    now,
                ),
            )
            self._db_conn.commit()
            cur.close()
        except Exception:
            pass

    def get_impression_profile_v3(
        self, group_id: str, user_id: str
    ) -> tuple[str, str, list[str], float, float] | None:
        if not self._db_conn:
            self.init_db_conn()
        gid = str(group_id or "").strip()
        uid = str(user_id or "").strip()
        if not gid or not uid:
            return None
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT display_name, summary, tags_json, confidence, updated_at FROM impression_profiles_v3 WHERE group_id = ? AND user_id = ?",
                (gid, uid),
            )
            row = cur.fetchone()
            cur.close()
            if not row:
                return None
            tags_raw = str(row[2] or "").strip()
            tags: list[str] = []
            if tags_raw:
                try:
                    arr = json.loads(tags_raw)
                    if isinstance(arr, list):
                        tags = [str(x).strip() for x in arr if str(x).strip()]
                except Exception:
                    tags = []
            return (
                str(row[0] or "User"),
                str(row[1] or ""),
                tags[:12],
                float(row[3] or 0.5),
                float(row[4] or 0.0),
            )
        except Exception:
            return None

    def upsert_impression_fact_v3(
        self,
        group_id: str,
        user_id: str,
        fact_text: str,
        reason: str = "",
        confidence: float = 0.5,
        source: str = "",
        seen_ts: float | None = None,
    ) -> str:
        if not self._db_conn:
            self.init_db_conn()
        gid = str(group_id or "").strip()
        uid = str(user_id or "").strip()
        fact = str(fact_text or "").strip()
        if not gid or not uid or not fact:
            return ""
        if len(fact) > 120:
            fact = fact[:120].rstrip()
        now = float(seen_ts or time.time())
        fact_id = self.hash_id(f"{gid}::{uid}::{fact}", "global", "impr_v3_fact")
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT evidence_count, first_seen FROM impression_facts_v3 WHERE fact_id = ?",
                (fact_id,),
            )
            row = cur.fetchone()
            prev_ev = int(row[0] or 0) if row else 0
            first_seen = float(row[1] or now) if row else now
            cur.execute(
                "INSERT OR REPLACE INTO impression_facts_v3(fact_id, group_id, user_id, fact_text, reason, confidence, evidence_count, source, first_seen, last_seen, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    fact_id,
                    gid,
                    uid,
                    fact,
                    str(reason or "").strip()[:180],
                    max(0.0, min(1.0, float(confidence))),
                    max(1, prev_ev + 1),
                    str(source or "").strip()[:40],
                    first_seen,
                    now,
                    time.time(),
                ),
            )
            self._db_conn.commit()
            cur.close()
            return fact_id
        except Exception:
            return ""

    def list_impression_facts_v3(
        self, group_id: str, user_id: str, limit: int = 8
    ) -> list[tuple[str, str, float, int, float]]:
        if not self._db_conn:
            self.init_db_conn()
        gid = str(group_id or "").strip()
        uid = str(user_id or "").strip()
        if not gid or not uid:
            return []
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT fact_text, reason, confidence, evidence_count, last_seen FROM impression_facts_v3 WHERE group_id = ? AND user_id = ? ORDER BY last_seen DESC LIMIT ?",
                (gid, uid, max(1, int(limit))),
            )
            rows = cur.fetchall()
            cur.close()
            out: list[tuple[str, str, float, int, float]] = []
            for r in rows:
                try:
                    out.append(
                        (
                            str(r[0] or ""),
                            str(r[1] or ""),
                            float(r[2] or 0.5),
                            int(r[3] or 1),
                            float(r[4] or 0.0),
                        )
                    )
                except Exception:
                    continue
            return out
        except Exception:
            return []

    def _upsert_impression_embedding_v3_sync(
        self,
        embedding_id: str,
        group_id: str,
        user_id: str,
        scope: str,
        ref_id: str,
        content: str,
        model_name: str,
        emb: list[float],
    ) -> None:
        if not self._db_conn:
            self.init_db_conn()
        now = time.time()
        cur = self._db_conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO impression_embeddings_v3(embedding_id, group_id, user_id, scope, ref_id, content, embedding, vector_dimension, model_name, created_at, last_updated) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                embedding_id,
                str(group_id or "").strip(),
                str(user_id or "").strip(),
                str(scope or "fact")[:16],
                str(ref_id or "")[:120],
                str(content or "")[:600],
                self._serialize_embedding(emb),
                len(emb),
                str(model_name or "text-embedding-3-small"),
                now,
                now,
            ),
        )
        self._db_conn.commit()
        cur.close()

    async def upsert_impression_embedding_v3(
        self,
        group_id: str,
        user_id: str,
        scope: str,
        ref_id: str,
        content: str,
    ) -> None:
        txt = str(content or "").strip()
        gid = str(group_id or "").strip()
        uid = str(user_id or "").strip()
        if not txt or not gid or not uid:
            return
        model_name = str(
            self.config.get("embedding_openai_model", "text-embedding-3-small")
            or "text-embedding-3-small"
        )
        emb = await self.get_embedding(txt)
        if not emb:
            return
        emb_id = self.hash_id(
            f"{gid}::{uid}::{scope}::{ref_id}::{txt}",
            "global",
            f"impr_v3::{model_name}",
        )
        try:
            async with self._db_write_lock:
                await asyncio.to_thread(
                    self._upsert_impression_embedding_v3_sync,
                    emb_id,
                    gid,
                    uid,
                    scope,
                    ref_id,
                    txt,
                    model_name,
                    emb,
                )
        except Exception:
            pass

    async def semantic_recall_impression_v3_with_qemb(
        self,
        qemb: list[float],
        group_id: str,
        limit: int = 12,
        threshold: float = 0.2,
        scan_limit: int | None = None,
    ) -> list[tuple[str, float, str, str]]:
        if not qemb:
            return []
        model_name = str(
            self.config.get("embedding_openai_model", "text-embedding-3-small")
            or "text-embedding-3-small"
        )
        gid = str(group_id or "").strip()
        if not gid:
            return []
        try:
            try:
                scan_limit_i = int(scan_limit or 0)
            except Exception:
                scan_limit_i = 0
            if scan_limit_i > 0:
                rows = await self._read_recall_rows(
                    "SELECT content, embedding, vector_dimension, user_id, scope FROM impression_embeddings_v3 WHERE group_id = ? AND model_name = ? ORDER BY last_updated DESC LIMIT ?",
                    (gid, model_name, scan_limit_i),
                    qemb,
                )
            else:
                rows = await self._read_recall_rows(
                    "SELECT content, embedding, vector_dimension, user_id, scope FROM impression_embeddings_v3 WHERE group_id = ? AND model_name = ?",
                    (gid, model_name),
                    qemb,
                )
            results: list[tuple[str, float, str, str]] = []
            for content, sim, uid, scope in rows:
                if sim < float(threshold):
                    continue
                results.append(
                    (str(content or ""), float(sim), str(uid or ""), str(scope or ""))
                )
            results.sort(key=lambda x: x[1], reverse=True)
            seen: set[str] = set()
            out: list[tuple[str, float, str, str]] = []
            for item in results:
                key = f"{item[2]}::{item[0]}"
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
                if len(out) >= max(1, int(limit)):
                    break
            return out
        except Exception:
            return []

    async def save_impression_v3_updates(
        self,
        *,
        group_id: str,
        target_user_id: str,
        target_display_name: str,
        source: str,
        summary: str = "",
        tags: list[str] | None = None,
        facts: list[dict] | None = None,
    ) -> None:
        gid = str(group_id or "").strip()
        uid = str(target_user_id or "").strip()
        if not gid or not uid:
            return
        display_name_clean = self._normalize_user_alias(target_display_name)
        if display_name_clean:
            try:
                async with self._db_write_lock:
                    await asyncio.to_thread(
                        self.upsert_user_alias,
                        gid,
                        uid,
                        display_name_clean,
                        source=source,
                        seen_ts=time.time(),
                        is_current=True,
                    )
            except Exception:
                pass
        clean_summary = str(summary or "").strip()
        clean_tags: list[str] = []
        for t in tags or []:
            s = str(t or "").strip()
            if s and s not in clean_tags:
                clean_tags.append(s[:24])
            if len(clean_tags) >= 12:
                break

        if clean_summary or clean_tags:
            try:
                async with self._db_write_lock:
                    await asyncio.to_thread(
                        self.upsert_impression_profile_v3,
                        gid,
                        uid,
                        str(target_display_name or "User"),
                        clean_summary,
                        clean_tags,
                        0.75,
                        source,
                    )
            except Exception:
                pass

        fact_items: list[tuple[str, str, float]] = []
        for item in facts or []:
            if not isinstance(item, dict):
                continue
            fact = str(item.get("fact") or item.get("impression") or "").strip()
            reason = str(item.get("reason") or "").strip()
            try:
                conf = float(item.get("confidence") or 0.6)
            except Exception:
                conf = 0.6
            if not fact:
                continue
            fact_items.append((fact[:120], reason[:180], max(0.0, min(1.0, conf))))
            if len(fact_items) >= 8:
                break

        for fact_text, reason, conf in fact_items:
            fact_id = ""
            try:
                async with self._db_write_lock:
                    fact_id = await asyncio.to_thread(
                        self.upsert_impression_fact_v3,
                        gid,
                        uid,
                        fact_text,
                        reason,
                        conf,
                        source,
                        time.time(),
                    )
            except Exception:
                fact_id = ""
            ref = fact_id or self.hash_id(
                f"{gid}::{uid}::{fact_text}", "global", "impr_v3_fact"
            )
            emb_txt = fact_text
            if reason:
                emb_txt = f"{fact_text}。依据：{reason}"
            await self.upsert_impression_embedding_v3(
                gid, uid, "fact", ref, emb_txt[:400]
            )

        if clean_summary or clean_tags:
            profile_txt = clean_summary
            if clean_tags:
                profile_txt = (profile_txt + " 标签：" + "、".join(clean_tags)).strip()
            if profile_txt:
                await self.upsert_impression_embedding_v3(
                    gid,
                    uid,
                    "profile",
                    f"profile::{gid}::{uid}",
                    profile_txt[:500],
                )

    async def _read_recall_rows(self, sql, params, qemb=None) -> list:
        """Read and optionally score embeddings without blocking the event loop.

        Args:
            sql: Parameterized read-only SQLite query.
            params: Query parameters.
            qemb: Optional query vector. Scored rows must start with content,
                embedding and dimension; those fields become content and score.

        Returns:
            Raw rows, or scored rows with their trailing metadata preserved.
        """
        cancelled = threading.Event()

        def read():
            deadline = time.monotonic() + 30.0
            connection = sqlite3.connect(
                Path(self.db_path).resolve().as_uri() + "?mode=ro",
                uri=True,
                timeout=2.0,
            )
            try:
                connection.set_progress_handler(
                    lambda: int(cancelled.is_set() or time.monotonic() >= deadline),
                    1000,
                )
                rows = []
                for row in connection.execute(sql, params):
                    if cancelled.is_set() or time.monotonic() >= deadline:
                        raise TimeoutError("Memory recall query cancelled or timed out")
                    if qemb is None:
                        rows.append(row)
                        continue
                    content, blob, dimension, *metadata = row
                    embedding = self._deserialize_embedding(blob, int(dimension or 0))
                    if embedding:
                        rows.append(
                            (
                                content,
                                self._cosine_similarity(qemb, embedding),
                                *metadata,
                            )
                        )
                return rows
            finally:
                connection.close()

        await self._recall_read_slots.acquire()
        task = asyncio.create_task(asyncio.to_thread(read))

        def finished(worker):
            self._recall_read_slots.release()
            # Retrieve errors even when the requesting coroutine was cancelled.
            if not worker.cancelled():
                worker.exception()

        task.add_done_callback(finished)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        except Exception:
            logger.exception("[mem_chat] Background memory recall failed")
            raise

    async def semantic_recall(
        self,
        query: str,
        group_id: str,
        limit: int,
        threshold: float,
        type: str = "memory",
    ) -> list[tuple[str, float]]:
        qemb = await self.get_embedding(query)
        if not qemb:
            return []

        model_name = str(
            self.config.get("embedding_openai_model", "text-embedding-3-small")
            or "text-embedding-3-small"
        )
        try:
            if type in ["memory", "impression"]:
                scan_limit = 0
                if type == "memory":
                    scan_limit = int(
                        self.config.get("semantic_recall_scan_limit_memory", 8000) or 0
                    )
                elif type == "impression":
                    scan_limit = int(
                        self.config.get("semantic_recall_scan_limit_impression", 2000)
                        or 0
                    )
                if scan_limit > 0:
                    sql = "SELECT content, embedding, vector_dimension FROM memory_embeddings WHERE group_id = ? AND model_name = ? AND type = ? ORDER BY created_at DESC LIMIT ?"
                    params = (group_id, model_name, type, scan_limit)
                else:
                    sql = "SELECT content, embedding, vector_dimension FROM memory_embeddings WHERE group_id = ? AND model_name = ? AND type = ?"
                    params = (group_id, model_name, type)
            else:
                if group_id == "protocol_admin":
                    sql = "SELECT content, embedding, vector_dimension FROM memory_embeddings WHERE group_id = ? AND model_name = ? AND type = ?"
                    params = (group_id, model_name, type)
                else:
                    sql = "SELECT content, embedding, vector_dimension FROM memory_embeddings WHERE (group_id = ? OR group_id = 'global') AND model_name = ? AND type = ?"
                    params = (group_id, model_name, type)

            results: list[tuple[str, float]] = []
            for content, sim in await self._read_recall_rows(sql, params, qemb):
                if sim > threshold:
                    results.append((content, sim))
            results.sort(key=lambda x: x[1], reverse=True)
            seen = set()
            deduped: list[tuple[str, float]] = []
            for content, sim in results:
                key = " ".join(str(content or "").split())
                if not key or key in seen:
                    continue
                seen.add(key)
                deduped.append((content, sim))
            results = deduped
            final_res = results[:limit]
            logger.info(
                f"[mem_chat] semantic_recall[{type}]: found {len(results)} candidates above threshold {threshold}, returning {len(final_res)} items. Top score={results[0][1] if results else 0:.4f}"
            )
            return final_res
        except Exception:
            return []

    async def semantic_recall_with_qemb(
        self,
        qemb: list[float],
        group_id: str,
        limit: int,
        threshold: float,
        type: str = "memory",
        scan_limit: int | None = None,
    ) -> list[tuple[str, float]]:
        if not qemb:
            return []
        model_name = str(
            self.config.get("embedding_openai_model", "text-embedding-3-small")
            or "text-embedding-3-small"
        )
        now_ts = time.time()
        try:
            scan_limit_override = int(scan_limit) if scan_limit is not None else None
        except Exception:
            scan_limit_override = None

        def _apply_time_decay(sim: float, created_at_val) -> float:
            if type != "memory":
                return float(sim)
            if not bool(self.config.get("memory_time_decay_enabled", True)):
                return float(sim)
            try:
                created_ts = float(created_at_val or 0.0)
            except Exception:
                created_ts = 0.0
            if created_ts <= 0.0:
                return float(sim)
            age_sec = max(0.0, now_ts - created_ts)
            try:
                half_life = float(
                    self.config.get("memory_time_decay_half_life_sec", 7 * 24 * 3600)
                    or (7 * 24 * 3600)
                )
            except Exception:
                half_life = float(7 * 24 * 3600)
            if half_life <= 0:
                return float(sim)
            try:
                min_factor = float(
                    self.config.get("memory_time_decay_min_factor", 0.30) or 0.30
                )
            except Exception:
                min_factor = 0.30
            if min_factor < 0.0:
                min_factor = 0.0
            if min_factor > 1.0:
                min_factor = 1.0
            factor = 0.5 ** (age_sec / half_life)
            if factor < min_factor:
                factor = min_factor
            if factor > 1.0:
                factor = 1.0
            return float(sim) * float(factor)

        def _blended_scores() -> bool:
            """True when scores come from the weighted blend, not raw cosine."""
            return bool(self.config.get("ga_scoring_enabled", True)) and type in (
                "memory",
                "reflection",
            )

        def _ga_score(sim: float, created_at_val, importance_val) -> float:
            """Blend relevance, recency, and importance into one score.

            Pure multiplication lets a single weak factor annihilate an
            otherwise excellent match, so the three signals are combined as a
            weighted sum in the style of Generative Agents. Relevance keeps the
            largest default weight to stay close to the previous ranking.

            Args:
                sim: Cosine relevance from the vector index.
                created_at_val: Row creation timestamp.
                importance_val: Stored salience in 0..1.

            Returns:
                The combined score, or the decay-only score when the blend is
                disabled.
            """
            if not bool(self.config.get("ga_scoring_enabled", True)):
                return _apply_time_decay(sim, created_at_val)
            if type not in ("memory", "reflection"):
                return _apply_time_decay(sim, created_at_val)
            try:
                w_rel = float(self.config.get("ga_weight_relevance", 1.0))
                w_rec = float(self.config.get("ga_weight_recency", 0.5))
                w_imp = float(self.config.get("ga_weight_importance", 0.5))
            except (TypeError, ValueError):
                w_rel, w_rec, w_imp = 1.0, 0.5, 0.5
            total = w_rel + w_rec + w_imp
            if total <= 0:
                return _apply_time_decay(sim, created_at_val)
            try:
                created_ts = float(created_at_val or 0.0)
            except (TypeError, ValueError):
                created_ts = 0.0
            if created_ts > 0.0:
                # The multiplicative path uses a 7-day half-life, which drives
                # month-old rows to ~0.05 and would let recency veto an
                # important memory even inside a weighted sum. Recency gets its
                # own, longer half-life here so importance stays meaningful.
                try:
                    half_life = float(
                        self.config.get("ga_recency_half_life_sec", 30 * 24 * 3600)
                        or (30 * 24 * 3600)
                    )
                except (TypeError, ValueError):
                    half_life = float(30 * 24 * 3600)
                if half_life <= 0:
                    recency = 1.0
                else:
                    recency = 0.5 ** (max(0.0, now_ts - created_ts) / half_life)
            else:
                recency = 0.5
            try:
                importance = float(
                    importance_val if importance_val is not None else 0.5
                )
            except (TypeError, ValueError):
                importance = 0.5
            importance = max(0.0, min(1.0, importance))
            relevance = max(0.0, min(1.0, float(sim)))
            return (w_rel * relevance + w_rec * recency + w_imp * importance) / total

        try:
            use_faiss = self._faiss_enabled() and bool(faiss) and bool(np)
            if use_faiss:
                dim = int(len(qemb))
                if dim <= 0:
                    return []
                if self._vector_dimension is None:
                    self._vector_dimension = dim

                def _post_filter(
                    results: list[tuple[str, float]],
                ) -> list[tuple[str, float]]:
                    results.sort(key=lambda x: x[1], reverse=True)
                    blended_scores = _blended_scores()
                    use_dynamic = bool(
                        self.config.get("dynamic_threshold_enable", True)
                    )
                    if use_dynamic:
                        try:
                            smax = results[0][1] if results else 0.0
                            floor = float(
                                self.config.get("dynamic_threshold_floor", 0.0)
                            )
                            if type == "persona":
                                alpha = float(
                                    self.config.get("dynamic_alpha_persona", 0.92)
                                )
                            elif type == "impression":
                                alpha = float(
                                    self.config.get("dynamic_alpha_impression", 0.88)
                                )
                            elif type == "memory":
                                alpha = float(
                                    self.config.get("dynamic_alpha_memory", 0.85)
                                )
                            else:
                                alpha = float(
                                    self.config.get("dynamic_alpha_knowledge", 0.80)
                                )
                            # The caller's fixed threshold was calibrated
                            # against raw cosine similarity. Blended GA scores
                            # live on a different scale, so honour only the
                            # relative (alpha * best) cut there; otherwise a
                            # cosine-era constant would discard everything.
                            if blended_scores:
                                thr = alpha * smax
                            else:
                                thr = threshold if threshold > 0.0 else (alpha * smax)
                                if thr < floor:
                                    thr = floor
                            results = [x for x in results if x[1] >= thr]
                        except Exception:
                            pass
                    else:
                        if threshold > 0.0 and not blended_scores:
                            results = [x for x in results if x[1] >= threshold]
                    seen = set()
                    deduped: list[tuple[str, float]] = []
                    for content, sim in results:
                        key = " ".join(str(content or "").split())
                        if not key or key in seen:
                            continue
                        seen.add(key)
                        deduped.append((content, sim))
                    return deduped[:limit]

                async def _search_bucket(target_group: str) -> list[tuple[str, float]]:
                    bucket_key = await self._ensure_faiss_bucket_ready(
                        target_group,
                        model_name,
                        type,
                        dim,
                    )
                    if not bucket_key:
                        return []
                    try:
                        k = int(self.config.get("faiss_search_k", 200) or 0)
                    except Exception:
                        k = 200
                    k = max(k, limit * 8, 32)
                    qv = np.asarray([qemb], dtype=np.float32)
                    faiss.normalize_L2(qv)
                    async with self._faiss_lock:
                        idx = self._faiss_indexes.get(bucket_key)
                        if not idx:
                            return []
                        distances, indices = idx.search(qv, int(k))
                    ids = [int(x) for x in list(indices[0]) if int(x) != -1]
                    if not ids:
                        return []
                    score_by_id: dict[int, float] = {}
                    for i, fid in enumerate(list(indices[0])):
                        fid_i = int(fid)
                        if fid_i == -1:
                            continue
                        try:
                            score_by_id[fid_i] = float(distances[0][i])
                        except Exception:
                            score_by_id[fid_i] = 0.0

                    if not self._db_conn:
                        self.init_db_conn()
                    if not self._db_conn:
                        return []

                    cur = self._db_conn.cursor()
                    out: list[tuple[str, float]] = []
                    try:
                        chunk_size = 900
                        for i in range(0, len(ids), chunk_size):
                            chunk = ids[i : i + chunk_size]
                            placeholders = ",".join("?" for _ in chunk)
                            sql = (
                                "SELECT content, faiss_id, created_at, importance FROM memory_embeddings "
                                f"WHERE group_id = ? AND model_name = ? AND type = ? AND faiss_id IN ({placeholders})"
                            )
                            params = [target_group, model_name, type, *chunk]
                            cur.execute(sql, params)
                            for content, fid, created_at, importance in cur.fetchall():
                                try:
                                    fid_i = int(fid)
                                except Exception:
                                    continue
                                sim = float(score_by_id.get(fid_i, 0.0))
                                sim = _ga_score(sim, created_at, importance)
                                if content:
                                    out.append((content, sim))
                    finally:
                        try:
                            cur.close()
                        except Exception:
                            pass
                    return out

                if type in ["persona", "knowledge", "persona_rag"]:
                    res_local = await _search_bucket(group_id)
                    res_global = []
                    if group_id != "protocol_admin":
                        res_global = await _search_bucket("global")
                    merged = res_local + res_global
                    return _post_filter(merged)

                res = await _search_bucket(group_id)
                return _post_filter(res)

            if type in ["memory", "impression"]:
                scan_limit = 0
                if type == "memory":
                    scan_limit = int(
                        self.config.get("semantic_recall_scan_limit_memory", 8000) or 0
                    )
                elif type == "impression":
                    scan_limit = int(
                        self.config.get("semantic_recall_scan_limit_impression", 2000)
                        or 0
                    )
                if scan_limit_override is not None:
                    scan_limit = max(0, scan_limit_override)
                if scan_limit > 0:
                    sql = "SELECT content, embedding, vector_dimension, created_at, importance FROM memory_embeddings WHERE group_id = ? AND model_name = ? AND type = ? ORDER BY created_at DESC LIMIT ?"
                    params = (group_id, model_name, type, scan_limit)
                else:
                    sql = "SELECT content, embedding, vector_dimension, created_at, importance FROM memory_embeddings WHERE group_id = ? AND model_name = ? AND type = ?"
                    params = (group_id, model_name, type)
            else:
                if group_id == "protocol_admin":
                    sql = "SELECT content, embedding, vector_dimension, created_at, importance FROM memory_embeddings WHERE group_id = ? AND model_name = ? AND type = ?"
                    params = (group_id, model_name, type)
                else:
                    sql = "SELECT content, embedding, vector_dimension, created_at, importance FROM memory_embeddings WHERE (group_id = ? OR group_id = 'global') AND model_name = ? AND type = ?"
                    params = (group_id, model_name, type)
            results: list[tuple[str, float]] = []
            for content, sim, created_at, importance in await self._read_recall_rows(
                sql, params, qemb
            ):
                results.append((content, _ga_score(sim, created_at, importance)))
            results.sort(key=lambda x: x[1], reverse=True)
            use_dynamic = bool(self.config.get("dynamic_threshold_enable", True))
            if type != "memory":
                if type == "persona_rag":
                    seen = set()
                    deduped: list[tuple[str, float]] = []
                    for content, sim in results:
                        key = " ".join(str(content or "").split())
                        if not key or key in seen:
                            continue
                        seen.add(key)
                        deduped.append((content, sim))
                    return deduped[:limit]
                if use_dynamic:
                    try:
                        smax = results[0][1] if results else 0.0
                        floor = float(self.config.get("dynamic_threshold_floor", 0.0))
                        if type == "persona":
                            alpha = float(
                                self.config.get("dynamic_alpha_persona", 0.92)
                            )
                        elif type == "impression":
                            alpha = float(
                                self.config.get("dynamic_alpha_impression", 0.88)
                            )
                        elif type == "memory":
                            alpha = float(self.config.get("dynamic_alpha_memory", 0.85))
                        else:
                            alpha = float(
                                self.config.get("dynamic_alpha_knowledge", 0.80)
                            )
                        if _blended_scores():
                            thr = alpha * smax
                        else:
                            thr = threshold if threshold > 0.0 else (alpha * smax)
                            if thr < floor:
                                thr = floor
                        results = [x for x in results if x[1] >= thr]
                    except Exception:
                        pass
                else:
                    if threshold > 0.0:
                        results = [x for x in results if x[1] >= threshold]
            seen = set()
            deduped: list[tuple[str, float]] = []
            for content, sim in results:
                key = " ".join(str(content or "").split())
                if not key or key in seen:
                    continue
                seen.add(key)
                deduped.append((content, sim))
            results = deduped
            final_res = results[:limit]
            logger.info(
                f"[mem_chat] semantic_recall[{type}]: found {len(results)} candidates above threshold {threshold}, returning {len(final_res)} items. Top score={results[0][1] if results else 0:.4f}"
            )
            return final_res
        except Exception:
            return []

    def get_impression(self, group_id: str, sender_id: str) -> str | None:
        if not self._db_conn:
            self.init_db_conn()
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT summary FROM impressions WHERE group_id = ? AND user_id = ?",
                (group_id, sender_id),
            )
            row = cur.fetchone()
            cur.close()
            if row:
                return row[0]
            return None
        except Exception as e:
            logger.error(f"[mem_chat] Failed to get impression: {e}")
            return None

    def save_impression(
        self, group_id: str, sender_id: str, summary: str, name: str = "User"
    ):
        if not self._db_conn:
            self.init_db_conn()
        try:
            cur = self._db_conn.cursor()
            self._upsert_user_alias_with_cursor(
                cur,
                group_id,
                sender_id,
                name,
                source="legacy_impression",
                seen_ts=time.time(),
                is_current=True,
            )
            cur.execute(
                "INSERT OR REPLACE INTO impressions (name, group_id, summary, user_id, score) VALUES (?, ?, ?, ?, ?)",
                (name, group_id, summary, sender_id, 0.0),
            )
            self._db_conn.commit()
            cur.close()
        except Exception as e:
            logger.error(f"[mem_chat] Failed to save impression: {e}")

    def list_impressions(self, group_id: str) -> list[tuple[str, str, str]]:
        if not self._db_conn:
            self.init_db_conn()
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT user_id, name, summary FROM impressions WHERE group_id = ?",
                (group_id,),
            )
            rows = cur.fetchall()
            cur.close()
            ret = []
            for r in rows:
                try:
                    uid = str(r[0])
                    name = r[1] or "User"
                    summary = r[2] or ""
                    ret.append((uid, name, summary))
                except Exception:
                    continue
            return ret
        except Exception as e:
            logger.error(f"[mem_chat] Failed to list impressions: {e}")
            return []

    def clean_legacy_content(self, content: str) -> str:
        import re

        if not content:
            return ""
        text = content
        try:
            text = re.sub(r"\[User ID:.*?, Nickname:.*?\]\s*", "", text)
            text = re.sub(r"\[User ID:.*?\]\s*", "", text)

            headers = [
                "【最近对话】",
                "【记忆库上下文】",
                "【相关记忆】",
                "【人物印象】",
                "【人物印象/设定】",
                "【人物印象-相关人物】",
                "【最近聊天消息】",
                "【历史对话】",
                "【相关知识】",
                "【相关人格】",
                "【关系倾向】",
                "【需要回复的群聊对话】",
                "【回复提示词】",
                "【表情包提示词】",
                "【爱丽丝语料片段】",
                "【我的观察】",
            ]
            # A leaked prompt block starts with one of these headers, so
            # dropping it keeps injected context out of the memory store. A
            # normal chat line that merely mentions a header mid-sentence used
            # to be dropped too, which silently erased real messages from the
            # history window; strip the marker and keep the sentence instead.
            stripped = text.lstrip()
            for h in headers:
                if stripped.startswith(h):
                    return ""
            for h in headers:
                if h in text:
                    text = text.replace(h, "")
        except Exception:
            pass
        return text.strip()

    async def get_recent_memories(self, group_id: str, limit: int) -> list[str]:
        try:
            if not self._db_conn:
                self.init_db_conn()
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT content FROM memory_embeddings WHERE group_id = ? ORDER BY created_at DESC LIMIT ?",
                (group_id, limit),
            )
            rows = cur.fetchall()
            cur.close()
            results = []
            for r in rows:
                c = self.clean_legacy_content(r[0])
                if c:
                    results.append(c)
            return results[::-1]
        except Exception:
            return []

    async def get_user_specific_history(
        self, group_id: str, user_id: str, limit: int = 20
    ) -> list[str]:
        cur = None
        try:
            if not self._db_conn:
                self.init_db_conn()
            cur = self._db_conn.cursor()
            like_pattern = f"%/{user_id}: %"
            cur.execute(
                "SELECT content FROM memory_embeddings WHERE group_id = ? AND content LIKE ? ORDER BY created_at DESC LIMIT ?",
                (group_id, like_pattern, limit),
            )
            rows = cur.fetchall()

            results = []
            for r in rows:
                c = self.clean_legacy_content(r[0])
                if c:
                    results.append(c)
            return results[::-1]
        except Exception as e:
            logger.error(f"[mem_chat] Failed to get user specific history: {e}")
            return []
        finally:
            if cur:
                cur.close()

    async def save_message(
        self, content: str, group_id: str, sender_info: str | None = None
    ):
        if "【最近对话】" in content or "【记忆库上下文】" in content:
            return

        model_name = "no-embedding"
        final_content = f"{sender_info}: {content}" if sender_info else content
        mid = self.hash_id(final_content or "", group_id, model_name)

        now = time.time()
        if not self._db_conn:
            self.init_db_conn()
        try:
            cur = self._db_conn.cursor()
            faiss_id = self._alloc_faiss_id_db(mid)
            cur.execute(
                "INSERT OR REPLACE INTO memory_embeddings(memory_id, content, embedding, vector_dimension, faiss_id, group_id, model_name, type, created_at, last_updated) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    mid,
                    final_content,
                    b"",
                    0,
                    faiss_id,
                    group_id,
                    model_name,
                    "memory",
                    now,
                    now,
                ),
            )
            self._db_conn.commit()
            cur.close()
        except Exception as e:
            logger.error(f"[mem_chat] Failed to save message: {e}")

    async def ensure_message_embedded(
        self,
        content: str,
        group_id: str,
        sender_info: str | None = None,
        importance: float | None = None,
    ):
        """Embed and store one chat line as a memory.

        Args:
            content: Message text.
            group_id: Owning group bucket.
            sender_info: Optional "name/qq" prefix prepended to the stored text.
            importance: Optional 0..1 salience from the model. When omitted the
                configured default is used, which keeps retrieval behaviour
                unchanged for rows written without a score.
        """
        if (
            "【最近对话】" in content
            or "【记忆库上下文】" in content
            or "【相关记忆】" in content
        ):
            return

        model_name = str(
            self.config.get("embedding_openai_model", "text-embedding-3-small")
            or "text-embedding-3-small"
        )
        # An unscored duplicate must preserve a score assigned after the
        # original write; get_or_compute_embedding supplies the new-row default.
        final_content = f"{sender_info}: {content}" if sender_info else content
        mid = self.hash_id(final_content or "", group_id, model_name)
        await self.get_or_compute_embedding(
            mid, final_content or "", group_id, model_name, importance=importance
        )

    async def save_raw_group_message(
        self,
        *,
        group_id: str,
        message_id: str,
        message_seq: str | None,
        sender_id: str,
        sender_name: str | None,
        text: str,
        raw_text: str | None = None,
        image_urls_json: str | None = None,
        reply_to_msg_id: str | None = None,
        reply_to_sender_id: str | None = None,
        reply_to_sender_name: str | None = None,
        at_targets_json: str | None = None,
        at_all: bool = False,
        explicit_target_type: str | None = None,
        explicit_target_id: str | None = None,
        explicit_target_name: str | None = None,
        created_at: float | None = None,
    ) -> None:
        gid = str(group_id or "").strip()
        mid = str(message_id or "").strip()
        sid = str(sender_id or "").strip()
        txt = str(text or "").strip()
        if not gid or not mid or not sid or not txt:
            return
        now = float(created_at or time.time())
        mseq = str(message_seq or "").strip()
        sname = str(sender_name or "").strip()
        raw = str(raw_text or "").strip()
        image_urls = str(image_urls_json or "").strip()
        reply_mid = str(reply_to_msg_id or "").strip()
        reply_sid = str(reply_to_sender_id or "").strip()
        reply_sname = str(reply_to_sender_name or "").strip()
        at_targets = str(at_targets_json or "").strip()
        exp_type = str(explicit_target_type or "").strip()
        exp_id = str(explicit_target_id or "").strip()
        exp_name = str(explicit_target_name or "").strip()
        if not self._db_conn:
            self.init_db_conn()
        try:
            async with self._db_write_lock:
                cur = self._db_conn.cursor()
                cur.execute(
                    """
                    INSERT INTO raw_group_messages(
                        group_id, message_id, message_seq, sender_id, sender_name, text, raw_text, image_urls_json,
                        reply_to_msg_id, reply_to_sender_id, reply_to_sender_name,
                        at_targets_json, at_all, explicit_target_type, explicit_target_id, explicit_target_name,
                        created_at, updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(group_id, message_id) DO UPDATE SET
                        message_seq=excluded.message_seq,
                        sender_id=excluded.sender_id,
                        sender_name=excluded.sender_name,
                        text=excluded.text,
                        raw_text=excluded.raw_text,
                        image_urls_json=excluded.image_urls_json,
                        reply_to_msg_id=excluded.reply_to_msg_id,
                        reply_to_sender_id=excluded.reply_to_sender_id,
                        reply_to_sender_name=excluded.reply_to_sender_name,
                        at_targets_json=excluded.at_targets_json,
                        at_all=excluded.at_all,
                        explicit_target_type=excluded.explicit_target_type,
                        explicit_target_id=excluded.explicit_target_id,
                        explicit_target_name=excluded.explicit_target_name,
                        updated_at=excluded.updated_at
                    """,
                    (
                        gid,
                        mid,
                        mseq,
                        sid,
                        sname,
                        txt,
                        raw,
                        image_urls,
                        reply_mid,
                        reply_sid,
                        reply_sname,
                        at_targets,
                        1 if at_all else 0,
                        exp_type,
                        exp_id,
                        exp_name,
                        now,
                        time.time(),
                    ),
                )
                self._upsert_user_alias_with_cursor(
                    cur,
                    gid,
                    sid,
                    sname,
                    source="message",
                    seen_ts=now,
                    is_current=True,
                )
                try:
                    at_items = json.loads(at_targets) if at_targets else []
                except Exception:
                    at_items = []
                if isinstance(at_items, list):
                    for item in at_items:
                        if not isinstance(item, dict):
                            continue
                        at_uid = str(item.get("id") or item.get("qq") or "").strip()
                        at_name = str(item.get("name") or "").strip()
                        if at_uid and at_name:
                            self._upsert_user_alias_with_cursor(
                                cur,
                                gid,
                                at_uid,
                                at_name,
                                source="at_target",
                                seen_ts=now,
                                is_current=False,
                            )
                self._db_conn.commit()
                cur.close()
        except Exception as e:
            logger.error(f"[mem_chat] Failed to save raw group message: {e}")

    async def get_recent_raw_group_messages(
        self,
        group_id: str,
        limit: int = 30,
    ) -> list[tuple[str, str, str, float]]:
        """获取最近群聊原始消息，返回 [(sender_name, sender_id, text, created_at), ...]"""
        gid = str(group_id or "").strip()
        if not gid:
            return []
        if not self._db_conn:
            self.init_db_conn()
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                """
                SELECT sender_name, sender_id, text, created_at
                FROM raw_group_messages
                WHERE group_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (gid, max(1, int(limit or 30))),
            )
            rows = cur.fetchall()
            cur.close()
            return [
                (r[0] or "", r[1] or "", r[2] or "", float(r[3] or 0.0)) for r in rows
            ]
        except Exception as e:
            logger.error(f"[mem_chat] Failed to get recent raw group messages: {e}")
            return []

    async def backfill_raw_group_messages_from_memory_embeddings(
        self,
        *,
        batch_size: int = 2000,
        force: bool = False,
    ) -> dict[str, int]:
        if not self._db_conn:
            self.init_db_conn()
        meta_key = "raw_group_messages_backfill_v1_done"
        if not force:
            done = await self.get_meta(meta_key)
            if str(done or "").strip() == "1":
                return {"scanned": 0, "parsed": 0, "inserted": 0, "skipped": 1}

        bs = max(200, int(batch_size or 2000))
        scanned = 0
        parsed = 0
        inserted = 0
        pat = re.compile(r"^\s*(.+?)\s*/\s*(\d{5,20})\s*:\s*(.+)\s*$")

        cur = None
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                """
                SELECT memory_id, group_id, content, created_at
                FROM memory_embeddings
                WHERE content LIKE '%/%:%'
                ORDER BY created_at ASC
                """
            )
            while True:
                rows = cur.fetchmany(bs)
                if not rows:
                    break
                scanned += len(rows)
                now = time.time()
                write_rows: list[tuple] = []
                alias_rows: list[tuple[str, str, str, float]] = []
                for memory_id, group_id, content, created_at in rows:
                    m = pat.match(str(content or "").strip())
                    if not m:
                        continue
                    sender_name = str(m.group(1) or "").strip()
                    sender_id = str(m.group(2) or "").strip()
                    text = str(m.group(3) or "").strip()
                    gid = str(group_id or "").strip()
                    if not gid or not sender_id or not text:
                        continue
                    parsed += 1
                    mid = f"mem:{str(memory_id or '').strip()}"
                    if not mid.strip():
                        continue
                    ts = float(created_at or now)
                    write_rows.append(
                        (
                            gid,
                            mid,
                            "",
                            sender_id,
                            sender_name,
                            text,
                            str(content or ""),
                            ts,
                            now,
                        )
                    )
                    alias_rows.append((gid, sender_id, sender_name, ts))
                if not write_rows:
                    continue
                async with self._db_write_lock:
                    wcur = self._db_conn.cursor()
                    wcur.executemany(
                        """
                        INSERT OR IGNORE INTO raw_group_messages(
                            group_id, message_id, message_seq, sender_id, sender_name, text, raw_text, created_at, updated_at
                        ) VALUES(?,?,?,?,?,?,?,?,?)
                        """,
                        write_rows,
                    )
                    inserted += int(wcur.rowcount or 0)
                    for gid, sender_id, sender_name, seen_ts in alias_rows:
                        self._upsert_user_alias_with_cursor(
                            wcur,
                            gid,
                            sender_id,
                            sender_name,
                            source="memory_backfill",
                            seen_ts=seen_ts,
                            is_current=False,
                        )
                    self._db_conn.commit()
                    wcur.close()
            await self.set_meta(meta_key, "1")
            return {
                "scanned": int(scanned),
                "parsed": int(parsed),
                "inserted": int(inserted),
                "skipped": 0,
            }
        except Exception as e:
            logger.error(f"[mem_chat] Failed to backfill raw_group_messages: {e}")
            return {
                "scanned": int(scanned),
                "parsed": int(parsed),
                "inserted": int(inserted),
                "skipped": 0,
            }
        finally:
            try:
                if cur:
                    cur.close()
            except Exception:
                pass

    async def backfill_user_aliases_from_raw_group_messages(
        self,
        *,
        batch_size: int = 5000,
        force: bool = False,
    ) -> dict[str, int]:
        if not self._db_conn:
            self.init_db_conn()
        meta_key = "user_aliases_backfill_v1_done"
        if not force:
            done = await self.get_meta(meta_key)
            if str(done or "").strip() == "1":
                return {"scanned": 0, "upserted": 0, "skipped": 1}

        bs = max(200, int(batch_size or 5000))
        scanned = 0
        upserted = 0
        cur = None
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                """
                SELECT group_id, sender_id, sender_name, created_at
                FROM raw_group_messages
                WHERE sender_id IS NOT NULL
                  AND sender_id != ''
                  AND sender_name IS NOT NULL
                  AND sender_name != ''
                ORDER BY created_at ASC
                """
            )
            while True:
                rows = cur.fetchmany(bs)
                if not rows:
                    break
                scanned += len(rows)
                async with self._db_write_lock:
                    wcur = self._db_conn.cursor()
                    for gid, uid, alias, seen_ts in rows:
                        if self._upsert_user_alias_with_cursor(
                            wcur,
                            str(gid or ""),
                            str(uid or ""),
                            str(alias or ""),
                            source="raw_backfill",
                            seen_ts=float(seen_ts or time.time()),
                            is_current=True,
                        ):
                            upserted += 1
                    self._db_conn.commit()
                    wcur.close()
            await self.set_meta(meta_key, "1")
            return {"scanned": int(scanned), "upserted": int(upserted), "skipped": 0}
        except Exception as e:
            logger.error(f"[mem_chat] Failed to backfill user aliases: {e}")
            return {"scanned": int(scanned), "upserted": int(upserted), "skipped": 0}
        finally:
            try:
                if cur:
                    cur.close()
            except Exception:
                pass

    async def get_count_by_type(self, type_: str) -> int:
        if not self._db_conn:
            self.init_db_conn()
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM memory_embeddings WHERE type = ?", (type_,)
            )
            row = cur.fetchone()
            cur.close()
            return row[0] if row else 0
        except Exception as e:
            logger.error(f"[mem_chat] Failed to count type {type_}: {e}")
            return 0

    async def get_all_ids_by_type(self, type_: str) -> set[str]:
        if not self._db_conn:
            self.init_db_conn()
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT memory_id FROM memory_embeddings WHERE type = ?", (type_,)
            )
            rows = cur.fetchall()
            cur.close()
            return {r[0] for r in rows}
        except Exception as e:
            logger.error(f"[mem_chat] Failed to get IDs for type {type_}: {e}")
            return set()

    async def delete_by_ids(self, ids: list[str]):
        if not ids:
            return
        if not self._db_conn:
            self.init_db_conn()
        try:
            bucket_to_ids: dict[str, list[int]] = {}
            cur = self._db_conn.cursor()
            chunk_size = 900
            for i in range(0, len(ids), chunk_size):
                chunk = ids[i : i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                try:
                    cur.execute(
                        f"SELECT group_id, model_name, type, faiss_id FROM memory_embeddings WHERE memory_id IN ({placeholders}) AND faiss_id IS NOT NULL",
                        chunk,
                    )
                    for gid, mname, tp, fid in cur.fetchall():
                        try:
                            fid_i = int(fid)
                        except Exception:
                            continue
                        bkey = self._bucket_key(
                            str(gid or ""), str(mname or ""), str(tp or "")
                        )
                        bucket_to_ids.setdefault(bkey, []).append(fid_i)
                except Exception:
                    pass
                cur.execute(
                    f"DELETE FROM memory_embeddings WHERE memory_id IN ({placeholders})",
                    chunk,
                )
            self._db_conn.commit()
            cur.close()
            logger.info(f"[mem_chat] Deleted {len(ids)} entries by ID")
            try:
                for bkey, fid_list in bucket_to_ids.items():
                    if not fid_list:
                        continue
                    parts = str(bkey).split("::", 2)
                    if len(parts) != 3:
                        continue
                    gid, mname, tp = parts
                    await self._faiss_delete_if_ready(gid, mname, tp, fid_list)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"[mem_chat] Failed to delete by IDs: {e}")

    async def get_latest_personas(self, limit: int = 3) -> list[str]:
        if not self._db_conn:
            self.init_db_conn()
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT content FROM memory_embeddings WHERE type = 'persona' AND group_id = 'global' ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
            cur.close()
            return [r[0] for r in rows]
        except Exception as e:
            logger.error(f"[mem_chat] Failed to get latest personas: {e}")
            return []

    async def delete_by_type(self, type_: str):
        if not self._db_conn:
            self.init_db_conn()
        try:
            bucket_to_ids: dict[str, list[int]] = {}
            cur = self._db_conn.cursor()
            try:
                cur.execute(
                    "SELECT group_id, model_name, type, faiss_id FROM memory_embeddings WHERE type = ? AND faiss_id IS NOT NULL",
                    (type_,),
                )
                for gid, mname, tp, fid in cur.fetchall():
                    try:
                        fid_i = int(fid)
                    except Exception:
                        continue
                    bkey = self._bucket_key(
                        str(gid or ""), str(mname or ""), str(tp or "")
                    )
                    bucket_to_ids.setdefault(bkey, []).append(fid_i)
            except Exception:
                pass
            cur.execute("DELETE FROM memory_embeddings WHERE type = ?", (type_,))
            self._db_conn.commit()
            cur.close()
            logger.info(f"[mem_chat] Deleted all entries of type {type_}")
            try:
                for bkey, fid_list in bucket_to_ids.items():
                    if not fid_list:
                        continue
                    parts = str(bkey).split("::", 2)
                    if len(parts) != 3:
                        continue
                    gid, mname, tp = parts
                    await self._faiss_delete_if_ready(gid, mname, tp, fid_list)
                    try:
                        if bkey in self._faiss_ready_buckets:
                            self._faiss_ready_buckets.discard(bkey)
                        if bkey in self._faiss_indexes:
                            self._faiss_indexes.pop(bkey, None)
                        self._faiss_last_save_ts.pop(bkey, None)
                        p = self._bucket_index_path(bkey)
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception as e:
            logger.error(f"[mem_chat] Failed to delete type {type_}: {e}")

    async def delete_by_content(self, content_snippet: str):
        if not self._db_conn:
            self.init_db_conn()
        try:
            bucket_to_ids: dict[str, list[int]] = {}
            cur = self._db_conn.cursor()
            try:
                cur.execute(
                    "SELECT group_id, model_name, type, faiss_id FROM memory_embeddings WHERE content LIKE ? AND faiss_id IS NOT NULL",
                    (f"%{content_snippet}%",),
                )
                for gid, mname, tp, fid in cur.fetchall():
                    try:
                        fid_i = int(fid)
                    except Exception:
                        continue
                    bkey = self._bucket_key(
                        str(gid or ""), str(mname or ""), str(tp or "")
                    )
                    bucket_to_ids.setdefault(bkey, []).append(fid_i)
            except Exception:
                pass
            cur.execute(
                "DELETE FROM memory_embeddings WHERE content LIKE ?",
                (f"%{content_snippet}%",),
            )
            deleted = cur.rowcount
            self._db_conn.commit()
            cur.close()
            logger.info(
                f"[mem_chat] Deleted {deleted} entries matching '{content_snippet}'"
            )
            try:
                for bkey, fid_list in bucket_to_ids.items():
                    if not fid_list:
                        continue
                    parts = str(bkey).split("::", 2)
                    if len(parts) != 3:
                        continue
                    gid, mname, tp = parts
                    await self._faiss_delete_if_ready(gid, mname, tp, fid_list)
            except Exception:
                pass
        except Exception as e:
            logger.error(
                f"[mem_chat] Failed to delete content '{content_snippet}': {e}"
            )

    async def delete_knowledge_by_like_patterns(self, patterns: list[str]) -> int:
        if not patterns:
            return 0
        if not self._db_conn:
            self.init_db_conn()
        try:
            cur = self._db_conn.cursor()
            clauses = []
            params: list[str] = []
            for p in patterns:
                s = str(p or "").strip().lower()
                if not s:
                    continue
                clauses.append("LOWER(content) LIKE ?")
                params.append(f"%{s}%")
            if not clauses:
                cur.close()
                return 0
            sql = (
                "SELECT memory_id FROM memory_embeddings WHERE type = 'knowledge' AND ("
                + " OR ".join(clauses)
                + ")"
            )
            cur.execute(sql, params)
            ids = [str(r[0] or "") for r in cur.fetchall() if r and r[0]]
            cur.close()
            if not ids:
                return 0
            await self.delete_by_ids(ids)
            return len(ids)
        except Exception as e:
            logger.error(f"[mem_chat] Failed to delete knowledge by patterns: {e}")
            return 0

    async def get_all_embeddings_by_type(
        self, type_: str, limit: int = 50
    ) -> list[str]:
        try:
            rows = await self._read_recall_rows(
                "SELECT content FROM memory_embeddings WHERE type = ? ORDER BY created_at DESC LIMIT ?",
                (type_, limit),
            )
            return [r[0] for r in rows if r[0]]
        except Exception as e:
            logger.error(f"[mem_chat] Failed to get embeddings by type {type_}: {e}")
            return []
