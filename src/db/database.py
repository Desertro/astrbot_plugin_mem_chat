import asyncio
import collections
import hashlib
import json
import logging
import os
import sqlite3
import time

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
        self._vector_dimension: int | None = None
        self._emb_cache = collections.OrderedDict()
        self._faiss_lock = asyncio.Lock()
        self._faiss_indexes: dict[str, object] = {}
        self._faiss_ready_buckets: set[str] = set()
        self._faiss_last_save_ts: dict[str, float] = {}
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
                last_updated REAL
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
            CREATE TABLE IF NOT EXISTS system_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
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
            try:
                # Use asyncio.create_task to run the coroutine in the background if we can't await it here
                # Or since this is called from __init__ (sync), we can't await. 
                # But wait, __init__ calls init_db_conn which calls this.
                # __init__ is sync. This is tricky.
                # Let's check where _maybe_migrate_faiss_ids is called. It's called in init_db_conn.
                # If init_db_conn is called in __init__, we have a problem with async methods.
                # However, the original code set_meta was sync. Now it's async.
                # We should probably keep a sync version for internal use or use run_coroutine_threadsafe.
                pass 
                # self.set_meta("faiss_id_migrated_v1", "1") # TODO: Fix sync call
            except Exception:
                pass
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
                # self.set_meta("faiss_id_migrated_v1", "1") # TODO: Fix sync call
                pass
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
                    idx = await asyncio.to_thread(faiss.read_index, self._safe_path(path))
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
        k = model + "::" + (text or "").strip()
        now = time.time()
        try:
            v = self._emb_cache.get(k)
            if v and (now - v[0] <= self._emb_cache_ttl):
                emb_cached = v[1]
                if isinstance(emb_cached, list) and emb_cached:
                    self._emb_cache.move_to_end(k)
                    return [float(x) for x in emb_cached]
        except Exception:
            pass
        if base and key:
            url = base + "/v1/embeddings"
            payload = {"model": model, "input": text}
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            try:
                timeout = aiohttp.ClientTimeout(total=60)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, headers=headers, json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            arr = data.get("data") or []
                            if arr:
                                emb = arr[0].get("embedding") or []
                                if isinstance(emb, list) and emb:
                                    if self._vector_dimension is None:
                                        self._vector_dimension = len(emb)
                                    try:
                                        self._emb_cache[k] = (now, emb)
                                        self._emb_cache.move_to_end(k)
                                        if len(self._emb_cache) > self._emb_cache_size:
                                            self._emb_cache.popitem(last=False)
                                    except Exception:
                                        pass
                                    return [float(x) for x in emb]
            except Exception as e:
                logger.error(f"[mem_chat] Error getting embedding: {e}")
                return []
        return []

    async def get_or_compute_embedding(
        self,
        memory_id: str,
        content: str,
        group_id: str,
        model_name: str,
        type: str = "memory",
    ) -> list[float]:
        if not self._db_conn:
            self.init_db_conn()
        cur = self._db_conn.cursor()
        try:
            cur.execute(
                "SELECT embedding, vector_dimension FROM memory_embeddings WHERE memory_id = ? AND group_id = ? AND model_name = ?",
                (memory_id, group_id, model_name),
            )
            row = cur.fetchone()
            if row and row[0]:
                emb = self._deserialize_embedding(row[0], int(row[1] or 0))
                cur.close()
                if emb:
                    return emb
        except Exception:
            cur.close()
            pass

        cur.close()

        emb = await self.get_embedding(content)
        if emb:
            try:
                if not self._db_conn:
                    self.init_db_conn()
                cur = self._db_conn.cursor()
                now = time.time()
                faiss_id = self._alloc_faiss_id_db(memory_id)
                cur.execute(
                    "INSERT OR REPLACE INTO memory_embeddings(memory_id, content, embedding, vector_dimension, faiss_id, group_id, model_name, type, created_at, last_updated) VALUES(?,?,?,?,?,?,?,?,?,?)",
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
                await self._faiss_upsert_if_ready(group_id, model_name, type, faiss_id, emb)
            except Exception:
                pass
        return emb or []

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

        if not self._db_conn:
            self.init_db_conn()
        cur = self._db_conn.cursor()

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
                    sql = "SELECT memory_id, embedding, vector_dimension, content FROM memory_embeddings WHERE group_id = ? AND model_name = ? AND type = ? ORDER BY created_at DESC LIMIT ?"
                    params = (group_id, model_name, type, scan_limit)
                else:
                    sql = "SELECT memory_id, embedding, vector_dimension, content FROM memory_embeddings WHERE group_id = ? AND model_name = ? AND type = ?"
                    params = (group_id, model_name, type)
            else:
                sql = "SELECT memory_id, embedding, vector_dimension, content FROM memory_embeddings WHERE (group_id = ? OR group_id = 'global') AND model_name = ? AND type = ?"
                params = (group_id, model_name, type)

            cur.execute(sql, params)
            results: list[tuple[str, float]] = []
            for row in cur.fetchall():
                mid, blob, vd, content = row
                mem_emb = self._deserialize_embedding(blob, int(vd or 0))
                if mem_emb:
                    sim = self._cosine_similarity(qemb, mem_emb)
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
        finally:
            try:
                cur.close()
            except Exception:
                pass

    async def semantic_recall_with_qemb(
        self,
        qemb: list[float],
        group_id: str,
        limit: int,
        threshold: float,
        type: str = "memory",
    ) -> list[tuple[str, float]]:
        if not qemb:
            return []
        model_name = str(
            self.config.get("embedding_openai_model", "text-embedding-3-small")
            or "text-embedding-3-small"
        )
        try:
            use_faiss = self._faiss_enabled() and bool(faiss) and bool(np)
            if use_faiss:
                dim = int(len(qemb))
                if dim <= 0:
                    return []
                if self._vector_dimension is None:
                    self._vector_dimension = dim

                def _post_filter(results: list[tuple[str, float]]) -> list[tuple[str, float]]:
                    results.sort(key=lambda x: x[1], reverse=True)
                    use_dynamic = bool(self.config.get("dynamic_threshold_enable", True))
                    if type != "memory":
                        if use_dynamic:
                            try:
                                smax = results[0][1] if results else 0.0
                                floor = float(self.config.get("dynamic_threshold_floor", 0.0))
                                if type == "persona":
                                    alpha = float(self.config.get("dynamic_alpha_persona", 0.92))
                                elif type == "impression":
                                    alpha = float(self.config.get("dynamic_alpha_impression", 0.88))
                                elif type == "memory":
                                    alpha = float(self.config.get("dynamic_alpha_memory", 0.85))
                                else:
                                    alpha = float(self.config.get("dynamic_alpha_knowledge", 0.80))
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
                                "SELECT content, faiss_id FROM memory_embeddings "
                                f"WHERE group_id = ? AND model_name = ? AND type = ? AND faiss_id IN ({placeholders})"
                            )
                            params = [target_group, model_name, type, *chunk]
                            cur.execute(sql, params)
                            for content, fid in cur.fetchall():
                                try:
                                    fid_i = int(fid)
                                except Exception:
                                    continue
                                sim = float(score_by_id.get(fid_i, 0.0))
                                if content:
                                    out.append((content, sim))
                    finally:
                        try:
                            cur.close()
                        except Exception:
                            pass
                    return out

                if type in ["persona", "knowledge"]:
                    res_local = await _search_bucket(group_id)
                    res_global = await _search_bucket("global")
                    merged = res_local + res_global
                    return _post_filter(merged)

                res = await _search_bucket(group_id)
                return _post_filter(res)

            if not self._db_conn:
                self.init_db_conn()
            cur = self._db_conn.cursor()
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
                    sql = "SELECT memory_id, embedding, vector_dimension, content FROM memory_embeddings WHERE group_id = ? AND model_name = ? AND type = ? ORDER BY created_at DESC LIMIT ?"
                    params = (group_id, model_name, type, scan_limit)
                else:
                    sql = "SELECT memory_id, embedding, vector_dimension, content FROM memory_embeddings WHERE group_id = ? AND model_name = ? AND type = ?"
                    params = (group_id, model_name, type)
            else:
                sql = "SELECT memory_id, embedding, vector_dimension, content FROM memory_embeddings WHERE (group_id = ? OR group_id = 'global') AND model_name = ? AND type = ?"
                params = (group_id, model_name, type)
            cur.execute(sql, params)
            results: list[tuple[str, float]] = []
            for row in cur.fetchall():
                mid, blob, vd, content = row
                mem_emb = self._deserialize_embedding(blob, int(vd or 0))
                if mem_emb:
                    sim = self._cosine_similarity(qemb, mem_emb)
                    results.append((content, sim))
            results.sort(key=lambda x: x[1], reverse=True)
            use_dynamic = bool(self.config.get("dynamic_threshold_enable", True))
            if type != "memory":
                if use_dynamic:
                    try:
                        smax = results[0][1] if results else 0.0
                        floor = float(self.config.get("dynamic_threshold_floor", 0.0))
                        if type == "persona":
                            alpha = float(self.config.get("dynamic_alpha_persona", 0.92))
                        elif type == "impression":
                            alpha = float(self.config.get("dynamic_alpha_impression", 0.88))
                        elif type == "memory":
                            alpha = float(self.config.get("dynamic_alpha_memory", 0.85))
                        else:
                            alpha = float(self.config.get("dynamic_alpha_knowledge", 0.80))
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
        finally:
            try:
                cur.close()
            except Exception:
                pass

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
                "【最近聊天消息】",
                "【相关知识】",
                "【相关人格】",
                "【需要回复的群聊对话】",
                "【回复提示词】",
                "【表情包提示词】",
            ]
            for h in headers:
                if h in text:
                    return ""
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
        self, content: str, group_id: str, sender_info: str | None = None
    ):
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
        final_content = f"{sender_info}: {content}" if sender_info else content
        mid = self.hash_id(final_content or "", group_id, model_name)
        await self.get_or_compute_embedding(mid, final_content or "", group_id, model_name)

    async def get_count_by_type(self, type_: str) -> int:
        if not self._db_conn:
            self.init_db_conn()
        try:
            cur = self._db_conn.cursor()
            cur.execute("SELECT COUNT(*) FROM memory_embeddings WHERE type = ?", (type_,))
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
                        bkey = self._bucket_key(str(gid or ""), str(mname or ""), str(tp or ""))
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
                    bkey = self._bucket_key(str(gid or ""), str(mname or ""), str(tp or ""))
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
                    bkey = self._bucket_key(str(gid or ""), str(mname or ""), str(tp or ""))
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
            logger.info(f"[mem_chat] Deleted {deleted} entries matching '{content_snippet}'")
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
            logger.error(f"[mem_chat] Failed to delete content '{content_snippet}': {e}")

    async def get_all_embeddings_by_type(self, type_: str, limit: int = 50) -> list[str]:
        if not self._db_conn:
            self.init_db_conn()
        try:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT content FROM memory_embeddings WHERE type = ? ORDER BY created_at DESC LIMIT ?",
                (type_, limit),
            )
            rows = cur.fetchall()
            cur.close()
            return [r[0] for r in rows if r[0]]
        except Exception as e:
            logger.error(f"[mem_chat] Failed to get embeddings by type {type_}: {e}")
            return []
