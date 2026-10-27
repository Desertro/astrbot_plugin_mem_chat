import argparse
import asyncio
import json
import os
import sqlite3
import time
from typing import Any

import aiohttp


def _load_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        with open(path, encoding="utf-8") as f:
            return json.load(f)


def _serialize_embedding(emb: list[float]) -> bytes:
    return json.dumps(emb).encode("utf-8")


async def _get_embedding(
    session: aiohttp.ClientSession,
    base: str,
    api_key: str,
    model: str,
    text: str,
) -> list[float]:
    base = (base or "").rstrip("/")
    if not base or not api_key:
        return []
    if not base.endswith("/v1"):
        base = base + "/v1"
    url = base + "/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": text}
    async with session.post(
        url,
        headers=headers,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        if resp.status != 200:
            return []
        data = await resp.json()
        arr = data.get("data") or []
        if not arr:
            return []
        emb = (arr[0] or {}).get("embedding") or []
        if isinstance(emb, list) and emb:
            return [float(x) for x in emb]
        return []


def _ensure_tables(conn: sqlite3.Connection):
    cur = conn.cursor()
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
    conn.commit()
    cur.close()


def _upsert_user_profile(
    conn: sqlite3.Connection,
    group_id: str,
    user_id: str,
    display_name: str,
    profile_json: str,
    canonical_text: str,
    now: float,
    version: int,
):
    cur = conn.cursor()
    cur.execute(
        "SELECT created_at, msg_count FROM user_profiles WHERE group_id = ? AND user_id = ?",
        (group_id, user_id),
    )
    row = cur.fetchone()
    created_at = float(row[0] or now) if row else now
    msg_count = int(row[1] or 0) if row else 0
    cur.execute(
        "INSERT OR REPLACE INTO user_profiles(group_id, user_id, display_name, profile_json, canonical_text, msg_count, version, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
        (
            group_id,
            user_id,
            display_name or "User",
            profile_json or "",
            canonical_text or "",
            msg_count,
            int(version),
            created_at,
            now,
        ),
    )
    conn.commit()
    cur.close()


def _upsert_embedding_row(
    conn: sqlite3.Connection,
    memory_id: str,
    content: str,
    emb: list[float],
    group_id: str,
    model_name: str,
    type_: str,
    now: float,
):
    cur = conn.cursor()
    cur.execute(
        "SELECT faiss_id FROM memory_embeddings WHERE memory_id = ?",
        (memory_id,),
    )
    row = cur.fetchone()
    faiss_id = int(row[0]) if row and row[0] is not None else None
    if faiss_id is None:
        cur.execute("SELECT MAX(faiss_id) FROM memory_embeddings")
        mx = cur.fetchone()
        faiss_id = int(mx[0] or 0) + 1
    emb_blob = _serialize_embedding(emb) if emb else None
    emb_dim = len(emb) if emb else None
    cur.execute(
        "INSERT OR REPLACE INTO memory_embeddings(memory_id, content, embedding, vector_dimension, faiss_id, group_id, model_name, type, created_at, last_updated) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (
            memory_id,
            content,
            emb_blob,
            emb_dim,
            faiss_id,
            group_id,
            model_name,
            type_,
            now,
            now,
        ),
    )
    conn.commit()
    cur.close()


def _normalize_summary(s: str) -> str:
    return " ".join(str(s or "").replace("\n", " ").replace("\r", " ").split()).strip()


def _build_canonical(profile: dict[str, Any]) -> str:
    parts: list[str] = []
    summary = str(profile.get("impression_summary") or "").strip()
    if summary:
        parts.append("印象：" + summary)
    tags = profile.get("tags") or []
    if isinstance(tags, list):
        t = [str(x).strip() for x in tags if str(x).strip()]
        if t:
            parts.append("标签：" + "、".join(t[:8]))
    for k, title in [
        ("traits", "特征"),
        ("preferences", "偏好"),
        ("dislikes", "不喜欢"),
        ("current_topics", "近期话题"),
        ("boundaries", "边界"),
        ("uncertain", "不确定"),
    ]:
        v = profile.get(k) or []
        if isinstance(v, list):
            vv = [str(x).strip() for x in v if str(x).strip()]
            if vv:
                parts.append(title + "：" + "；".join(vv[:5]))
    txt = "。".join([p for p in parts if p]).strip()
    return txt[:600]


def _legacy_profile_json(summary: str) -> str:
    s = _normalize_summary(summary)
    payload: dict[str, Any] = {
        "impression_summary": s[:80] if s else "",
        "tags": [],
        "legacy_summary": s,
        "traits": [],
        "preferences": [],
        "dislikes": [],
        "current_topics": [],
        "boundaries": [],
        "uncertain": [],
    }
    return json.dumps(payload, ensure_ascii=False)


def _derive_summary_from_text(text: str) -> str:
    raw = _normalize_summary(text)
    if not raw:
        return ""
    for sep in ["。", "！", "？", ".", "!", "?"]:
        if sep in raw:
            head = raw.split(sep, 1)[0].strip()
            if head:
                return head[:80]
    return raw[:80]


def _upgrade_profile_to_v3(profile_json: str, canonical_text: str) -> tuple[dict[str, Any], str]:
    data: dict[str, Any] = {}
    try:
        obj = json.loads(str(profile_json or "").strip())
        if isinstance(obj, dict):
            data = obj
    except Exception:
        data = {}

    summary = str(data.get("impression_summary") or "").strip()
    if not summary:
        legacy = str(data.get("legacy_summary") or "").strip()
        summary = _derive_summary_from_text(legacy) if legacy else ""
    if not summary:
        summary = _derive_summary_from_text(canonical_text)

    tags = data.get("tags")
    if not isinstance(tags, list):
        tags = []
    tags = [str(x).strip() for x in tags if str(x).strip()][:8]

    def _lst(key: str, limit: int) -> list[str]:
        v = data.get(key)
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()][:limit]
        if isinstance(v, str) and v.strip():
            return [v.strip()][:limit]
        return []

    v3: dict[str, Any] = {
        "impression_summary": _normalize_summary(summary)[:80] if summary else "",
        "tags": tags,
        "legacy_summary": str(data.get("legacy_summary") or "").strip(),
        "traits": _lst("traits", 5),
        "preferences": _lst("preferences", 5),
        "dislikes": _lst("dislikes", 5),
        "current_topics": _lst("current_topics", 5),
        "boundaries": _lst("boundaries", 5),
        "uncertain": _lst("uncertain", 5),
    }
    canon = _build_canonical(v3)
    return v3, canon


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reset-faiss", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plugin_dir = os.path.dirname(script_dir)
    default_db = os.path.join(plugin_dir, "mem_chat_embeddings.db")
    db_path = args.db or default_db

    default_cfg = os.path.join(
        os.getcwd(), "data", "config", "astrbot_plugin_mem_chat_config.json"
    )
    cfg_path = args.config or default_cfg
    cfg = _load_json(cfg_path)

    emb_base = str(cfg.get("embedding_openai_base_url") or "").strip()
    emb_key = str(cfg.get("embedding_openai_api_key") or "").strip()
    emb_model = str(cfg.get("embedding_openai_model") or "text-embedding-3-small").strip()

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    _ensure_tables(conn)

    if args.reset_faiss:
        faiss_dir = os.path.join(plugin_dir, "faiss")
        if os.path.isdir(faiss_dir):
            for name in os.listdir(faiss_dir):
                if name.endswith(".faiss"):
                    try:
                        os.remove(os.path.join(faiss_dir, name))
                    except Exception:
                        pass

    cur = conn.cursor()
    where = ""
    params: list[Any] = []
    if args.group:
        where = "WHERE group_id = ?"
        params.append(str(args.group))
    cur.execute(
        f"SELECT group_id, user_id, name, summary FROM impressions {where}",
        params,
    )
    raw_rows = [
        (str(r[0] or ""), str(r[1] or ""), str(r[2] or "User"), str(r[3] or ""))
        for r in cur.fetchall()
    ]
    cur.close()

    total_impressions = len(raw_rows)
    best_by_user: dict[tuple[str, str], tuple[str, str]] = {}
    duplicates = 0
    invalid = 0
    for gid, uid, name, summary in raw_rows:
        if not gid or not uid:
            invalid += 1
            continue
        s = _normalize_summary(summary)
        if not s:
            invalid += 1
            continue
        key = (gid, uid)
        prev = best_by_user.get(key)
        if prev is not None:
            duplicates += 1
            prev_name, prev_sum = prev
            if len(s) > len(prev_sum):
                best_by_user[key] = (name, s)
            elif len(s) == len(prev_sum) and len(str(name or "")) > len(str(prev_name or "")):
                best_by_user[key] = (name, s)
        else:
            best_by_user[key] = (name, s)

    rows = [(gid, uid, name, summary) for (gid, uid), (name, summary) in best_by_user.items()]
    total_users = len(rows)
    migrated = 0
    skipped = 0
    upgraded_profiles = 0
    updated_impr2_rows = 0

    async with aiohttp.ClientSession() as session:
        for gid, uid, name, summary in rows:
            if not gid or not uid:
                skipped += 1
                continue
            summary_norm = str(summary or "").strip()
            if not summary_norm:
                skipped += 1
                continue

            now = time.time()
            memory_id = f"impr2::{gid}::{uid}"
            prof_json = _legacy_profile_json(summary_norm)
            prof_obj, canon = _upgrade_profile_to_v3(prof_json, summary_norm)
            prof_json = json.dumps(prof_obj, ensure_ascii=False)
            impr_content = f"IMPR2 gid={gid} uid={uid} name={name} :: {canon}"

            if not args.force:
                c = conn.cursor()
                c.execute(
                    "SELECT 1 FROM user_profiles WHERE group_id = ? AND user_id = ?",
                    (gid, uid),
                )
                exists = bool(c.fetchone())
                c.close()
                if exists:
                    skipped += 1
                    continue

            if args.dry_run:
                migrated += 1
                continue

            _upsert_user_profile(
                conn,
                gid,
                uid,
                name,
                prof_json,
                canon,
                now,
                version=3,
            )
            if not args.skip_embedding:
                if not emb_base or not emb_key:
                    raise RuntimeError("embedding_openai_base_url 或 embedding_openai_api_key 未配置")
                emb = await _get_embedding(session, emb_base, emb_key, emb_model, impr_content)
                if emb:
                    _upsert_embedding_row(
                        conn,
                        memory_id,
                        impr_content,
                        emb,
                        gid,
                        emb_model,
                        "impression",
                        now,
                    )
                else:
                    _upsert_embedding_row(
                        conn,
                        memory_id,
                        impr_content,
                        [],
                        gid,
                        emb_model,
                        "impression",
                        now,
                    )
            else:
                _upsert_embedding_row(
                    conn,
                    memory_id,
                    impr_content,
                    [],
                    gid,
                    emb_model,
                    "impression",
                    now,
                )
            migrated += 1

        cur2 = conn.cursor()
        where2 = "WHERE COALESCE(version, 0) < 3"
        params2: list[Any] = []
        if args.group:
            where2 += " AND group_id = ?"
            params2.append(str(args.group))
        cur2.execute(
            f"SELECT group_id, user_id, display_name, profile_json, canonical_text, version FROM user_profiles {where2}",
            params2,
        )
        prof_rows = [
            (
                str(r[0] or ""),
                str(r[1] or ""),
                str(r[2] or "User"),
                str(r[3] or ""),
                str(r[4] or ""),
                int(r[5] or 0),
            )
            for r in cur2.fetchall()
        ]
        cur2.close()

        for gid, uid, name, pjson, canon_old, ver in prof_rows:
            if not gid or not uid:
                continue
            now = time.time()
            obj, canon_new = _upgrade_profile_to_v3(pjson, canon_old)
            pjson_new = json.dumps(obj, ensure_ascii=False)
            if args.dry_run:
                upgraded_profiles += 1
                continue
            _upsert_user_profile(
                conn,
                gid,
                uid,
                name,
                pjson_new,
                canon_new,
                now,
                version=3,
            )
            memory_id = f"impr2::{gid}::{uid}"
            impr_content = f"IMPR2 gid={gid} uid={uid} name={name} :: {canon_new}"
            if not args.skip_embedding and emb_base and emb_key:
                emb = await _get_embedding(session, emb_base, emb_key, emb_model, impr_content)
                _upsert_embedding_row(
                    conn,
                    memory_id,
                    impr_content,
                    emb,
                    gid,
                    emb_model,
                    "impression",
                    now,
                )
            else:
                _upsert_embedding_row(
                    conn,
                    memory_id,
                    impr_content,
                    [],
                    gid,
                    emb_model,
                    "impression",
                    now,
                )
            upgraded_profiles += 1
            updated_impr2_rows += 1

    conn.close()
    print(
        json.dumps(
            {
                "total_impressions": total_impressions,
                "total_users_deduped": total_users,
                "invalid_rows": invalid,
                "duplicates_collapsed": duplicates,
                "migrated": migrated,
                "skipped": skipped,
                "upgraded_profiles": upgraded_profiles,
                "updated_impr2_rows": updated_impr2_rows,
                "dry_run": bool(args.dry_run),
                "skip_embedding": bool(args.skip_embedding),
                "forced": bool(args.force),
                "reset_faiss": bool(args.reset_faiss),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
