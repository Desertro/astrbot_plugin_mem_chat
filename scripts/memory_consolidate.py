import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import time
import uuid

import aiohttp


def _sha1(text: str) -> str:
    h = hashlib.sha1()
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def _load_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        with open(path, encoding="utf-8") as f:
            return json.load(f)


def _serialize_embedding(emb: list[float]) -> bytes:
    return json.dumps(emb).encode("utf-8")


async def _get_embedding(session: aiohttp.ClientSession, base: str, api_key: str, model: str, text: str) -> list[float]:
    base = (base or "").rstrip("/")
    if not base or not api_key:
        return []
    if not base.endswith("/v1"):
        base = base + "/v1"
    url = base + "/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": text}
    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
        if resp.status != 200:
            return []
        data = await resp.json()
        arr = data.get("data") or []
        if not arr:
            return []
        emb = arr[0].get("embedding") or []
        if isinstance(emb, list) and emb:
            return [float(x) for x in emb]
        return []


async def _call_llm_openai(session: aiohttp.ClientSession, base_url: str, api_key: str, model: str, prompt: str, temperature: float | None) -> str:
    base_url = (base_url or "").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    url = base_url + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
        "stream": False,
    }
    if temperature is not None:
        try:
            payload["temperature"] = float(temperature)
        except Exception:
            pass
    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
        if resp.status != 200:
            return ""
        data = await resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        return str(content or "").strip()


async def _call_llm_google(session: aiohttp.ClientSession, base_url: str, api_key: str, model: str, prompt: str, temperature: float | None) -> str:
    base_url = (base_url or "").rstrip("/")
    url = f"{base_url}/v1beta/models/{model}:generateContent" if not str(model).startswith("models/") else f"{base_url}/v1beta/{model}:generateContent"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    if temperature is not None:
        try:
            payload["generationConfig"] = {"temperature": float(temperature)}
        except Exception:
            pass
    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
        if resp.status != 200:
            return ""
        data = await resp.json()
        cands = data.get("candidates") or []
        if not cands:
            return ""
        content = (cands[0] or {}).get("content") or {}
        parts = content.get("parts") or []
        texts: list[str] = []
        for p in parts:
            if isinstance(p, dict) and p.get("text"):
                texts.append(str(p.get("text")))
        return "".join(texts).strip()


async def _summarize_chunk(session: aiohttp.ClientSession, cfg: dict, text: str) -> str:
    fmt = str(cfg.get("llm_format") or cfg.get("custom_llm_format") or "openai").lower()
    base_url = str(cfg.get("llm_base_url") or cfg.get("custom_llm_base_url") or "").strip()
    api_key = str(cfg.get("llm_api_key") or cfg.get("custom_llm_api_key") or "").strip()
    model = str(cfg.get("llm_model") or cfg.get("custom_llm_model") or "").strip()
    temperature = cfg.get("llm_temperature")
    if fmt == "google":
        return await _call_llm_google(session, base_url, api_key, model, text, temperature)
    return await _call_llm_openai(session, base_url, api_key, model, text, temperature)


def _normalize_text(s: str) -> str:
    return " ".join(str(s or "").split()).strip()


def _slice_chunks(rows: list[tuple[str, float, str]], chunk_size: int, keep_recent: int) -> list[list[tuple[str, float, str]]]:
    if chunk_size <= 0:
        return []
    if keep_recent < 0:
        keep_recent = 0
    usable = rows
    if keep_recent > 0 and len(rows) > keep_recent:
        usable = rows[:-keep_recent]
    chunks: list[list[tuple[str, float, str]]] = []
    for i in range(0, len(usable), chunk_size):
        ch = usable[i : i + chunk_size]
        if len(ch) >= chunk_size:
            chunks.append(ch)
    return chunks


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--chunk", type=int, default=200)
    parser.add_argument("--keep-recent", type=int, default=200)
    parser.add_argument("--max-input-chars", type=int, default=12000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plugin_dir = os.path.dirname(script_dir)
    default_db = os.path.join(plugin_dir, "mem_chat_embeddings.db")
    db_path = args.db or default_db

    default_cfg = os.path.join(os.getcwd(), "data", "config", "astrbot_plugin_mem_chat_config.json")
    cfg_path = args.config or default_cfg
    cfg = _load_json(cfg_path)

    emb_base = str(cfg.get("embedding_openai_base_url") or "").strip()
    emb_key = str(cfg.get("embedding_openai_api_key") or "").strip()
    emb_model = str(cfg.get("embedding_openai_model") or "text-embedding-3-small").strip()
    if not emb_base or not emb_key:
        raise RuntimeError("embedding_openai_base_url 或 embedding_openai_api_key 未配置")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    cur = conn.cursor()

    params = [emb_model, "memory"]
    where = "WHERE model_name = ? AND type = ?"
    if args.group:
        where += " AND group_id = ?"
        params.append(str(args.group))

    cur.execute(f"SELECT DISTINCT group_id FROM memory_embeddings {where}", params)
    groups = [str(r[0]) for r in cur.fetchall() if r and r[0] is not None]
    if args.group and args.group not in groups:
        groups = [str(args.group)]

    async with aiohttp.ClientSession() as session:
        for gid in groups:
            cur.execute(
                "SELECT memory_id, created_at, content FROM memory_embeddings WHERE group_id = ? AND model_name = ? AND type = ? ORDER BY created_at ASC",
                (gid, emb_model, "memory"),
            )
            rows = [(str(r[0]), float(r[1] or 0.0), str(r[2] or "")) for r in cur.fetchall()]
            chunks = _slice_chunks(rows, args.chunk, args.keep_recent)
            if not chunks:
                continue

            for ch in chunks:
                contents = []
                for _, _, c in ch:
                    t = _normalize_text(c)
                    if t:
                        contents.append(t)
                raw = "\n".join(contents)
                if not raw:
                    continue
                if args.max_input_chars > 0 and len(raw) > args.max_input_chars:
                    raw = raw[: args.max_input_chars]

                prompt = "请将以下内容概括为不超过50个中文字符的一句话：\n" + raw
                summary = await _summarize_chunk(session, cfg, prompt)
                summary = _normalize_text(summary)
                if not summary:
                    continue
                if len(summary) > 80:
                    summary = summary[:80]

                if args.dry_run:
                    print(f"group={gid} chunk={len(ch)} summary={summary}")
                    continue

                emb = await _get_embedding(session, emb_base, emb_key, emb_model, summary)
                if not emb:
                    continue

                now = time.time()
                last_ts = max((ts for _, ts, _ in ch), default=now)
                mid = _sha1(f"{gid}::{emb_model}::sum::{uuid.uuid4().hex}")
                cur.execute(
                    "INSERT OR REPLACE INTO memory_embeddings(memory_id, content, embedding, vector_dimension, group_id, model_name, type, created_at, last_updated) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        mid,
                        summary,
                        _serialize_embedding(emb),
                        len(emb),
                        gid,
                        emb_model,
                        "memory",
                        float(last_ts),
                        float(now),
                    ),
                )
                ids = [m for m, _, _ in ch]
                placeholders = ",".join(["?"] * len(ids))
                cur.execute(
                    f"UPDATE memory_embeddings SET type = 'memory_raw', last_updated = ? WHERE memory_id IN ({placeholders})",
                    [float(now), *ids],
                )
                conn.commit()
                print(f"group={gid} chunk={len(ch)} summarized=1")

    try:
        cur.close()
    except Exception:
        pass
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
