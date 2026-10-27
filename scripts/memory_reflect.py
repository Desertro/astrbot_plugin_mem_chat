"""Generate reflections from recent memories (Generative Agents style).

Consolidation compresses many raw memories into one shorter summary. Reflection
is different: it reads a window of memories and *induces* statements that none
of them contain on their own, e.g. "she deflects with jokes whenever someone
asks her for tech support". Those conclusions are stored as ``type=reflection``
rows with a high importance weight plus pointers to their evidence, so normal
retrieval can surface them like any other memory.

Two LLM passes per window, following the paper:
  1. ask which high-level questions the window raises;
  2. answer each question against that same window.

Runs offline (cron). Nothing here touches the live reply path.
"""

import argparse
import asyncio
import os
import sqlite3
import sys
import time
import uuid

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_consolidate import (  # noqa: E402
    _call_llm_google,
    _call_llm_openai,
    _get_embedding,
    _load_json,
    _normalize_text,
    _serialize_embedding,
)

REFLECTION_TYPE = "reflection"


async def _ask_llm(session: aiohttp.ClientSession, cfg: dict, prompt: str) -> str:
    """Send one prompt through whichever LLM format the plugin is configured for."""
    fmt = str(cfg.get("llm_format") or cfg.get("custom_llm_format") or "openai").lower()
    base_url = str(
        cfg.get("llm_base_url") or cfg.get("custom_llm_base_url") or ""
    ).strip()
    api_key = str(cfg.get("llm_api_key") or cfg.get("custom_llm_api_key") or "").strip()
    model = str(cfg.get("llm_model") or cfg.get("custom_llm_model") or "").strip()
    temperature = cfg.get("llm_temperature")
    if fmt == "google":
        return await _call_llm_google(
            session, base_url, api_key, model, prompt, temperature
        )
    return await _call_llm_openai(
        session, base_url, api_key, model, prompt, temperature
    )


def _parse_lines(text: str, limit: int) -> list[str]:
    """Pull up to ``limit`` clean one-per-line items out of an LLM reply."""
    out: list[str] = []
    for raw in str(text or "").splitlines():
        line = _normalize_text(raw)
        if not line:
            continue
        for prefix in ("- ", "* ", "\u2022 "):
            if line.startswith(prefix):
                line = line[len(prefix) :]
                break
        head, dot, tail = line.partition(".")
        if dot and head.isdigit() and tail.strip():
            line = tail.strip()
        head, paren, tail = line.partition("\uff09")
        if paren and head.strip("\uff08").isdigit() and tail.strip():
            line = tail.strip()
        line = line.strip("\u3010\u3011").strip()
        if len(line) < 4:
            continue
        out.append(line)
        if len(out) >= limit:
            break
    return out


QUESTION_PROMPT = (
    "以下是你最近在群里观察到的片段：\n"
    "{window}\n\n"
    "根据这些片段，提出 {n} 个最值得追问的高层问题。要求：问题要能归纳出"
    "规律、偏好或人际关系，不要问单条消息的细节，也不要复述原文。\n"
    "每行一个问题，不要输出额外解释。"
)

INSIGHT_PROMPT = (
    "片段：\n{window}\n\n"
    "问题：{question}\n\n"
    "请给出最多 {n} 条结论。要求：\n"
    "- 每条一句话，不超过 60 个中文字\n"
    "- 必须是从片段里归纳出来的，不是复述某一条\n"
    "- 不要编造片段里没有的事\n"
    "- 没有值得归纳的就什么都不输出\n"
    "每行一条结论。"
)


def _fetch_window(
    cur: sqlite3.Cursor, group_id: str, model: str, limit: int, since_ts: float
) -> list[tuple[str, str]]:
    """Read the newest raw memories for one group, oldest-first.

    Args:
        cur: Open cursor.
        group_id: Group bucket to read.
        model: Embedding model name the rows were written under.
        limit: Maximum number of memories in the window.
        since_ts: Only consider memories created at or after this timestamp.

    Returns:
        ``(memory_id, content)`` pairs in chronological order.
    """
    cur.execute(
        "SELECT memory_id, content FROM memory_embeddings "
        "WHERE group_id = ? AND model_name = ? AND type = ? "
        "AND created_at >= ? AND content IS NOT NULL AND length(content) > 0 "
        "ORDER BY created_at DESC LIMIT ?",
        (group_id, model, "memory", float(since_ts), int(limit)),
    )
    rows = [(str(r[0]), _normalize_text(r[1])) for r in cur.fetchall()]
    rows = [r for r in rows if r[1]]
    rows.reverse()
    return rows


def _already_stored(cur: sqlite3.Cursor, group_id: str, model: str, text: str) -> bool:
    """Return True when an identical reflection already exists for the group."""
    cur.execute(
        "SELECT 1 FROM memory_embeddings "
        "WHERE group_id = ? AND model_name = ? AND type = ? AND content = ? LIMIT 1",
        (group_id, model, REFLECTION_TYPE, text),
    )
    return cur.fetchone() is not None


def _store_reflection(
    conn: sqlite3.Connection,
    group_id: str,
    model: str,
    text: str,
    emb: list[float],
    importance: float,
    evidence: list[str],
) -> str:
    """Insert one reflection row and return its memory id.

    The row carries no ``faiss_id``; the plugin assigns one and rebuilds the
    bucket from the stored embedding blob on first use, which is the same path
    the consolidation script relies on.
    """
    mid = "reflect-" + uuid.uuid4().hex[:16]
    now = time.time()
    note = text
    if evidence:
        note = text + "  [依据:" + ",".join(evidence[:5]) + "]"
    conn.execute(
        "INSERT OR REPLACE INTO memory_embeddings("
        "memory_id, content, embedding, vector_dimension, group_id, model_name,"
        " type, created_at, last_updated, importance) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (
            mid,
            note,
            _serialize_embedding(emb),
            len(emb),
            group_id,
            model,
            REFLECTION_TYPE,
            now,
            now,
            float(importance),
        ),
    )
    return mid


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Induce reflections from recent memories."
    )
    parser.add_argument("--db", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--group", default=None, help="Limit to one group id.")
    parser.add_argument(
        "--window", type=int, default=100, help="Memories per reflection window."
    )
    parser.add_argument(
        "--questions", type=int, default=3, help="Questions asked per window."
    )
    parser.add_argument(
        "--insights", type=int, default=2, help="Conclusions kept per question."
    )
    parser.add_argument(
        "--days", type=float, default=7.0, help="Only read memories this recent."
    )
    parser.add_argument(
        "--importance", type=float, default=0.85,
        help="Importance stored on each reflection.",
    )
    parser.add_argument("--max-window-chars", type=int, default=12000)
    parser.add_argument("--min-window", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plugin_dir = os.path.dirname(script_dir)
    db_path = args.db or os.path.join(plugin_dir, "mem_chat_embeddings.db")
    cfg_path = args.config or os.path.join(
        os.getcwd(), "data", "config", "astrbot_plugin_mem_chat_config.json"
    )
    cfg = _load_json(cfg_path)

    emb_base = str(cfg.get("embedding_openai_base_url") or "").strip()
    emb_key = str(cfg.get("embedding_openai_api_key") or "").strip()
    emb_model = str(
        cfg.get("embedding_openai_model") or "text-embedding-3-small"
    ).strip()
    if not emb_base or not emb_key:
        raise RuntimeError(
            "embedding_openai_base_url / embedding_openai_api_key not configured"
        )

    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")
    cur = conn.cursor()

    if args.group:
        groups = [str(args.group)]
    else:
        cur.execute(
            "SELECT DISTINCT group_id FROM memory_embeddings "
            "WHERE model_name = ? AND type = ?",
            (emb_model, "memory"),
        )
        groups = [str(r[0]) for r in cur.fetchall() if r and r[0] is not None]

    since_ts = time.time() - max(0.0, float(args.days)) * 86400.0
    total_written = 0

    async with aiohttp.ClientSession() as session:
        for gid in groups:
            window = _fetch_window(cur, gid, emb_model, args.window, since_ts)
            if len(window) < args.min_window:
                print(f"[skip] {gid}: only {len(window)} memories in window")
                continue

            ids = [mid for mid, _ in window]
            text = "\n".join(f"- {c}" for _, c in window)
            if args.max_window_chars > 0 and len(text) > args.max_window_chars:
                text = text[-args.max_window_chars :]

            raw_q = await _ask_llm(
                session, cfg,
                QUESTION_PROMPT.format(window=text, n=int(args.questions)),
            )
            questions = _parse_lines(raw_q, int(args.questions))
            if not questions:
                print(f"[skip] {gid}: no questions returned")
                continue

            for q in questions:
                raw_i = await _ask_llm(
                    session, cfg,
                    INSIGHT_PROMPT.format(
                        window=text, question=q, n=int(args.insights)
                    ),
                )
                for insight in _parse_lines(raw_i, int(args.insights)):
                    if _already_stored(cur, gid, emb_model, insight):
                        print(f"[dup] {gid}: {insight[:40]}")
                        continue
                    if args.dry_run:
                        print(f"[dry] {gid}: {insight}")
                        continue
                    emb = await _get_embedding(
                        session, emb_base, emb_key, emb_model, insight
                    )
                    if not emb:
                        print(f"[warn] {gid}: embedding failed, skipped")
                        continue
                    _store_reflection(
                        conn, gid, emb_model, insight, emb,
                        float(args.importance), ids,
                    )
                    conn.commit()
                    total_written += 1
                    print(f"[ok] {gid}: {insight}")

    conn.close()
    print(f"done: {total_written} reflection(s) written")


if __name__ == "__main__":
    asyncio.run(main())

