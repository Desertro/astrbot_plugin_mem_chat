"""Exact sticker references and recoverable removal from the active library."""

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class MemeLibrary:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.root = (data_dir / "memes").resolve()
        self.db_path = data_dir / "meme_index.db"

    @contextmanager
    def _connect(self):
        """Open the shared index and initialize persistent conversation references.

        Yields:
            A connection whose rows expose column names.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.db_path, timeout=5)
        con.row_factory = sqlite3.Row
        con.execute(
            "CREATE TABLE IF NOT EXISTS meme_references ("
            "scope TEXT, meme_id TEXT, path TEXT, sent_hash TEXT, sent_at REAL, "
            "PRIMARY KEY(scope, meme_id))"
        )
        try:
            with con:
                yield con
        finally:
            con.close()

    def remember(self, scope: str, path: str, sent_file: str = "") -> str:
        """Remember an exact inspected or successfully sent library sticker.

        Args:
            scope: Current conversation origin.
            path: Original library file, never a temporary resized file.
            sent_file: Actual delivered file; empty for an inspected reference.

        Returns:
            Opaque reference ID, or an empty string for an invalid target.
        """
        target = Path(path).resolve()
        if not scope or not target.is_relative_to(self.root) or not target.is_file():
            return ""
        meme_id = hashlib.sha256(str(target).encode()).hexdigest()[:20]
        sent_hash = ""
        if sent_file:
            sent_hash = hashlib.sha256(Path(sent_file).read_bytes()).hexdigest()
        with self._connect() as con:
            con.execute(
                "INSERT INTO meme_references VALUES(?,?,?,?,?) "
                "ON CONFLICT(scope,meme_id) DO UPDATE SET "
                "sent_hash=CASE WHEN excluded.sent_at>0 THEN excluded.sent_hash "
                "ELSE meme_references.sent_hash END, "
                "sent_at=MAX(meme_references.sent_at,excluded.sent_at)",
                (
                    scope,
                    meme_id,
                    str(target),
                    sent_hash,
                    time.time() if sent_file else 0,
                ),
            )
        return meme_id

    def recent(self, scope: str) -> list[dict]:
        """List actual sent stickers from this conversation, newest first.

        Args:
            scope: Current conversation origin.

        Returns:
            IDs, timestamps and indexed descriptions for existing stickers.
        """
        result = []
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM meme_references WHERE scope=? AND sent_at>0 "
                "ORDER BY sent_at DESC LIMIT 12",
                (scope,),
            ).fetchall()
            for row in rows:
                target = Path(row["path"]).resolve()
                if not target.is_relative_to(self.root) or not target.is_file():
                    continue
                description = ""
                if con.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meme_index'"
                ).fetchone():
                    item = con.execute(
                        "SELECT description FROM meme_index WHERE path=?",
                        (row["path"],),
                    ).fetchone()
                    description = item[0] if item else ""
                result.append(
                    {
                        "meme_id": row["meme_id"],
                        "sent_at": time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(row["sent_at"])
                        ),
                        "description": str(description or "")[:500],
                    }
                )
        return result

    def match(self, scope: str, raw: bytes) -> list[str]:
        """Match exact bytes, capture fingerprints, or a delivered resized version.

        Args:
            scope: Current conversation origin.
            raw: Image bytes obtained from the selected chat message.

        Returns:
            IDs for all exact library matches. Multiple matches remain explicit.
        """
        digest = hashlib.sha256(raw).hexdigest()
        targets = set()
        with self._connect() as con:
            for row in con.execute(
                "SELECT path FROM meme_references WHERE scope=? AND sent_hash=?",
                (scope, digest),
            ):
                targets.add(row[0])
        for target in self.root.glob("*/*"):
            resolved = target.resolve()
            if not resolved.is_relative_to(self.root) or not resolved.is_file():
                continue
            try:
                # Captures are named after the source bytes before JPEG conversion.
                if (
                    resolved.stem == digest[:32]
                    or hashlib.sha256(resolved.read_bytes()).hexdigest() == digest
                ):
                    targets.add(str(resolved))
            except OSError:
                # Another conversation may delete a file while lookup is running.
                continue
        return sorted({ref for path in targets if (ref := self.remember(scope, path))})

    def delete(self, scope: str, meme_id: str) -> dict:
        """Archive one exact sticker and remove its active retrieval records.

        Args:
            scope: Conversation that previously inspected or sent the sticker.
            meme_id: Opaque ID returned by recent or inspect.

        Returns:
            Actual deletion outcome. A failed transaction restores the file.
        """
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT path FROM meme_references WHERE scope=? AND meme_id=?",
                (scope, meme_id),
            ).fetchone()
            if not row:
                return {
                    "ok": False,
                    "reason": "没有这个表情的准确记录，请先 recent 或 inspect 定位。",
                }
            target = Path(row[0]).resolve()
            if not target.is_relative_to(self.root) or target == self.root:
                return {"ok": False, "reason": "目标不在表情图库内，未删除。"}
            if not target.is_file():
                return {"ok": False, "reason": "表情文件已不存在，未执行删除。"}
            archive_dir = self.data_dir / "deleted" / uuid.uuid4().hex
            archive = archive_dir / target.relative_to(self.root)
            archive.parent.mkdir(parents=True)
            metadata = {
                "original_path": str(target),
                "deleted_at": time.time(),
                "records": {},
            }
            # Keep index rows, including embeddings, with the recoverable file.
            for table in ("meme_index", "meme_tags"):
                if con.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone():
                    records = con.execute(
                        f"SELECT * FROM {table} WHERE path=?", (str(target),)
                    ).fetchall()
                    metadata["records"][table] = [dict(record) for record in records]
            (archive_dir / "metadata.json").write_text(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    default=lambda value: {"hex": value.hex()},
                ),
                encoding="utf-8",
            )
            moved = False
            try:
                target.replace(archive)
                moved = True
                for table in metadata["records"]:
                    con.execute(f"DELETE FROM {table} WHERE path=?", (str(target),))
                con.execute("DELETE FROM meme_references WHERE path=?", (str(target),))
                con.commit()
            except Exception:
                con.rollback()
                if moved:
                    archive.replace(target)
                raise
        return {
            "ok": True,
            "meme_id": meme_id,
            "result": "已从共享表情图库删除，以后各聊天都不会再选到这张。",
        }
