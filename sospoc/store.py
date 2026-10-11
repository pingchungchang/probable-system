"""SQLite metadata + vector storage.

Two vector backends behind one interface, mirroring the `VectorIndex` split the
real system needs (../../IMPLEMENTATION_PLAN.md §4): sqlite-vec when available,
a numpy brute-force table when it is not. The PoC does not need the extension —
only the real system does — so a missing wheel must not block the experiment
(PLAN.md §8 P1).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import sqlite3
import time
from pathlib import Path

import numpy as np

from . import DIM

SCHEMA = Path(__file__).with_name("schema.sql")
HASH_CHUNK = 1 << 20


def _load_schema() -> tuple[str, str]:
    base, _, vec = SCHEMA.read_text().partition("-- @vector-index")
    return base, vec


class VecIndex:
    """sqlite-vec vec0 backend.

    Vectors are serialised as raw little-endian float32 — sqlite-vec's own blob
    format — so this works against a bare `vec0.so` without the Python package,
    which is what the Nix devShell provides.
    """

    name = "sqlite-vec"

    def __init__(self, conn: sqlite3.Connection, vec_ddl: str):
        self.conn = conn
        conn.executescript(vec_ddl)

    def put(self, file_id: int, v: np.ndarray) -> None:
        blob = v.astype("<f4").tobytes()
        self.conn.execute("DELETE FROM file_vectors WHERE file_id = ?", (file_id,))
        self.conn.execute(
            "INSERT INTO file_vectors(file_id, embedding) VALUES (?, ?)", (file_id, blob)
        )

    def drop(self, file_id: int) -> None:
        self.conn.execute("DELETE FROM file_vectors WHERE file_id = ?", (file_id,))

    def knn(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        blob = q.astype("<f4").tobytes()
        rows = self.conn.execute(
            "SELECT file_id, distance FROM file_vectors "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (blob, k),
        ).fetchall()
        # vec0 returns cosine *distance*; report similarity for readability.
        return [(fid, 1.0 - d) for fid, d in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT count(*) FROM file_vectors").fetchone()[0]


class NumpyIndex:
    """Brute-force fallback. Fine to ~50k vectors; the real system does better."""

    name = "numpy-bruteforce"

    def __init__(self, conn: sqlite3.Connection, _vec_ddl: str):
        self.conn = conn
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS file_vectors_np ("
            "  file_id INTEGER PRIMARY KEY REFERENCES files(file_id) ON DELETE CASCADE,"
            "  embedding BLOB NOT NULL);"
        )
        self._cache: tuple[np.ndarray, np.ndarray] | None = None

    def put(self, file_id: int, v: np.ndarray) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO file_vectors_np(file_id, embedding) VALUES (?, ?)",
            (file_id, v.astype(np.float32).tobytes()),
        )
        self._cache = None

    def drop(self, file_id: int) -> None:
        self.conn.execute("DELETE FROM file_vectors_np WHERE file_id = ?", (file_id,))
        self._cache = None

    def _load(self) -> tuple[np.ndarray, np.ndarray]:
        if self._cache is None:
            rows = self.conn.execute("SELECT file_id, embedding FROM file_vectors_np").fetchall()
            ids = np.array([r[0] for r in rows], dtype=np.int64)
            mat = (
                np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(-1, DIM)
                if rows
                else np.zeros((0, DIM), dtype=np.float32)
            )
            self._cache = (ids, mat)
        return self._cache

    def knn(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        ids, mat = self._load()
        if len(ids) == 0:
            return []
        sims = mat @ q.astype(np.float32)  # both sides L2-normalised
        top = np.argsort(-sims)[:k]
        return [(int(ids[i]), float(sims[i])) for i in top]

    def count(self) -> int:
        return self.conn.execute("SELECT count(*) FROM file_vectors_np").fetchone()[0]


def _try_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Load the extension from $SQLITE_VEC_PATH (Nix) or the pip package.

    Returns False rather than raising: the brute-force fallback is a supported
    path, not a degraded one, at PoC scale (PLAN.md §8 P1).
    """
    if os.environ.get("SOS_FORCE_NUMPY") == "1":
        return False
    try:
        conn.enable_load_extension(True)
    except AttributeError:
        # Python built without --enable-loadable-sqlite-extensions.
        return False

    try:
        explicit = os.environ.get("SQLITE_VEC_PATH")
        if explicit:
            for cand in (explicit, f"{explicit}/vec0", f"{explicit}/lib/vec0"):
                try:
                    conn.load_extension(cand)
                    return True
                except sqlite3.OperationalError:
                    continue
        try:
            import sqlite_vec

            sqlite_vec.load(conn)
            return True
        except (ImportError, sqlite3.OperationalError):
            return False
    finally:
        conn.enable_load_extension(False)


class Store:
    def __init__(self, db_path: str | Path, readonly: bool = False):
        self.path = Path(db_path)
        if readonly:
            self.conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        else:
            self.conn = sqlite3.connect(self.path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row

        base, vec_ddl = _load_schema()
        has_vec = _try_sqlite_vec(self.conn)
        if not readonly:
            self.conn.executescript(base)
        self.index = (VecIndex if has_vec else NumpyIndex)(self.conn, vec_ddl)

    # --- metadata ---------------------------------------------------------

    def upsert(self, path: Path, mime: str, digest: bytes) -> tuple[int, bool]:
        """Insert or update a file row. Returns (file_id, needs_embedding)."""
        st = path.stat()
        row = self.conn.execute(
            "SELECT file_id, content_hash, state FROM files WHERE file_path = ?",
            (str(path),),
        ).fetchone()
        now = int(time.time())
        if row is None:
            cur = self.conn.execute(
                "INSERT INTO files(file_path, dev, inode, size, mtime_ns, mime_type,"
                " content_hash, state, last_seen) VALUES (?,?,?,?,?,?,?,'pending',?)",
                (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, mime, digest, now),
            )
            return int(cur.lastrowid), True
        self.conn.execute(
            "UPDATE files SET dev=?, inode=?, size=?, mtime_ns=?, mime_type=?,"
            " content_hash=?, last_seen=? WHERE file_id=?",
            (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, mime, digest, now, row["file_id"]),
        )
        unchanged = row["content_hash"] == digest and row["state"] == "embedded"
        return int(row["file_id"]), not unchanged

    def clone_vector_by_hash(self, file_id: int, digest: bytes) -> bool:
        """Dedup gate: reuse an embedding already computed for identical bytes."""
        row = self.conn.execute(
            "SELECT file_id FROM files WHERE content_hash = ? AND state = 'embedded'"
            " AND file_id != ? LIMIT 1",
            (digest, file_id),
        ).fetchone()
        if row is None:
            return False
        v = self.vector_of(row["file_id"])
        if v is None:
            return False
        self.put_vector(file_id, v)
        return True

    def vector_of(self, file_id: int) -> np.ndarray | None:
        if isinstance(self.index, NumpyIndex):
            row = self.conn.execute(
                "SELECT embedding FROM file_vectors_np WHERE file_id = ?", (file_id,)
            ).fetchone()
            return np.frombuffer(row[0], dtype=np.float32) if row else None
        row = self.conn.execute(
            "SELECT embedding FROM file_vectors WHERE file_id = ?", (file_id,)
        ).fetchone()
        return np.frombuffer(row[0], dtype=np.float32) if row else None

    def put_vector(self, file_id: int, v: np.ndarray) -> None:
        self.index.put(file_id, v)
        self.conn.execute(
            "UPDATE files SET state='embedded', indexed_at=? WHERE file_id=?",
            (int(time.time()), file_id),
        )

    def fail(self, file_id: int) -> None:
        self.conn.execute("UPDATE files SET state='failed' WHERE file_id=?", (file_id,))

    def forget(self, path: Path) -> None:
        row = self.conn.execute(
            "SELECT file_id FROM files WHERE file_path = ?", (str(path),)
        ).fetchone()
        if row:
            self.index.drop(row["file_id"])
            # explicit: ON DELETE CASCADE needs foreign_keys=ON, which is off by default
            self.conn.execute("DELETE FROM file_tags WHERE file_id = ?", (row["file_id"],))
            self.conn.execute("DELETE FROM files WHERE file_id = ?", (row["file_id"],))

    # --- lexical ---------------------------------------------------------

    def put_text(self, file_id: int, ocr_text: str, path_tokens: str) -> None:
        if ocr_text:
            # Append only the pieces of compound words, not a second copy of
            # the whole text. BM25 normalises by document length, so doubling
            # every field quietly penalised exactly the documents with the most
            # recognised text.
            extra = _split_extras(ocr_text)
            if extra:
                ocr_text = ocr_text + " " + " ".join(extra)
        self.conn.execute(
            "INSERT INTO file_tags(file_id, ocr_text, path_tokens) VALUES (?,?,?) "
            "ON CONFLICT(file_id) DO UPDATE SET ocr_text=excluded.ocr_text, "
            "path_tokens=excluded.path_tokens",
            (file_id, ocr_text, path_tokens),
        )

    def bm25(self, query: str, k: int, column: str) -> list[tuple[int, float]]:
        """BM25 restricted to one column. Lower is better.

        Filenames and OCR text are ranked separately and fused downstream
        (proposal.md §6.4). Scoring them as one document lets a verbose
        screenshot that happens to mention a word outrank a file actually named
        after it — BM25 cannot distinguish "curated label" from "incidental
        body text", but two rankings plus RRF can.
        """
        expr = fts_expression(query)
        if not expr:
            return []
        try:
            rows = self.conn.execute(
                "SELECT rowid, bm25(tags_fts) AS score FROM tags_fts "
                "WHERE tags_fts MATCH ? ORDER BY score LIMIT ?",
                (f"{{{column}}} : ({expr})", k),
            ).fetchall()
        except sqlite3.OperationalError:  # malformed MATCH expression
            return []
        return [(int(r["rowid"]), float(r["score"])) for r in rows]

    def paths_for(self, ids: list[int]) -> dict[int, str]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT file_id, file_path FROM files WHERE file_id IN ({q})", ids
        ).fetchall()
        return {r["file_id"]: r["file_path"] for r in rows}

    def stats(self) -> dict[str, int]:
        total = self.conn.execute("SELECT count(*) FROM files").fetchone()[0]
        out = {"files": total, "embedded": self.index.count()}
        for state, n in self.conn.execute("SELECT state, count(*) FROM files GROUP BY state"):
            out[state] = n
        return out


TOKEN_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")
WORD_RE = re.compile(r"\w+", re.UNICODE)


def _split_extras(text: str) -> list[str]:
    """Pieces of compound words only — nothing for ordinary prose."""
    out: list[str] = []
    for word in WORD_RE.findall(text):
        pieces = TOKEN_RE.findall(word)
        if len(pieces) > 1:
            out.extend(p.lower() for p in pieces)
    return out


def _variants(text: str) -> list[str]:
    """Whole words plus their camelCase/digit-run pieces, lowercased.

    Both forms are needed. FTS5's unicode61 tokenizer keeps `VectorIndex` as one
    token, so a query split into `vector`/`index` would miss it; conversely a
    path written `code_v2.jpg` only matches if it *is* split. Emitting both on
    each side makes the two meet.
    """
    out: list[str] = []
    for word in WORD_RE.findall(text):
        out.append(word.lower())
        out.extend(t.lower() for t in TOKEN_RE.findall(word))
    seen: set[str] = set()
    return [t for t in out if t and not (t in seen or seen.add(t))]


def path_tokens(path: Path, root: Path | None = None) -> str:
    """Split a path into searchable words.

    `Screenshot_2026-10-11_code.jpg` -> `screenshot 2026 10 11 code jpg`, with
    the unsplit segments kept too so an exact filename still matches.
    """
    text = str(path)
    if root is not None:
        with contextlib.suppress(ValueError):
            text = str(path.relative_to(root))
    return " ".join(_variants(text))


def fts_expression(query: str) -> str:
    """Build a safe FTS5 MATCH expression.

    Every token is quoted, so operators and punctuation in user input are data
    rather than syntax. OR rather than AND: this is one input to rank fusion,
    where recall matters more than precision.

    `OR` must stay uppercase. Lowercasing the joined expression turns the
    operator into a bareword, silently converting every multi-word query into an
    AND that also requires the literal token "or" — which matches nothing.
    """
    return " OR ".join(f'"{t}"' for t in _variants(query))


def digest_of(path: Path) -> bytes:
    """blake2b stands in for the real system's BLAKE3 — same role, stdlib only."""
    h = hashlib.blake2b(digest_size=32)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK), b""):
            h.update(chunk)
    return h.digest()
