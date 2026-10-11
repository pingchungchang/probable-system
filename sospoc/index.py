"""Watch a directory, embed images as they settle.

Stage 1 uses watchdog's inotify `on_closed`, which is the same signal the real
driver takes from FUSE `release`. Stage 2 replaces `_watch_inotify` with a FIFO
reader and changes nothing else — that is the whole point of keeping the event
source behind one function (PLAN.md §3).
"""

from __future__ import annotations

import logging
import mimetypes
import os
import queue
import sys
import time
from pathlib import Path

from .embed import check_dim, get_encoder
from .ocr import get_ocr
from .store import Store, digest_of, path_tokens

log = logging.getLogger(__name__)

DEBOUNCE_S = float(os.environ.get("SOS_DEBOUNCE", "3.0"))
BATCH = 8
MAX_BYTES = 64 << 20
SKIP_DIRS = {".git", "node_modules", "target", "__pycache__", ".venv", ".cache"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def mime_of(path: Path) -> str | None:
    """Extension first; python-magic only if it happens to be installed."""
    guess, _ = mimetypes.guess_type(path.name)
    if guess:
        return guess
    try:
        import magic

        return magic.from_file(str(path), mime=True)
    except Exception:
        return None


def is_candidate(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    if path.suffix.lower() not in IMAGE_EXT:
        return False
    try:
        return 0 < path.stat().st_size <= MAX_BYTES
    except OSError:
        return False


class Indexer:
    def __init__(
        self,
        root: Path,
        db: Path,
        threads: int | None = None,
        encoder=None,
        ocr=None,
        use_ocr: bool = True,
    ):
        self.root = root.resolve()
        self.store = Store(db)
        self.clip = encoder if encoder is not None else get_encoder(threads)
        self.ocr = ocr if ocr is not None else get_ocr(enabled=use_ocr)
        self.events: queue.Queue[tuple[str, Path]] = queue.Queue()
        self.pending: dict[Path, float] = {}
        self.done = 0
        self.skipped = 0
        self.ocred = 0  # yielded recognised text
        self.failed = 0  # could not be decoded
        self.dropped = 0  # vanished or filtered out between queue and flush
        self.backfilled = False
        self.t0 = time.monotonic()

    # --- event sources ----------------------------------------------------

    def backfill(self) -> None:
        n = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                p = Path(dirpath) / name
                if is_candidate(p):
                    self.events.put(("put", p))
                    n += 1
        log.info("backfill queued %d images", n)

    def _watch_inotify(self) -> object:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        put, drop = self.events.put, self.events.put
        outer = self

        class H(FileSystemEventHandler):
            def on_closed(self, e):  # inotify IN_CLOSE_WRITE — the `release` analogue
                if not e.is_directory:
                    put(("put", Path(e.src_path)))

            def on_created(self, e):  # belt and braces; debounce collapses duplicates
                if not e.is_directory:
                    put(("put", Path(e.src_path)))

            def on_modified(self, e):
                if not e.is_directory:
                    put(("put", Path(e.src_path)))

            def on_deleted(self, e):
                if not e.is_directory:
                    drop(("del", Path(e.src_path)))

            def on_moved(self, e):
                # PoC treats a move as delete + add. The real system re-hashes
                # instead, because atomic-save editors rename over the target
                # (proposal §6.1) — out of scope here.
                drop(("del", Path(e.src_path)))
                if not e.is_directory:
                    put(("put", Path(e.dest_path)))

            def on_any_event(self, e):
                outer.last_event = time.monotonic()

        obs = Observer()
        obs.schedule(H(), str(self.root), recursive=True)
        obs.start()
        return obs

    # --- ingest -----------------------------------------------------------

    def _flush(self, paths: list[Path]) -> None:
        """Embed a batch.

        Files are grouped by content hash first. Two identical files arriving in
        the same batch must cost one encode, not two — the cross-batch dedup gate
        cannot catch them, because neither is marked `embedded` yet when the
        other is checked.
        """
        groups: dict[bytes, list[tuple[int, Path]]] = {}
        for p in paths:
            if not p.exists() or not is_candidate(p):
                self.dropped += 1
                continue
            mime = mime_of(p) or "application/octet-stream"
            try:
                digest = digest_of(p)
                fid, needs = self.store.upsert(p, mime, digest)
            except OSError as exc:
                log.warning("stat/hash failed %s: %s", p, exc)
                self.failed += 1
                continue
            if not needs:
                self.skipped += 1
                continue
            if self.store.clone_vector_by_hash(fid, digest):
                # identical bytes, already embedded; but the path is new, and a
                # filename is per-path information the cloned vector lacks.
                self._index_text(fid, p, reuse_ocr_from=digest)
                self.skipped += 1
                continue
            groups.setdefault(digest, []).append((fid, p))

        if not groups:
            return

        reps = [entries[0][1] for entries in groups.values()]
        t = time.monotonic()
        vectors = self.clip.encode_images(reps)
        per_image = (time.monotonic() - t) / len(reps)

        for entries in groups.values():
            v = vectors.get(entries[0][1])
            if v is None:  # unreadable, truncated, or not actually an image
                for fid, _ in entries:
                    self.store.fail(fid)
                    self.failed += 1
                continue
            v = check_dim(v)
            text = self.ocr.text(entries[0][1])
            if text:
                self.ocred += 1
            for fid, p in entries:
                self.store.put_vector(fid, v)
                self._index_text(fid, p, ocr_text=text)
            self.done += 1
            self.skipped += len(entries) - 1  # same bytes, same batch, one encode

        self._progress(per_image)

    def _index_text(
        self, file_id: int, path: Path, ocr_text: str = "", reuse_ocr_from: bytes | None = None
    ) -> None:
        """Path tokens are per-path; OCR text is per-content, so it can be reused."""
        if reuse_ocr_from is not None:
            row = self.store.conn.execute(
                "SELECT t.ocr_text FROM file_tags t JOIN files f USING(file_id) "
                "WHERE f.content_hash = ? AND t.ocr_text != '' LIMIT 1",
                (reuse_ocr_from,),
            ).fetchone()
            ocr_text = row["ocr_text"] if row else ""
        self.store.put_text(file_id, ocr_text, path_tokens(path, self.root))

    def _progress(self, per_image: float) -> None:
        q = self.events.qsize() + len(self.pending)
        if not sys.stderr.isatty():
            return
        bad = f"  failed {self.failed}" if self.failed else ""
        ocr = f"  ocr {self.ocred}" if self.ocred else ""
        print(
            f"\rindexed {self.done}  deduped/unchanged {self.skipped}{bad}{ocr}  "
            f"queued {q}  {per_image * 1000:.0f} ms/img",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def run(self, watch: bool = True) -> None:
        self.backfill()
        obs = self._watch_inotify() if watch else None
        try:
            while True:
                deadline = time.monotonic() + 0.5
                while time.monotonic() < deadline:
                    try:
                        kind, path = self.events.get(timeout=0.1)
                    except queue.Empty:
                        break
                    if kind == "del":
                        self.store.forget(path)
                        self.pending.pop(path, None)
                    else:
                        self.pending[path] = time.monotonic() + DEBOUNCE_S

                now = time.monotonic()
                ready = [p for p, t in self.pending.items() if now >= t]
                for p in ready:
                    self.pending.pop(p, None)
                for i in range(0, len(ready), BATCH):
                    self._flush(ready[i : i + BATCH])

                drained = not self.pending and self.events.empty()
                if drained and not self.backfilled:
                    self.backfilled = True
                    self._summarise("backfill complete")
                    if watch:
                        log.info(
                            "watching %s for changes — Ctrl-C to stop "
                            "(use --once to exit after backfill)",
                            self.root,
                        )
                if not watch and drained:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            if obs:
                obs.stop()
                obs.join()
            if sys.stderr.isatty():
                print(file=sys.stderr)
            self._summarise("stopped")

    def _summarise(self, label: str) -> None:
        s = self.store.stats()
        seen = self.done + self.skipped + self.failed + self.dropped
        log.info(
            "%s: %d encoded, %d deduped/unchanged, %d failed, %d dropped "
            "(%d accounted for) \u2014 %d rows, %d embedded, %d with text, "
            "%.0fs, backend=%s, ocr=%s",
            label,
            self.done,
            self.skipped,
            self.failed,
            self.dropped,
            seen,
            s["files"],
            s["embedded"],
            self.ocred,
            time.monotonic() - self.t0,
            self.store.index.name,
            self.ocr.name,
        )
        if s.get("failed"):
            log.warning(
                "%d file(s) could not be decoded and are not searchable; list them with: "
                "sqlite3 <db> \"SELECT file_path FROM files WHERE state='failed'\"",
                s["failed"],
            )
