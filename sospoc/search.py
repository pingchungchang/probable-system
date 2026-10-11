"""Query side: encode, kNN, rank, print.

Timings are split so the §8 baseline number excludes process startup — loading
CLIP costs seconds and has nothing to do with search latency. Use --repl when
working through the twenty evaluation queries; it loads the model once.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .embed import get_encoder
from .store import Store

RRF_K = 60  # proposal.md §6.4


@dataclass
class Hit:
    path: str
    score: float  # fused RRF score
    alive: bool
    dense: float | None = None  # cosine similarity
    path_score: float | None = None  # bm25 over the filename
    ocr_score: float | None = None  # bm25 over recognised in-image text

    @property
    def sparse(self) -> float | None:
        """Best of the two lexical channels (bm25 is negative; lower is better)."""
        vals = [v for v in (self.path_score, self.ocr_score) if v is not None]
        return min(vals) if vals else None

    @property
    def channels(self) -> str:
        return "".join(
            c
            for c, v in (("d", self.dense), ("p", self.path_score), ("o", self.ocr_score))
            if v is not None
        )


@dataclass
class Result:
    hits: list[Hit]
    encode_ms: float
    knn_ms: float
    bm25_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.encode_ms + self.knn_ms + self.bm25_ms


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion: sum of 1/(k + rank) across rankings.

    Rank-based rather than score-based on purpose. CLIP cosines sit in a narrow,
    query-dependent band and BM25 is unbounded and negative in SQLite; the two
    are not comparable as numbers, but their orderings are.
    """
    out: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            out[doc] = out.get(doc, 0.0) + 1.0 / (k + rank)
    return out


class Searcher:
    def __init__(self, db: Path, threads: int | None = None, encoder=None):
        self.store = Store(db, readonly=True)
        self.clip = encoder if encoder is not None else get_encoder(threads)

    def query(self, text: str, k: int = 10, pool: int | None = None) -> Result:
        """Dense kNN fused with lexical BM25 by RRF (proposal.md §6.4).

        Each channel retrieves a pool several times larger than k, because a
        document ranked 30th by one channel and 2nd by the other should still
        surface — which is the entire point of fusing them.
        """
        pool = pool or max(k * 5, 50)

        t0 = time.perf_counter()
        q = self.clip.encode_text(text)
        t1 = time.perf_counter()
        dense = self.store.index.knn(q, pool)
        t2 = time.perf_counter()
        by_path = self.store.bm25(text, pool, "path_tokens")
        by_ocr = self.store.bm25(text, pool, "ocr_text")
        t3 = time.perf_counter()

        dense_scores, path_scores, ocr_scores = dict(dense), dict(by_path), dict(by_ocr)
        fused = rrf(
            [
                [fid for fid, _ in dense],
                [fid for fid, _ in by_path],
                [fid for fid, _ in by_ocr],
            ]
        )
        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], -dense_scores.get(kv[0], 0.0)))

        paths = self.store.paths_for([fid for fid, _ in ordered[:k]])
        hits = []
        for fid, score in ordered[:k]:
            p = paths.get(fid)
            if p is None:
                continue  # row vanished between the two reads
            hits.append(
                Hit(
                    p,
                    score,
                    os.path.exists(p),
                    dense_scores.get(fid),
                    path_scores.get(fid),
                    ocr_scores.get(fid),
                )
            )
        return Result(hits, (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000)


def render(r: Result, show_timings: bool = False, root: Path | None = None) -> str:
    lines = []
    for i, h in enumerate(r.hits, 1):
        p = h.path
        if root:
            with contextlib.suppress(ValueError):  # hit outside root: keep absolute
                p = str(Path(p).relative_to(root))
        mark = "" if h.alive else "  [missing]"
        lines.append(f"{i:2}. {h.score:.4f} [{h.channels:<3}]  {p}{mark}")
    if not lines:
        lines.append("(no results — is the index still building?)")
    if show_timings:
        lines.append("")
        lines.append(
            f"    encode {r.encode_ms:.1f} ms · knn {r.knn_ms:.1f} ms · "
            f"bm25 {r.bm25_ms:.1f} ms · total {r.total_ms:.1f} ms (excludes startup)"
        )
        lines.append("    channels: d=dense (CLIP)  p=filename  o=in-image text")
    return "\n".join(lines)
