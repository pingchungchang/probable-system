#!/usr/bin/env python3
"""Work through eval/queries.txt and score it (PLAN.md §6.3).

Loads CLIP once, shows the top 5 for each query, and asks whether at least one
result is something you would have accepted had you been searching by hand.
Tallies by kind, because *which* kinds fail is more informative than the count,
and records p50 search latency for the §8 baseline.

    python eval/score.py --db sos.db
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sospoc.search import Searcher, render  # noqa: E402

KIND_RE = re.compile(r"^#\s*---\s*(.+?)\s*---\s*$")
QUERY_RE = re.compile(r"^\s*\d+\.\s*(.+?)\s*$")

VERDICTS = [
    (16, "PREMISE HOLDS — proceed to Stage 2 and the full plan."),
    (
        10,
        "WORKABLE, NOT MAGICAL — hybrid retrieval is load-bearing. Move RRF tuning\n"
        "         earlier and lower the recall@10 target before it reaches a slide.",
    ),
    (
        0,
        "STOP AND RECONSIDER — check B/16 vs B/32 and L2 normalisation before\n"
        "         concluding anything. Otherwise proposal §12.2 has arrived early.",
    ),
]


def parse(path: Path) -> list[tuple[str, str]]:
    kind, out = "unsorted", []
    for line in path.read_text().splitlines():
        if m := KIND_RE.match(line):
            kind = m.group(1)
        elif m := QUERY_RE.match(line):
            out.append((kind, m.group(1)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=Path("sos.db"))
    ap.add_argument("--queries", type=Path, default=Path(__file__).with_name("queries.txt"))
    ap.add_argument("-n", type=int, default=5, help="results to judge")
    ap.add_argument("--relative-to", type=Path, default=None)
    a = ap.parse_args()

    queries = parse(a.queries)
    if not queries:
        ap.error(
            f"no queries in {a.queries}. Fill it in BEFORE searching anything — "
            "writing queries after seeing results measures nothing (PLAN.md §8 P4)."
        )

    s = Searcher(a.db)
    tally: dict[str, list[int]] = {}
    latencies: list[float] = []

    for i, (kind, q) in enumerate(queries, 1):
        r = s.query(q, a.n)
        latencies.append(r.total_ms)
        print(f"\n[{i}/{len(queries)}] ({kind})  {q!r}")
        print(render(r, show_timings=False, root=a.relative_to))
        while (ans := input("    acceptable hit in these? [y/n/q] ").strip().lower()) not in (
            "y",
            "n",
            "q",
        ):
            pass
        if ans == "q":
            break
        tally.setdefault(kind, []).append(1 if ans == "y" else 0)

    print("\n" + "=" * 60)
    total = hits = 0
    for kind, scores in tally.items():
        print(f"  {kind:<24} {sum(scores)} / {len(scores)}")
        hits += sum(scores)
        total += len(scores)
    print(f"  {'TOTAL':<24} {hits} / {total}")

    if latencies:
        print(f"\n  search latency p50: {statistics.median(latencies):.1f} ms (excl. startup)")
        print(f"  vector backend:     {s.store.index.name}")
        print(f"  indexed:            {s.store.stats()['embedded']} images")

    scaled = round(hits * 20 / total) if total else 0
    for threshold, verdict in VERDICTS:
        if scaled >= threshold:
            print(f"\n  {scaled}/20 → {verdict}")
            break
    print("\n  Record this in eval/results.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
