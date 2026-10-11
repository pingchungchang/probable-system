"""Query path: ranking, liveness, timings, rendering."""

from __future__ import annotations

import unittest
from pathlib import Path

from sospoc.embed import FakeEncoder
from sospoc.index import Indexer
from sospoc.search import Searcher, render
from tests.helpers import TempTree


class TestSearch(TempTree):
    def setUp(self) -> None:
        super().setUp()
        for name in ("cat.png", "dog.png", "bicycle.png", "harbour.png"):
            self.write_image(name)
        Indexer(self.root, self.db, encoder=FakeEncoder()).run(watch=False)
        self.s = Searcher(self.db, encoder=FakeEncoder())

    def test_exact_concept_ranks_first(self) -> None:
        # FakeEncoder keys on the filename stem, so "cat" must retrieve cat.png.
        top = self.s.query("cat", k=4).hits[0]
        self.assertEqual(Path(top.path).name, "cat.png")
        # score is now the fused RRF value; the cosine lives on .dense
        self.assertAlmostEqual(top.dense, 1.0, places=4)
        self.assertGreater(top.score, 0)

    def test_k_is_respected(self) -> None:
        self.assertEqual(len(self.s.query("cat", k=2).hits), 2)

    def test_k_above_corpus_size(self) -> None:
        self.assertEqual(len(self.s.query("cat", k=99).hits), 4)

    def test_results_sorted_descending(self) -> None:
        scores = [h.score for h in self.s.query("dog", k=4).hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_timings_are_populated_and_consistent(self) -> None:
        r = self.s.query("cat")
        self.assertGreater(r.encode_ms, 0)
        self.assertGreaterEqual(r.knn_ms, 0)
        self.assertGreaterEqual(r.bm25_ms, 0)
        self.assertAlmostEqual(r.total_ms, r.encode_ms + r.knn_ms + r.bm25_ms, places=6)

    def test_deleted_file_is_flagged_not_hidden(self) -> None:
        (self.root / "cat.png").unlink()
        hit = next(h for h in self.s.query("cat", k=4).hits if h.path.endswith("cat.png"))
        self.assertFalse(hit.alive, "a vanished file must be reported as missing")

    def test_all_hits_alive_when_nothing_removed(self) -> None:
        self.assertTrue(all(h.alive for h in self.s.query("dog", k=4).hits))

    def test_empty_index_returns_nothing(self) -> None:
        empty = self.root / "empty.db"
        Indexer(self.root / "nonexistent-sub", empty, encoder=FakeEncoder())
        s = Searcher(empty, encoder=FakeEncoder())
        self.assertEqual(s.query("anything").hits, [])

    def test_search_does_not_lock_out_the_indexer(self) -> None:
        """Read-only searcher and a live writer must coexist (WAL)."""
        ix = Indexer(self.root, self.db, encoder=FakeEncoder())
        self.write_image("late.png")
        ix.run(watch=False)
        self.assertEqual(len(self.s.query("late", k=5).hits), 5)


class TestFusedChannels(TempTree):
    """Which channel found a hit, and that both can contribute."""

    def setUp(self) -> None:
        super().setUp()
        from sospoc.store import Store, path_tokens

        for name in ("cat.png", "dog.png", "invoice.png"):
            self.write_image(name)
        Indexer(self.root, self.db, encoder=FakeEncoder(), use_ocr=False).run(watch=False)
        # give one file OCR text no filename mentions
        st = Store(self.db)
        fid = st.conn.execute(
            "SELECT file_id FROM files WHERE file_path LIKE '%invoice.png'"
        ).fetchone()["file_id"]
        st.put_text(
            fid,
            "TOTAL DUE 4218 EUR acme logistics",
            path_tokens(self.root / "invoice.png", self.root),
        )
        st.conn.close()
        self.s = Searcher(self.db, encoder=FakeEncoder())

    def test_filename_match_reports_both_channels(self) -> None:
        top = self.s.query("cat", k=3).hits[0]
        self.assertEqual(Path(top.path).name, "cat.png")
        self.assertEqual(top.channels, "dp", "filename should hit dense and the path channel")

    def test_ocr_only_match_is_found_by_sparse(self) -> None:
        """The original complaint: content visible only inside the image."""
        hits = self.s.query("acme logistics", k=3).hits
        self.assertEqual(Path(hits[0].path).name, "invoice.png")
        self.assertIsNotNone(hits[0].sparse)

    def test_ocr_only_match_is_invisible_to_dense_alone(self) -> None:
        from sospoc.store import Store

        st = Store(self.db, readonly=True)
        dense = st.index.knn(FakeEncoder().encode_text("acme logistics"), 3)
        top_dense = st.paths_for([dense[0][0]])[dense[0][0]]
        self.assertNotIn("invoice", top_dense, "dense should not have found this")


class TestRender(TempTree):
    def setUp(self) -> None:
        super().setUp()
        self.write_image("cat.png")
        Indexer(self.root, self.db, encoder=FakeEncoder()).run(watch=False)
        self.s = Searcher(self.db, encoder=FakeEncoder())

    def test_plain_render_lists_results(self) -> None:
        out = render(self.s.query("cat"))
        self.assertIn("cat.png", out)
        self.assertIn("1.", out)

    def test_timings_suffix_is_opt_in(self) -> None:
        r = self.s.query("cat")
        self.assertNotIn("encode", render(r))
        self.assertIn("encode", render(r, show_timings=True))

    def test_relative_paths(self) -> None:
        out = render(self.s.query("cat"), root=self.root)
        self.assertIn("cat.png", out)
        self.assertNotIn(str(self.root), out)

    def test_missing_marker(self) -> None:
        (self.root / "cat.png").unlink()
        self.assertIn("[missing]", render(self.s.query("cat")))

    def test_no_results_message(self) -> None:
        from sospoc.search import Result

        self.assertIn("no results", render(Result([], 1.0, 1.0)))


if __name__ == "__main__":
    unittest.main()
