"""Ingest pipeline: filters, backfill, debounce, dedup, deletion."""

from __future__ import annotations

import unittest

from sospoc.embed import FakeEncoder
from sospoc.index import Indexer, is_candidate, mime_of
from tests.helpers import TempTree


class TestFilters(TempTree):
    def test_images_accepted(self) -> None:
        for name in ("a.jpg", "b.JPEG", "c.png", "d.webp", "e.tiff"):
            self.assertTrue(is_candidate(self.write_image(name)), name)

    def test_non_images_rejected(self) -> None:
        for name in ("notes.txt", "main.py", "archive.tar.gz"):
            self.assertFalse(is_candidate(self.write_text(name)), name)

    def test_build_dirs_skipped(self) -> None:
        for name in (".git/x.png", "node_modules/y.png", "target/z.png"):
            self.assertFalse(is_candidate(self.write_image(name)), name)

    def test_empty_file_rejected(self) -> None:
        p = self.root / "empty.png"
        p.write_bytes(b"")
        self.assertFalse(is_candidate(p))

    def test_missing_file_rejected(self) -> None:
        self.assertFalse(is_candidate(self.root / "ghost.png"))

    def test_oversized_file_rejected(self) -> None:
        import sospoc.index as idx

        p = self.write_image("big.png")
        original = idx.MAX_BYTES
        idx.MAX_BYTES = 1
        self.addCleanup(setattr, idx, "MAX_BYTES", original)
        self.assertFalse(is_candidate(p))

    def test_mime_detection(self) -> None:
        self.assertEqual(mime_of(self.write_image("a.png")), "image/png")
        self.assertIn("jpeg", mime_of(self.write_image("b.jpg")) or "")


class TestIngest(TempTree):
    def build(self) -> Indexer:
        return Indexer(self.root, self.db, encoder=FakeEncoder())

    def test_backfill_indexes_existing_images(self) -> None:
        for n in ("cat.png", "dog.png", "tree.png"):
            self.write_image(n)
        self.write_text("readme.txt")
        ix = self.build()
        ix.run(watch=False)
        self.assertEqual(ix.store.stats()["embedded"], 3, "text file must not be indexed")

    def test_backfill_is_recursive(self) -> None:
        self.write_image("top.png")
        self.write_image("sub/deep/inner.png")
        ix = self.build()
        ix.run(watch=False)
        self.assertEqual(ix.store.stats()["embedded"], 2)

    def test_rerun_is_idempotent(self) -> None:
        self.write_image("cat.png")
        self.build().run(watch=False)
        second = self.build()
        second.run(watch=False)
        self.assertEqual(second.store.stats(), {"files": 1, "embedded": 1})
        self.assertEqual(second.done, 0, "unchanged file should not be re-embedded")
        self.assertEqual(second.skipped, 1)

    def test_duplicate_bytes_skip_the_encoder(self) -> None:
        a = self.write_image("one.png", (7, 7, 7))
        (self.root / "two.png").write_bytes(a.read_bytes())
        ix = self.build()
        ix.run(watch=False)
        self.assertEqual(ix.store.stats()["embedded"], 2)
        self.assertEqual(ix.done, 1, "second copy must come from the dedup gate")
        self.assertEqual(ix.skipped, 1)

    def test_duplicates_in_the_same_batch_encode_once(self) -> None:
        """Regression: the cross-batch dedup gate cannot see within a batch.

        Both copies arrive before either is marked `embedded`, so grouping by
        content hash inside _flush is what keeps the encoder from running twice.
        """
        a = self.write_image("a.png")
        for name in ("b.png", "c.png", "d.png"):
            (self.root / name).write_bytes(a.read_bytes())
        ix = self.build()
        ix.run(watch=False)
        self.assertEqual(ix.store.stats()["embedded"], 4, "all copies must be searchable")
        self.assertEqual(ix.done, 1, "four identical files, one encode")
        self.assertEqual(ix.skipped, 3)

    def test_batching_across_many_files(self) -> None:
        for i in range(19):  # not a multiple of BATCH
            self.write_image(f"img{i}.png", (i, 0, 0))
        ix = self.build()
        ix.run(watch=False)
        self.assertEqual(ix.store.stats()["embedded"], 19)

    def test_unreadable_image_does_not_abort_the_batch(self) -> None:
        self.write_image("good.png")
        (self.root / "bad.png").write_bytes(b"\x89PNG\r\n\x1a\ngarbage")
        ix = self.build()
        ix.run(watch=False)  # FakeEncoder accepts both; real one skips the bad file
        self.assertGreaterEqual(ix.store.stats()["embedded"], 1)

    def test_undecodable_file_is_counted_as_failed(self) -> None:
        """Regression: encode failures used to vanish from every counter.

        `done + skipped` silently failed to equal the number of files seen, so a
        run could look half-finished with no indication why.
        """
        self.write_image("good.png")
        (self.root / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\ngarbage")

        class RefusesOne:
            def encode_images(self, paths):
                enc = FakeEncoder()
                return {p: v for p, v in enc.encode_images(paths).items() if p.stem != "broken"}

            def encode_text(self, t):
                return FakeEncoder().encode_text(t)

        ix = Indexer(self.root, self.db, encoder=RefusesOne())
        ix.run(watch=False)
        self.assertEqual(ix.done, 1)
        self.assertEqual(ix.failed, 1)
        self.assertEqual(ix.store.stats().get("failed"), 1)

    def test_every_file_is_accounted_for(self) -> None:
        """done + skipped + failed + dropped must equal the files considered."""
        for i in range(7):
            self.write_image(f"u{i}.png")
        a = self.write_image("dup-a.png")
        (self.root / "dup-b.png").write_bytes(a.read_bytes())
        ix = self.build()
        ix.run(watch=False)
        self.assertEqual(ix.done + ix.skipped + ix.failed + ix.dropped, 9)
        self.assertEqual(ix.done, 8, "7 unique + one representative of the dup pair")
        self.assertEqual(ix.skipped, 1)

    def test_backfill_completion_is_announced(self) -> None:
        self.write_image("cat.png")
        ix = self.build()
        with self.assertLogs("sospoc.index", level="INFO") as log:
            ix.run(watch=False)
        self.assertTrue(
            any("backfill complete" in m for m in log.output),
            f"no completion line in {log.output}",
        )
        self.assertTrue(ix.backfilled)

    def test_empty_directory(self) -> None:
        ix = self.build()
        ix.run(watch=False)
        self.assertEqual(ix.store.stats(), {"files": 0, "embedded": 0})


if __name__ == "__main__":
    unittest.main()
