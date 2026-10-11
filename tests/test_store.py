"""Store and vector-index behaviour, exercised against whichever backend is active.

`nix flake check` runs this file twice — once with SOS_FORCE_NUMPY=1 and once
with sqlite-vec — so the two backends are held to the same contract.
"""

from __future__ import annotations

import sqlite3
import unittest

import numpy as np

from sospoc import DIM
from sospoc.store import NumpyIndex, Store, VecIndex, digest_of
from tests.helpers import TempTree


def unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class TestStore(TempTree):
    def setUp(self) -> None:
        super().setUp()
        self.store = Store(self.db)

    def add(self, name: str, seed: int) -> tuple[int, np.ndarray]:
        p = self.write_image(name, (seed % 255, 0, 0))
        fid, needs = self.store.upsert(p, "image/png", digest_of(p))
        self.assertTrue(needs, "a brand-new file must be queued for embedding")
        v = unit(seed)
        self.store.put_vector(fid, v)
        return fid, v

    def test_backend_is_one_of_the_two(self) -> None:
        self.assertIsInstance(self.store.index, (VecIndex, NumpyIndex))

    def test_schema_applies_and_starts_empty(self) -> None:
        self.assertEqual(self.store.stats(), {"files": 0, "embedded": 0})

    def test_insert_and_self_match(self) -> None:
        fid, v = self.add("a.png", 1)
        self.add("b.png", 2)
        hits = self.store.index.knn(v, 2)
        self.assertEqual(hits[0][0], fid)
        self.assertAlmostEqual(hits[0][1], 1.0, places=4)

    def test_knn_sorted_descending(self) -> None:
        for i in range(6):
            self.add(f"f{i}.png", i + 10)
        hits = self.store.index.knn(unit(99), 5)
        self.assertEqual(len(hits), 5)
        scores = [s for _, s in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_knn_k_larger_than_corpus(self) -> None:
        self.add("only.png", 7)
        self.assertEqual(len(self.store.index.knn(unit(7), 50)), 1)

    def test_knn_on_empty_index(self) -> None:
        self.assertEqual(self.store.index.knn(unit(0), 5), [])

    def test_unchanged_file_is_not_requeued(self) -> None:
        p = self.write_image("c.png", (9, 9, 9))
        fid, needs = self.store.upsert(p, "image/png", digest_of(p))
        self.assertTrue(needs)
        self.store.put_vector(fid, unit(3))
        again, needs = self.store.upsert(p, "image/png", digest_of(p))
        self.assertEqual(fid, again)
        self.assertFalse(needs, "identical content must not be re-embedded")

    def test_changed_content_is_requeued(self) -> None:
        p = self.write_image("d.png", (1, 1, 1))
        fid, _ = self.store.upsert(p, "image/png", digest_of(p))
        self.store.put_vector(fid, unit(4))
        p.write_bytes(p.read_bytes() + b"\x00")  # same path, new bytes
        again, needs = self.store.upsert(p, "image/png", digest_of(p))
        self.assertEqual(fid, again)
        self.assertTrue(needs)

    def test_dedup_gate_clones_vector(self) -> None:
        fid_a, v = self.add("orig.png", 5)
        dup = self.root / "copy.png"
        dup.write_bytes((self.root / "orig.png").read_bytes())
        fid_b, needs = self.store.upsert(dup, "image/png", digest_of(dup))
        self.assertTrue(needs)
        self.assertTrue(self.store.clone_vector_by_hash(fid_b, digest_of(dup)))
        np.testing.assert_allclose(self.store.vector_of(fid_b), v, atol=1e-6)
        self.assertEqual(self.store.stats()["embedded"], 2)

    def test_dedup_gate_misses_on_novel_content(self) -> None:
        self.add("x.png", 6)
        p = self.write_image("y.png", (200, 1, 1))
        fid, _ = self.store.upsert(p, "image/png", digest_of(p))
        self.assertFalse(self.store.clone_vector_by_hash(fid, digest_of(p)))

    def test_forget_removes_row_and_vector(self) -> None:
        self.add("gone.png", 8)
        self.assertEqual(self.store.stats(), {"files": 1, "embedded": 1})
        self.store.forget(self.root / "gone.png")
        self.assertEqual(self.store.stats(), {"files": 0, "embedded": 0})

    def test_forget_unknown_path_is_a_noop(self) -> None:
        self.store.forget(self.root / "never-existed.png")  # must not raise

    def test_paths_for_round_trip(self) -> None:
        fid, _ = self.add("named.png", 11)
        self.assertEqual(self.store.paths_for([fid])[fid], str(self.root / "named.png"))
        self.assertEqual(self.store.paths_for([]), {})

    def test_fail_marks_state(self) -> None:
        p = self.write_image("bad.png")
        fid, _ = self.store.upsert(p, "image/png", digest_of(p))
        self.store.fail(fid)
        row = self.store.conn.execute("SELECT state FROM files WHERE file_id=?", (fid,)).fetchone()
        self.assertEqual(row["state"], "failed")

    def test_vector_survives_reopen(self) -> None:
        fid, v = self.add("persist.png", 12)
        self.store.conn.close()
        reopened = Store(self.db)
        np.testing.assert_allclose(reopened.vector_of(fid), v, atol=1e-6)

    def test_readonly_store_rejects_writes(self) -> None:
        self.add("ro.png", 13)
        self.store.conn.close()
        ro = Store(self.db, readonly=True)
        self.assertEqual(ro.stats()["files"], 1)
        with self.assertRaises(sqlite3.OperationalError):
            ro.conn.execute("DELETE FROM files")


class TestDigest(TempTree):
    def test_same_bytes_same_digest(self) -> None:
        a = self.write_text("a.txt", "identical")
        b = self.write_text("b.txt", "identical")
        self.assertEqual(digest_of(a), digest_of(b))

    def test_different_bytes_differ(self) -> None:
        a = self.write_text("a.txt", "one")
        b = self.write_text("b.txt", "two")
        self.assertNotEqual(digest_of(a), digest_of(b))

    def test_digest_is_32_bytes(self) -> None:
        self.assertEqual(len(digest_of(self.write_text("a.txt"))), 32)


if __name__ == "__main__":
    unittest.main()
