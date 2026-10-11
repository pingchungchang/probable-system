"""FakeEncoder contract, plus the real CLIP model when explicitly requested.

The CLIP test is skipped unless SOS_TEST_CLIP=1 — it downloads ~600 MB of
weights, which has no place in a routine test run or in CI.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from sospoc import DIM
from sospoc.embed import FakeEncoder, check_dim, get_encoder
from tests.helpers import TempTree


class TestFakeEncoder(TempTree):
    def setUp(self) -> None:
        super().setUp()
        self.enc = FakeEncoder()

    def test_text_vector_shape_and_norm(self) -> None:
        v = self.enc.encode_text("a cat")
        self.assertEqual(v.shape, (DIM,))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_deterministic(self) -> None:
        np.testing.assert_array_equal(self.enc.encode_text("a cat"), self.enc.encode_text("a cat"))

    def test_case_and_whitespace_insensitive(self) -> None:
        np.testing.assert_array_equal(
            self.enc.encode_text("  A Cat "), self.enc.encode_text("a cat")
        )

    def test_distinct_strings_differ(self) -> None:
        a, b = self.enc.encode_text("cat"), self.enc.encode_text("dog")
        self.assertLess(abs(float(a @ b)), 0.5, "unrelated strings should not align")

    def test_image_vector_keyed_on_stem(self) -> None:
        p = self.write_image("cat.png")
        got = self.enc.encode_images([p])[p]
        np.testing.assert_allclose(got, self.enc.encode_text("cat"), atol=1e-6)

    def test_missing_file_skipped_not_raised(self) -> None:
        self.assertEqual(self.enc.encode_images([self.root / "nope.png"]), {})

    def test_batch_returns_one_vector_each(self) -> None:
        ps = [self.write_image(f"i{i}.png") for i in range(4)]
        self.assertEqual(len(self.enc.encode_images(ps)), 4)


class TestEncoderSelection(unittest.TestCase):
    def test_env_selects_fake(self) -> None:
        os.environ["SOS_FAKE_ENCODER"] = "1"
        self.addCleanup(os.environ.pop, "SOS_FAKE_ENCODER", None)
        self.assertIsInstance(get_encoder(), FakeEncoder)


class TestCheckDim(unittest.TestCase):
    def test_accepts_correct_width(self) -> None:
        v = np.zeros(DIM, dtype=np.float32)
        self.assertIs(check_dim(v), v)

    def test_rejects_wrong_width(self) -> None:
        with self.assertRaises(SystemExit):
            check_dim(np.zeros(384, dtype=np.float32))


@unittest.skipUnless(os.environ.get("SOS_TEST_CLIP") == "1", "SOS_TEST_CLIP!=1")
class TestRealClip(TempTree):
    """The assumptions about open_clip that nothing else verifies."""

    @classmethod
    def setUpClass(cls) -> None:
        from sospoc.embed import ClipCpu

        cls.enc = ClipCpu(threads=2)

    def test_text_tower_width_matches_schema(self) -> None:
        self.assertEqual(self.enc.encode_text("a cat").shape, (DIM,))

    def test_image_tower_width_matches_schema(self) -> None:
        p = self.write_image("x.png", (120, 30, 30))
        self.assertEqual(self.enc.encode_images([p])[p].shape, (DIM,))

    def test_outputs_are_normalised(self) -> None:
        v = self.enc.encode_text("a photograph of a bicycle")
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=4)

    def test_corrupt_image_is_skipped(self) -> None:
        bad = self.root / "truncated.png"
        bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)
        self.assertEqual(self.enc.encode_images([bad]), {})


if __name__ == "__main__":
    unittest.main()
