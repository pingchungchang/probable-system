"""The OCR gate and backend selection."""

from __future__ import annotations

import unittest
from pathlib import Path

from sospoc.ocr import GATE_THRESHOLD, NoOcr, Tesseract, get_ocr
from tests.helpers import TempTree, text_png


class TestNoOcr(TempTree):
    def test_returns_nothing(self) -> None:
        self.assertEqual(NoOcr().text(self.write_image("a.png")), "")

    def test_get_ocr_honours_disabled(self) -> None:
        self.assertIsInstance(get_ocr(enabled=False), NoOcr)


class TestGate(TempTree):
    """The threshold must sit below the text range, not inside it.

    A 0.045 cutoff previously landed between code screenshots (~0.039) and the
    rest, so code was silently never OCR'd — the exact failure that motivated
    this pipeline.
    """

    def setUp(self) -> None:
        super().setUp()
        self.t = Tesseract()

    def test_dense_text_is_admitted(self) -> None:
        p = self.root / "text.png"
        p.write_bytes(text_png(lines=26))
        self.assertGreater(self.t.edge_fraction(p), GATE_THRESHOLD)
        self.assertTrue(self.t.looks_texty(p))

    def test_flat_image_is_rejected(self) -> None:
        p = self.write_image("flat.png", (120, 160, 90))
        self.assertLess(self.t.edge_fraction(p), GATE_THRESHOLD)
        self.assertFalse(self.t.looks_texty(p))

    def test_threshold_has_margin_below_the_text_range(self) -> None:
        p = self.root / "text.png"
        p.write_bytes(text_png(lines=26))
        self.assertGreater(self.t.edge_fraction(p), GATE_THRESHOLD * 1.5)

    def test_unreadable_file_is_admitted_not_crashed(self) -> None:
        bad = self.root / "bad.png"
        bad.write_bytes(b"\x89PNG\r\n\x1a\ngarbage")
        self.assertTrue(self.t.looks_texty(bad), "a gate failure must not hide the file")

    def test_gate_can_be_disabled(self) -> None:
        p = self.write_image("flat.png", (120, 160, 90))
        self.assertFalse(Tesseract(gate=True).looks_texty(p))
        t = Tesseract(gate=False)
        self.assertFalse(t.gate)

    def test_missing_binary_degrades_quietly(self) -> None:
        t = Tesseract()
        t.available = False
        self.assertEqual(t.text(Path("/nonexistent.png")), "")


if __name__ == "__main__":
    unittest.main()
